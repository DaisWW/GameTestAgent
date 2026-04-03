from __future__ import annotations

"""多维度 Bug 检测节点。

遍历 checkers/ 中注册的所有 BugChecker，收集 BugReport，
统一持久化到 ExperiencePool，critical 级别自动触发恢复流程。
"""

import logging
import os
from typing import List, Set, Tuple, TYPE_CHECKING

from .checkers import get_enabled_checkers
from .checkers.base import BugChecker, BugReport
from core.types import BugSeverity

if TYPE_CHECKING:
    from core.agent.worker import LangGraphWorker

logger = logging.getLogger(__name__)


def _save_bug_screenshot(state: dict, worker: "LangGraphWorker", bug: BugReport) -> str:
    """保存 Bug 截图，返回文件路径。"""
    page_hash  = state.get("page_hash", "")
    screenshot = state.get("screenshot")
    if not page_hash or screenshot is None:
        return ""
    shot_path = os.path.join(
        state.get("run_dir", worker.config.run_dir),
        f"bug_{bug.category}_{page_hash[:8]}_step{state.get('step', 0):03d}.png",
    )
    os.makedirs(os.path.dirname(shot_path), exist_ok=True)
    screenshot.save(shot_path)
    return shot_path


def _persist_bug(bug: BugReport, state: dict, worker: "LangGraphWorker") -> None:
    """将 BugReport 写入 ExperiencePool。"""
    page_hash = state.get("page_hash", "")
    if not page_hash:
        return
    worker.memory.experience.save_bug(
        page_hash=page_hash,
        description=bug.description,
        tags=bug.tags,
        severity=bug.severity,
        category=bug.category,
        evidence=bug.evidence,
    )


def make_node(worker: "LangGraphWorker"):
    """多维度 Bug 检测 + Bug 快照保存 + 触发恢复子图。"""

    # 在节点创建时根据配置实例化启用的 checker（仅一次）
    checkers: List[BugChecker] = get_enabled_checkers(worker.config)
    logger.info("已注册 %d 个 Bug 检测器: %s",
                len(checkers), [c.name for c in checkers])

    # 去重集合：(page_hash, category, 首个tag) — 同页面同类 Bug 只报一次
    _reported: Set[Tuple[str, str, str]] = set()

    def _dedup_key(page_hash: str, bug: BugReport) -> Tuple[str, str, str]:
        tag0 = bug.tags[0] if bug.tags else ""
        return (page_hash, bug.category, tag0)

    def node(state):
        if state.get("done"):
            return {}

        step = state.get("step", 0)
        page_hash = state.get("page_hash", "")
        all_bugs: List[BugReport] = []

        # ── 遍历所有 checker ────────────────────────────────────
        for checker in checkers:
            try:
                if not checker.is_applicable(state):
                    continue
                found = checker.check(state, worker)
                if found:
                    all_bugs.extend(found)
                    for bug in found:
                        logger.warning(
                            "[CHECK][%s] %s — %s",
                            bug.severity.upper(), bug.category, bug.description[:80],
                        )
            except Exception as exc:
                logger.warning("Checker '%s' 执行失败（忽略）: %s", checker.name, exc)

        if not all_bugs:
            return {}

        # ── 去重：同页面同类 Bug 只持久化一次 ─────────────────────
        new_bugs = [b for b in all_bugs if _dedup_key(page_hash, b) not in _reported]
        for b in new_bugs:
            _reported.add(_dedup_key(page_hash, b))

        logger.info("[CHECK] Step %02d: 发现 %d 个问题（新增 %d）",
                    step, len(all_bugs), len(new_bugs))

        # ── 统一持久化（仅新增 Bug）──────────────────────────────
        for bug in new_bugs:
            _save_bug_screenshot(state, worker, bug)
            _persist_bug(bug, state, worker)

        # ── critical 级别触发恢复 ───────────────────────────────
        critical_bugs = [b for b in all_bugs if b.severity == BugSeverity.CRITICAL]
        if critical_bugs:
            reason = critical_bugs[0].description
            logger.error("[CHECK] 发现 critical 级 Bug，触发恢复: %s", reason[:80])
            recovered = worker.run_recovery(reason)
            if not recovered:
                return {"done": True, "result": reason}
            logger.info("[CHECK] 恢复成功，继续测试")

        return {}

    return node
