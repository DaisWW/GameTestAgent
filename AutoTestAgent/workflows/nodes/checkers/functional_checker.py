from __future__ import annotations

"""维度 5：功能异常检测 (Functional Checker)

基于当前动作执行结果 + UI 元素状态分析：
- 按钮无响应：tap 操作后 page_hash 未变且元素仍在
- 死胡同页面：当前页面无任何可交互元素
- 无返回路径：非首页且无关闭 / 返回按钮
"""

import logging
from typing import List, TYPE_CHECKING

from .base import BugChecker, BugReport
from core.types import ActionType, BugCategory, BugSeverity, BugTag, ElementType

if TYPE_CHECKING:
    from core.agent.worker import LangGraphWorker

logger = logging.getLogger(__name__)

_CLICKABLE_TYPES = {ElementType.BUTTON, ElementType.ICON, ElementType.INPUT}
_BACK_KEYWORDS = {"返回", "关闭", "back", "close", "取消", "cancel", "×", "X"}


class FunctionalChecker(BugChecker):
    name = "functional"

    def check(self, state: dict, worker: "LangGraphWorker") -> List[BugReport]:
        bugs: List[BugReport] = []
        elements = state.get("ui_elements", [])
        action = state.get("current_action") or {}

        clickable = [e for e in elements if e.get("type") in _CLICKABLE_TYPES]

        # ── 死胡同检测：无任何可交互元素 ────────────────────────
        # ABA 循环页面（加载屏循环）不视为真实死胡同
        _anomaly = ""
        _packet = state.get("context_packet")
        if _packet is not None:
            _anomaly = getattr(_packet, "anomaly_flag", "") or ""
        _in_loading_loop = "ABA" in _anomaly or "循环" in _anomaly

        if elements and not clickable and not _in_loading_loop:
            bugs.append(BugReport(
                category=BugCategory.FUNCTIONAL,
                severity=BugSeverity.MAJOR,
                description=(
                    f"死胡同页面: 检测到 {len(elements)} 个元素"
                    f"但无可交互元素 (button/icon/input)"
                ),
                tags=[BugTag.DEAD_END],
                evidence={
                    "total_elements": len(elements),
                    "element_types": list({e.get("type", "unknown") for e in elements}),
                },
            ))

        # ── 无返回路径检测 ──────────────────────────────────────
        # 如果已经离开了首页（step > 0），但页面上找不到“返回/关闭”类按钮
        # 仅当可点击元素 ≥5 时检测，避免对弹窗、引导页等正常页面过多误报
        _BACK_CHECK_MIN_CLICKABLE = 20  # Unity 游戏用场景导航，小于此数的页面不检查
        step = state.get("step", 0)
        if step > 0 and len(clickable) >= _BACK_CHECK_MIN_CLICKABLE:
            # 检查全部元素（含 text 类型），因为返回按钮可能被 OCR 识别为 text
            has_back = any(
                any(kw in (e.get("label", "") or "").lower() for kw in _BACK_KEYWORDS)
                for e in elements
            )
            if not has_back:
                # 仅当页面有按钮但没有返回类按钮时报告（纯展示页可能正常）
                labels = [e.get("label", "") for e in clickable[:10]]
                bugs.append(BugReport(
                    category=BugCategory.FUNCTIONAL,
                    severity=BugSeverity.MINOR,
                    description=(
                        f"疑似无返回路径: 当前页面有 {len(clickable)} 个可交互元素"
                        f"但未检测到返回/关闭按钮"
                    ),
                    tags=[BugTag.NO_BACK_BUTTON],
                    evidence={"clickable_labels": labels},
                ))

        # ── 单次 tap 无响应检测 ──────────────────────────────────
        # validate 节点已判断 page_changed，这里检查最近一次操作
        try:
            recent = worker.memory.working.recent(n=1)
            if recent:
                last = recent[-1]
                if last.action == ActionType.TAP and not last.success:
                    bugs.append(BugReport(
                        category=BugCategory.FUNCTIONAL,
                        severity=BugSeverity.MINOR,
                        description=(
                            f"按钮可能无响应: tap [{last.element_label}] 后页面未变化"
                        ),
                        tags=[BugTag.TAP_NO_RESPONSE],
                        evidence={
                            "action": last.action,
                            "element_label": last.element_label,
                            "page_hash": last.page_hash,
                        },
                    ))
        except Exception as exc:
            logger.debug("tap 无响应检测异常: %s", exc)

        return bugs
