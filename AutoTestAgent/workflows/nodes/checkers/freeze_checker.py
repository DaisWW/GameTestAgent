from __future__ import annotations

"""维度 4：卡死 / 循环检测 (Freeze Checker)

利用 ContextBuilder 已计算好的 anomaly_flag（UI 卡死、ABA 循环），
以及 working_memory 中的操作历史来检测：
- UI 冻结（连续多帧 phash 几乎不变）
- ABA 循环（页面被反复切换）
- 连续操作无效（多次 tap 后页面不变）
"""

import logging
from typing import List, TYPE_CHECKING

from .base import BugChecker, BugReport

if TYPE_CHECKING:
    from core.agent.worker import LangGraphWorker

logger = logging.getLogger(__name__)

_NO_CHANGE_STREAK_THRESHOLD = 3  # 连续 N 次操作页面不变则报告


class FreezeChecker(BugChecker):
    name = "freeze"

    def check(self, state: dict, worker: "LangGraphWorker") -> List[BugReport]:
        bugs: List[BugReport] = []

        # ── 读取 anomaly_flag（由 ContextBuilder 在 perception 中已计算）──
        packet = state.get("context_packet")
        anomaly_flag = ""
        if packet is not None:
            anomaly_flag = getattr(packet, "anomaly_flag", "") or ""

        if anomaly_flag:
            # 区分卡死和循环
            if "卡死" in anomaly_flag or "冻结" in anomaly_flag:
                bugs.append(BugReport(
                    category="freeze",
                    severity="critical",
                    description=anomaly_flag,
                    tags=["ui_freeze"],
                    evidence={"anomaly_flag": anomaly_flag},
                ))
            elif "ABA" in anomaly_flag or "循环" in anomaly_flag:
                bugs.append(BugReport(
                    category="freeze",
                    severity="major",
                    description=anomaly_flag,
                    tags=["aba_loop"],
                    evidence={"anomaly_flag": anomaly_flag},
                ))
            else:
                bugs.append(BugReport(
                    category="freeze",
                    severity="major",
                    description=f"异常标记: {anomaly_flag}",
                    tags=["anomaly"],
                    evidence={"anomaly_flag": anomaly_flag},
                ))

        # ── 连续操作无效检测 ──────────────────────────────────────
        # 从 working_memory 取最近的操作记录，看是否连续 page_changed=False
        try:
            recent = worker.memory.working.recent(n=_NO_CHANGE_STREAK_THRESHOLD)
            if len(recent) >= _NO_CHANGE_STREAK_THRESHOLD:
                # MemoryStep 中 success 字段对应 page_changed
                no_change_streak = all(
                    not step_rec.get("success", True)
                    for step_rec in recent[-_NO_CHANGE_STREAK_THRESHOLD:]
                )
                if no_change_streak:
                    actions = [s.get("action", "?") for s in recent[-_NO_CHANGE_STREAK_THRESHOLD:]]
                    bugs.append(BugReport(
                        category="freeze",
                        severity="major",
                        description=(
                            f"连续 {_NO_CHANGE_STREAK_THRESHOLD} 次操作页面未变化: "
                            f"{actions}"
                        ),
                        tags=["no_response_streak"],
                        evidence={"actions": actions, "streak": _NO_CHANGE_STREAK_THRESHOLD},
                    ))
        except Exception as exc:
            logger.debug("连续无效检测异常: %s", exc)

        return bugs
