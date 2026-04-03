from __future__ import annotations

"""维度 4：卡死 / 循环检测 (Freeze Checker)

完全依赖 ContextBuilder 预先写入 ContextPacket.anomaly_flag 的字符串来判断，
不再直接访问 working_memory：
- UI 冻结（detect_ui_freeze：连续帧 phash 几乎不变）
- ABA 循环（is_in_aba_loop：页面被反复交替访问）
- 连续操作无效（detect_stale_click：多次操作后页面不变）
"""

import logging
from typing import List, TYPE_CHECKING

from .base import BugChecker, BugReport
from core.types import BugCategory, BugSeverity, BugTag

if TYPE_CHECKING:
    from core.agent.worker import LangGraphWorker

logger = logging.getLogger(__name__)


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
            # 区分卡死 / ABA 循环 / 连续无响应（detect_stale_click 的输出含此关键词）
            if "卡死" in anomaly_flag or "冻结" in anomaly_flag:
                bugs.append(BugReport(
                    category=BugCategory.FREEZE,
                    severity=BugSeverity.CRITICAL,
                    description=anomaly_flag,
                    tags=[BugTag.UI_FREEZE],
                    evidence={"anomaly_flag": anomaly_flag},
                ))
            elif "ABA" in anomaly_flag or "循环" in anomaly_flag:
                bugs.append(BugReport(
                    category=BugCategory.FREEZE,
                    severity=BugSeverity.MAJOR,
                    description=anomaly_flag,
                    tags=[BugTag.ABA_LOOP],
                    evidence={"anomaly_flag": anomaly_flag},
                ))
            elif "未引起页面变化" in anomaly_flag:
                # detect_stale_click() 触发：使用具体标签而非通用 ANOMALY，避免与下方直接检测产生双重报告
                bugs.append(BugReport(
                    category=BugCategory.FREEZE,
                    severity=BugSeverity.MAJOR,
                    description=anomaly_flag,
                    tags=[BugTag.NO_RESPONSE_STREAK],
                    evidence={"anomaly_flag": anomaly_flag},
                ))
            else:
                bugs.append(BugReport(
                    category=BugCategory.FREEZE,
                    severity=BugSeverity.MAJOR,
                    description=f"异常标记: {anomaly_flag}",
                    tags=[BugTag.ANOMALY],
                    evidence={"anomaly_flag": anomaly_flag},
                ))

        return bugs
