from __future__ import annotations

"""维度 1：崩溃检测 (Crash Checker)

通过 ADB 检测游戏进程是否崩溃 / ANR。
严重程度：critical —— 触发恢复流程。
"""

import logging
from typing import List, TYPE_CHECKING

from .base import BugChecker, BugReport

if TYPE_CHECKING:
    from core.agent.worker import LangGraphWorker

logger = logging.getLogger(__name__)


class CrashChecker(BugChecker):
    name = "crash"

    def check(self, state: dict, worker: "LangGraphWorker") -> List[BugReport]:
        try:
            if worker._get_adb().check_crash():
                task = state.get("task", "")[:60]
                return [BugReport(
                    category="crash",
                    severity="critical",
                    description=f"游戏崩溃，当前任务: {task}",
                    tags=["crash"],
                    evidence={"page_hash": state.get("page_hash", ""),
                              "step": state.get("step", 0)},
                )]
        except Exception as exc:
            logger.warning("崩溃检测失败（忽略）: %s", exc)
        return []
