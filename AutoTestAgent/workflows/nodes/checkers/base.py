from __future__ import annotations

"""Bug 检测基础模块 —— BugReport 数据模型 + BugChecker 抽象基类。

所有具体 Checker 继承 BugChecker 并实现 check() 方法，
返回 0~N 个 BugReport 实例。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, TYPE_CHECKING

from core.types import BugCategory, BugSeverity, BugTag

if TYPE_CHECKING:
    from core.agent.worker import LangGraphWorker


@dataclass
class BugReport:
    """一条 Bug 检测结果。"""

    category: BugCategory
    """检测维度，见 BugCategory 枚举"""

    severity: BugSeverity
    """严重程度，见 BugSeverity 枚举"""

    description: str
    """人类可读的 Bug 描述"""

    tags: List[BugTag] = field(default_factory=list)
    """细分标签列表，见 BugTag 枚举，存入 experience_pool.tags"""

    evidence: Dict[str, Any] = field(default_factory=dict)
    """附加证据（bbox 坐标、像素值、IoU 等），方便后续回溯与可视化"""


class BugChecker(ABC):
    """Bug 检测器抽象基类。

    每个维度的检测器继承此类，放在 checkers/ 目录下的独立文件中。
    """

    name: str = "base"
    """检测器名称，子类必须覆盖"""

    @abstractmethod
    def check(self, state: dict, worker: "LangGraphWorker") -> List[BugReport]:
        """执行检测，返回发现的 Bug 列表。无 Bug 时返回空列表。

        Args:
            state:  当前 LangGraph AgentState 字典，包含 screenshot / ui_elements /
                    page_hash / context_packet / current_action 等。
            worker: LangGraphWorker 实例，可访问 ADB、memory 等。

        Returns:
            检测到的 BugReport 列表。
        """

    def is_applicable(self, state: dict) -> bool:
        """可选：判断当前 state 是否满足本检测器的前置条件。

        默认返回 True。子类可覆盖，例如 LayoutChecker 需要 ui_elements 非空。
        """
        return True
