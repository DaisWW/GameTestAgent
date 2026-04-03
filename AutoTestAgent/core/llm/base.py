from __future__ import annotations

"""BrainProvider：Brain 层统一抽象接口。

所有决策组件（LLMAdapter、SequentialDecider 等）均须实现此接口，
使 Worker / Workflow 与具体实现完全解耦，便于测试注入和扩展。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from core.context.protocol import ContextPacket


class BrainProvider(ABC):
    """Brain 层抽象基类。

    实现者须实现 ask(packet) 方法，返回符合 ActionDecision 格式的字典::

        {
            "action":    "tap" | "swipe" | "input_text" | "press_home" | "wait" | "done",
            "params":    {...},
            "reasoning": "...",
            "done":      false,
            "result":    "" | "pass",
        }
    """

    @abstractmethod
    def ask(self, packet: "ContextPacket") -> Dict[str, Any]:
        """根据当前上下文包返回 ActionDecision 字典。"""
