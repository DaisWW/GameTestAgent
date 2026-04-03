from __future__ import annotations

"""SequentialDecider：不依赖 LLM 的顺序遍历决策器。

接口与 LLMAdapter 完全一致（都有 ask(ContextPacket) -> dict），
可通过 LLM_PROVIDER=sequential 或 --llm-provider sequential 直接替换。

决策逻辑：
    1. 从 ContextPacket.get_unvisited_ids() 取未访问元素
    2. 跳过 input 类型（文本输入框）和非可交互元素（纯文本标签）
    3. 全部访问完（或均不可点）→ done=True
"""

import logging
from typing import Any, Dict, TYPE_CHECKING

from core.llm.base import BrainProvider

if TYPE_CHECKING:
    from core.context.protocol import ContextPacket

logger = logging.getLogger(__name__)


_SKIP_TYPES = {"input"}  # 跳过这些类型：需要键盘输入，顺序模式无法处理


class SequentialDecider(BrainProvider):
    """顺序遍历决策器，无需任何 LLM 或 API Key。"""

    def _is_tappable(self, elem: Dict[str, Any]) -> bool:
        """判断元素是否可以被顺序点击。"""
        if elem.get("type") in _SKIP_TYPES:
            return False
        if not elem.get("interactable", True):
            return False
        return True

    def ask(self, packet: "ContextPacket") -> Dict[str, Any]:
        unvisited  = packet.get_unvisited_ids()
        elem_map   = {e["id"]: e for e in packet.current_observation.get("omni_boxes", [])}

        # 从未访问列表中过滤出可点击的元素
        tappable = [eid for eid in unvisited if self._is_tappable(elem_map.get(eid, {}))]
        skipped  = [eid for eid in unvisited if not self._is_tappable(elem_map.get(eid, {}))]

        if skipped:
            skip_labels = [elem_map.get(s, {}).get("label", str(s)) for s in skipped]
            logger.info("  [Sequential] 跳过 %d 个元素（input/非交互）: %s",
                        len(skipped), skip_labels)

        if not tappable:
            logger.info("  [Sequential] 所有可点击元素已遍历，结束任务")
            return {
                "action":    "done",
                "params":    {},
                "reasoning": "所有可交互元素已按顺序遍历完毕",
                "done":      True,
                "result":    "pass",
            }

        elem_id = tappable[0]
        label   = elem_map.get(elem_id, {}).get("label", "")

        logger.info("  [Sequential] tap [%d] %s（剩余 %d 个可点击）",
                    elem_id, label, len(tappable) - 1)

        return {
            "action":    "tap",
            "params":    {"id": elem_id},
            "reasoning": f"顺序遍历: [{elem_id}] {label}",
            "done":      False,
            "result":    "",
        }

    def __repr__(self) -> str:
        return "SequentialDecider()"
