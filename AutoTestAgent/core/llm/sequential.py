from __future__ import annotations

"""SequentialDecider：不依赖 LLM 的顺序遍历决策器。

接口与 LLMAdapter 完全一致（都有 ask(ContextPacket) -> dict），
可通过 LLM_PROVIDER=sequential 或 --llm-provider sequential 直接替换。

决策逻辑：
    1. 从 ContextPacket.get_unvisited_ids() 取第一个未访问元素 → tap
    2. 全部访问完 → done=True
"""

import logging
from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from core.context.protocol import ContextPacket

logger = logging.getLogger(__name__)


class SequentialDecider:
    """顺序遍历决策器，无需任何 LLM 或 API Key。"""

    def ask(self, packet: "ContextPacket") -> Dict[str, Any]:
        unvisited = packet.get_unvisited_ids()

        if not unvisited:
            logger.info("  [Sequential] 所有元素已遍历，结束任务")
            return {
                "action":    "done",
                "params":    {},
                "reasoning": "所有可交互元素已按顺序遍历完毕",
                "done":      True,
                "result":    "pass",
            }

        elem_id   = unvisited[0]
        elem_map  = {e["id"]: e for e in packet.current_observation.get("omni_boxes", [])}
        label     = elem_map.get(elem_id, {}).get("label", "")

        logger.info("  [Sequential] tap [%d] %s（剩余 %d 个未访问）",
                    elem_id, label, len(unvisited) - 1)

        return {
            "action":    "tap",
            "params":    {"id": elem_id},
            "reasoning": f"顺序遍历: [{elem_id}] {label}",
            "done":      False,
            "result":    "",
        }

    def __repr__(self) -> str:
        return "SequentialDecider()"
