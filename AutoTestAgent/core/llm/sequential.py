from __future__ import annotations

"""SequentialDecider：不依赖 LLM 的顺序遍历决策器。

接口与 LLMAdapter 完全一致（都有 ask(ContextPacket) -> dict），
可通过 LLM_PROVIDER=sequential 或 --llm-provider sequential 直接替换。

决策逻辑：
    1. 从 ContextPacket.get_unvisited_ids() 取未访问元素
    2. 跳过 input 类型（文本输入框）和非可交互元素（纯文本标签）
    3. 当前页访问完 → 优先 DFS 深入子页；无子页则 PRESS_BACK 回溯
    4. 连续回退超过 _MAX_BACK_STREAK 次仍无新内容 → done=True
"""

import logging
from typing import Any, Dict, TYPE_CHECKING

from core.llm.base import BrainProvider
from core.types import ActionType, ElementType, TestStatus

if TYPE_CHECKING:
    from core.context.protocol import ContextPacket

logger = logging.getLogger(__name__)


_SKIP_TYPES = {ElementType.INPUT}  # 跳过这些类型：需要键盘输入，顺序模式无法处理


_MAX_BACK_STREAK = 15  # 连续回退超过此次数仍无新内容，认为探索完毕


class SequentialDecider(BrainProvider):
    """顺序遍历决策器，无需任何 LLM 或 API Key。"""

    def __init__(self) -> None:
        self._back_streak: int = 0  # 连续 PRESS_BACK 次数（找到新内容时重置）

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

        # 从未访问列表中一次遍历分出可点击 / 需跳过的元素
        tappable, skipped = [], []
        for eid in unvisited:
            (tappable if self._is_tappable(elem_map.get(eid, {})) else skipped).append(eid)

        if skipped:
            skip_labels = [elem_map.get(s, {}).get("label", str(s)) for s in skipped]
            logger.info("  [Sequential] 跳过 %d 个元素（input/非交互）: %s",
                        len(skipped), skip_labels)

        if not tappable:
            nav_hints = packet.get_nav_hints()
            nav_tappable = [h for h in nav_hints if h in elem_map]
            if nav_tappable:
                self._back_streak = 0
                nav_id    = nav_tappable[0]
                nav_label = elem_map.get(nav_id, {}).get("label", "")
                logger.info("  [Sequential] 当前页已探索完，点击 [%d] %s 深入子页", nav_id, nav_label)
                return {
                    "action":    ActionType.TAP,
                    "params":    {"id": nav_id},
                    "reasoning": f"DFS 导航：[{nav_id}] {nav_label} → 进入有未探索内容的子页",
                    "done":      False,
                    "result":    "",
                }

            # 当前页及已知子页均已探索完，按返回键向上回溯
            # 连续回退超过 _MAX_BACK_STREAK 次仍无新内容 → 真正结束
            self._back_streak += 1
            if self._back_streak > _MAX_BACK_STREAK:
                logger.info("  [Sequential] 连续回退 %d 次仍无新内容，探索完毕", self._back_streak)
                return {
                    "action":    ActionType.DONE,
                    "params":    {},
                    "reasoning": f"连续回退 {self._back_streak} 次仍无未探索内容，结束",
                    "done":      True,
                    "result":    TestStatus.PASS,
                }

            logger.info("  [Sequential] 当前页已探索完，按返回键回退 (streak=%d/%d)",
                        self._back_streak, _MAX_BACK_STREAK)
            return {
                "action":    ActionType.PRESS_BACK,
                "params":    {},
                "reasoning": f"当前页及已知子页探索完毕，回退寻找未访问分支 (streak={self._back_streak})",
                "done":      False,
                "result":    "",
            }

        self._back_streak = 0  # 找到可点击内容，重置回退计数
        elem_id = tappable[0]
        label   = elem_map.get(elem_id, {}).get("label", "")

        logger.info("  [Sequential] tap [%d] %s（剩余 %d 个可点击）",
                    elem_id, label, len(tappable) - 1)

        return {
            "action":    ActionType.TAP,
            "params":    {"id": elem_id},
            "reasoning": f"顺序遍历: [{elem_id}] {label}",
            "done":      False,
            "result":    "",
        }

    def __repr__(self) -> str:
        return "SequentialDecider()"
