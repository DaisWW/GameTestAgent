from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.agent.worker import LangGraphWorker

logger = logging.getLogger(__name__)


def make_node(worker: "LangGraphWorker"):
    """调用 LLM/Sequential 大脑，输出 ActionDecision。"""

    def node(state):
        step = state.get("step", 0)
        logger.info("[Step %02d] ── COGNITION ───────────────────────", step)

        if step >= worker.config.max_steps:
            logger.warning("  已达最大步数 %d", worker.config.max_steps)
            return {"done": True, "result": f"超出最大步数 {worker.config.max_steps}"}

        packet   = state["context_packet"]
        decision = worker.llm.ask(packet)

        elem_id = decision.get("params", {}).get("id")
        if elem_id is not None:
            try:
                elem_id = int(elem_id)
                decision["params"]["id"] = elem_id
            except (TypeError, ValueError):
                logger.warning("  elem_id 无法转为 int: %r，跳过 bbox 注入", elem_id)
                elem_id = None

        if elem_id is not None:
            elem_map = {e["id"]: e for e in state.get("ui_elements", [])}
            elem = elem_map.get(elem_id)
            if elem:
                decision["_bbox"]          = elem["bbox"]
                decision["_element_label"] = elem.get("label", "")
            else:
                logger.warning("  Brain 引用了不存在的元素 id=%s", elem_id)

        logger.info("  决策: action=%-12s params=%s", decision.get("action"), decision.get("params"))
        logger.info("  推理: %s", decision.get("reasoning", "")[:80])

        raw_done = decision.get("done", False)
        is_done  = raw_done is True or str(raw_done).lower() == "true"
        return {
            "current_action": decision,
            "done":           is_done,
            "result":         decision.get("result", ""),
        }

    return node
