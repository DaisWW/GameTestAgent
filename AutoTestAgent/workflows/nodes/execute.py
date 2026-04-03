from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.agent.worker import LangGraphWorker

logger = logging.getLogger(__name__)


def make_node(worker: "LangGraphWorker"):
    """通过 ADB 执行大脑决策（tap / swipe / input_text 等）。"""

    def node(state):
        action = state.get("current_action")
        if not action or state.get("done"):
            return {}

        step = state.get("step", 0)
        logger.info("[Step %02d] ── EXECUTE: %s %s", step, action.get("action"), action.get("params"))

        try:
            worker.execute(action)
        except Exception as exc:
            logger.error("  执行失败: %s", exc)
            return {"done": True, "result": f"执行失败: {exc}"}

        return {}

    return node
