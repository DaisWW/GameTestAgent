from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from core.vision.perception import compute_phash, is_page_changed

if TYPE_CHECKING:
    from core.agent.worker import LangGraphWorker

logger = logging.getLogger(__name__)


def make_node(worker: "LangGraphWorker"):
    """截图对比 phash，更新三层记忆（WorkingMemory + NavigationGraph）。"""

    def node(state):
        if state.get("done"):
            return {}

        step   = state.get("step", 0)
        action = state.get("current_action", {})
        logger.info("[Step %02d] ── VALIDATE ─────────────────────────", step)

        if worker.config.step_delay > 0:
            time.sleep(worker.config.step_delay)

        after_shot   = worker.capture()
        result_hash  = compute_phash(after_shot)
        prev_hash    = state.get("page_hash", "")
        page_changed = is_page_changed(prev_hash, result_hash)

        logger.info(
            "  Hash: %s → %s  [%s]",
            prev_hash[:8], result_hash[:8],
            "CHANGED" if page_changed else "NO_CHANGE",
        )

        act_type = action.get("action", "")
        elem_id  = action.get("params", {}).get("id")

        worker.memory.record_step(
            step=step,
            action=act_type,
            params=action.get("params", {}),
            reasoning=action.get("reasoning", ""),
            page_hash=prev_hash,
            result_hash=result_hash,
            page_changed=page_changed,
            element_label=action.get("_element_label", ""),
        )

        # 无论页面是否变化，tap 过的元素都应标为已访问，避免 SequentialDecider 无限重试
        if act_type == "tap" and elem_id is not None:
            worker.memory.nav_graph.mark_visited(prev_hash, elem_id)

        if page_changed:
            worker.memory.record_transition(
                prev_hash=prev_hash,
                result_hash=result_hash,
                action_type=act_type,
                elem_id=elem_id,
                element_label=action.get("_element_label", ""),
            )

        return {"prev_hash": prev_hash, "page_hash": result_hash, "step": step + 1}

    return node
