from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from core.vision.perception import compute_phash

if TYPE_CHECKING:
    from core.agent.worker import LangGraphWorker

logger = logging.getLogger(__name__)


def make_node(worker: "LangGraphWorker"):
    """截图 → OmniParser 检测 → phash → 注册导航图 → 构建 ContextPacket。"""

    def node(state):
        step = state.get("step", 0)
        logger.info("[Step %02d] ── PERCEPTION ──────────────────────", step)

        screenshot  = worker.capture()
        worker.save_screenshot(screenshot, step, state.get("run_dir", worker.config.run_dir))

        ui_elements = worker.detect(screenshot)
        logger.info("  OmniParser: %d 个元素", len(ui_elements))

        page_hash = compute_phash(screenshot)
        logger.info("  PageHash: %s", page_hash[:8])

        worker.save_page_screenshot(screenshot, page_hash, state.get("memory_dir", worker.config.memory_dir))
        worker.memory.nav_graph.register_page(page_hash, elements=ui_elements)

        packet = worker.context_builder.build(
            screenshot=screenshot,
            ui_elements=ui_elements,
            page_hash=page_hash,
            task=state["task"],
            step=step,
        )

        if packet.anomaly_flag:
            logger.warning("  [ANOMALY] %s", packet.anomaly_flag)

        return {
            "screenshot":     screenshot,
            "ui_elements":    ui_elements,
            "page_hash":      page_hash,
            "context_packet": packet,
        }

    return node
