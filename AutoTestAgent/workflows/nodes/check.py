from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.agent.worker import LangGraphWorker

logger = logging.getLogger(__name__)


def make_node(worker: "LangGraphWorker"):
    """崩溃检测 + Bug 快照保存 + 触发恢复子图。"""

    def node(state):
        if state.get("done"):
            return {}

        try:
            if worker._get_adb().check_crash():
                logger.error("[CRASH] 检测到游戏崩溃！")
                page_hash  = state.get("page_hash", "")
                shot_path  = ""
                screenshot = state.get("screenshot")
                if page_hash and screenshot:
                    shot_path = os.path.join(
                        state.get("memory_dir", worker.config.memory_dir),
                        "screenshots",
                        f"crash_{page_hash[:8]}_step{state.get('step', 0):03d}.png",
                    )
                    os.makedirs(os.path.dirname(shot_path), exist_ok=True)
                    screenshot.save(shot_path)
                if page_hash:
                    worker.memory.experience.save_bug(
                        page_hash=page_hash,
                        description=f"崩溃，任务: {state.get('task', '')[:60]}",
                        screenshot_path=shot_path,
                        tags=["crash"],
                    )
                recovered = worker.run_recovery("游戏崩溃")
                if not recovered:
                    return {"done": True, "result": "游戏崩溃"}
                logger.info("[CRASH] 恢复成功，继续测试")
        except Exception as exc:
            logger.warning("崩溃检测失败（忽略）: %s", exc)

        return {}

    return node
