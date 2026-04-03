from __future__ import annotations

"""LangGraphWorker：测试流程统一协调器。

职责（仅此一项）：
- 持有 VisionProvider、BrainProvider、MemoryManager、ADBController 引用
- 构建并运行 LangGraph 导向图（run / run_recovery）
- 展出节点所需的简单操作接口（capture / detect / execute）

截图存储 → MediaStore    报告生成 → core.reporting.writer
"""

import logging
from typing import Any, Dict, List, Optional

from PIL import Image

from config.settings import AgentConfig
from core.context.protocol import ContextBuilder
from core.llm.base import BrainProvider
from core.vision.base import VisionProvider
from core.memory.manager import MemoryManager
from core.agent.media_store import MediaStore
from core.reporting.writer import write_report
from tools import ADBController, ADBError, RunnerSettings
from core.agent.executor import ActionExecutor

logger = logging.getLogger(__name__)


class LangGraphWorker:
    """封装完整 Agent 执行流的工作类。

    Args:
        vision:  VisionProvider 实例。
        llm:     BrainProvider 实例（LLMAdapter 或 SequentialDecider）。
        config:  AgentConfig 配置实例。
    """

    def __init__(
        self,
        vision: VisionProvider,
        llm: BrainProvider,
        config: AgentConfig,
    ) -> None:
        self.vision = vision
        self.llm    = llm
        self.config = config

        self.memory = MemoryManager(config)
        self.media  = MediaStore(config)

        self.context_builder = ContextBuilder(
            working_memory=self.memory.working,
            nav_graph=self.memory.nav_graph,
            experience_pool=self.memory.experience,
            max_steps=config.max_steps,
            history_window=5,
        )

        self._adb:      Optional[ADBController] = None
        self._executor: Optional[ActionExecutor] = None
        self._graph = None

    # ── ADB 初始化 ──────────────────────────────────────────────────

    def _get_adb(self) -> ADBController:
        if self._adb is None:
            settings = RunnerSettings(
                game_package=self.config.adb.game_package,
                game_activity=self.config.adb.game_activity,
                game_launch_wait=self.config.adb.game_launch_wait,
            )
            self._adb = ADBController(settings=settings, serial=self.config.adb.serial)
            try:
                self._adb.connect()
            except ADBError as exc:
                raise RuntimeError(f"ADB 设备连接失败: {exc}") from exc
            self._executor = ActionExecutor(self._adb)
        return self._adb

    # ── LangGraph 图（延迟初始化）──────────────────────────────────

    def _get_graph(self):
        if self._graph is None:
            from workflows.waterfall_flow import build_graph
            self._graph = build_graph(self)
        return self._graph

    # ── 节点动作（供 waterfall_flow 节点调用）──────────────────────

    def capture(self) -> Image.Image:
        """截取当前屏幕。"""
        return self._get_adb().screenshot()

    def detect(self, image: Image.Image) -> List[Dict[str, Any]]:
        """调用视觉 Provider 检测 UI 元素。"""
        return self.vision.detect(image)

    def execute(self, action: Dict[str, Any]) -> None:
        """根据大脑决策执行 ADB 动作（委托给 ActionExecutor）。"""
        self._get_adb()  # 确保 _executor 已初始化
        self._executor.execute(action)

    def save_screenshot(self, image: Image.Image, step: int, run_dir: str = "") -> None:
        self.media.save_step(image, step)

    def save_page_screenshot(self, image: Image.Image, page_hash: str, memory_dir: str = "") -> str:
        return self.media.save_page(image, page_hash)

    def save_annotated_screenshot(
        self,
        image: Image.Image,
        ui_elements: List[Dict[str, Any]],
        step: int,
        run_dir: str = "",
    ) -> None:
        self.media.save_annotated(image, ui_elements, step)

    # ── 主入口 ──────────────────────────────────────────────────────

    def run(self, task: str) -> Dict[str, Any]:
        """执行完整的测试任务。

        Args:
            task: 自然语言测试任务描述，例如 "进入设置页面并开启夜间模式"。

        Returns:
            测试结果字典::

                {
                    "status":  "pass" | "fail" | "error",
                    "steps":   12,
                    "reason":  "...",
                    "history": [...],
                    "nav_stats": {...},
                }
        """
        logger.info("=" * 60)
        logger.info("开始测试任务: %s", task)
        logger.info("Vision: %r  LLM: %r", self.vision, self.llm)
        logger.info("Run ID  : %s", self.config.run_id)
        logger.info("Run Dir : %s", self.config.run_dir)
        logger.info("=" * 60)

        # 记录本轮开始时的 Bug 水位，用于报告中只展示新增 Bug
        pre_run_bug_id = self.memory.experience.get_last_bug_id()

        # ── 连接设备 + 启动游戏 ─────────────────────────────────────
        adb = self._get_adb()
        if self.config.adb.game_package:
            logger.info("启动游戏: %s", self.config.adb.game_package)
            adb.launch_game()

        graph = self._get_graph()

        initial_state = {
            "task":           task,
            "screenshot":     None,
            "ui_elements":    [],
            "page_hash":      "",
            "context_packet": None,
            "current_action": None,
            "step":           0,
            "done":           False,
            "result":         "",
            "run_dir":        self.config.run_dir,
            "memory_dir":     self.config.memory_dir,
        }

        try:
            final_state = dict(initial_state)
            for event in graph.stream(initial_state, stream_mode="updates"):
                node_name, node_state = next(iter(event.items()))
                if node_state:
                    final_state.update(node_state)
                logger.info(
                    ">>> [NODE] %-12s | step=%-3s done=%s result=%r",
                    node_name,
                    final_state.get("step", "?"),
                    final_state.get("done", False),
                    (final_state.get("result") or "")[:60],
                )
        except Exception as exc:
            logger.error("Agent 执行异常: %s", exc, exc_info=True)
            self.memory.persist()
            return {"status": "error", "steps": 0, "reason": str(exc),
                    "history": [], "nav_stats": {}, "bug_summary": {}, "bugs": []}

        # ── 测试结束后持久化记忆 ──────────────────────────────────
        result_str = final_state.get("result", "")
        is_pass    = result_str == "pass"
        steps      = final_state.get("step", 0)

        if is_pass:
            self.memory.save_successful_path(task=task, steps=steps)

        self.memory.persist()

        nav_stats = self.memory.nav_graph.stats()
        logger.info(
            "导航图统计: pages=%d transitions=%d visited_elements=%d",
            nav_stats["pages"], nav_stats["transitions"], nav_stats["visited_elements"],
        )

        # ── Bug 汇总 ────────────────────────────────────────────
        all_bugs = self.memory.experience.get_all_bugs()
        bug_summary = {"total": len(all_bugs)}
        for b in all_bugs:
            cat = b.get("category", "unknown") or "unknown"
            bug_summary[cat] = bug_summary.get(cat, 0) + 1
        if all_bugs:
            logger.info("Bug 汇总: %s", bug_summary)

        status   = "pass" if is_pass else "fail"
        history  = self.memory.working.recent(n=steps)
        new_bugs = self.memory.experience.get_bugs_since(pre_run_bug_id)
        self._save_report(
            run_dir=self.config.run_dir,
            run_id=self.config.run_id,
            game_package=self.config.adb.game_package,
            task=task, status=status, steps=steps, history=history,
            nav_stats=nav_stats, new_bugs=new_bugs, reason=result_str,
        )

        return {
            "status":      status,
            "steps":       steps,
            "reason":      result_str,
            "history":     history,
            "nav_stats":   nav_stats,
            "bug_summary": bug_summary,
            "bugs":        all_bugs,
        }

    def _save_report(self, **kwargs) -> None:
        write_report(**kwargs)

    def run_recovery(self, reason: str) -> bool:
        """触发异常恢复子图。

        Args:
            reason: 触发恢复的原因描述，传入 recovery_flow。

        Returns:
            True = 恢复成功，可继续主流程；False = 放弃，需终止测试。
        """
        from workflows.recovery_flow import build_recovery_graph
        logger.warning("[Recovery] 触发恢复流程，原因: %s", reason[:80])
        result = build_recovery_graph(self).invoke({"reason": reason})
        recovered = bool(result.get("recovered", False))
        if recovered:
            logger.info("[Recovery] 恢复成功，继续主流程")
        else:
            logger.error("[Recovery] 无法恢复，终止测试")
        return recovered

    def teardown(self) -> None:
        """释放资源（显存、数据库连接等）。"""
        self.vision.teardown()
        self.memory.close()
        logger.info("Worker 资源已释放")
