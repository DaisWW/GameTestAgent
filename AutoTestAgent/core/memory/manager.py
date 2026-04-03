from __future__ import annotations

"""MemoryManager：三层记忆系统统一管理器。

将 WorkingMemory / NavigationGraph / ExperiencePool 的初始化、
持久化和高层操作封装为统一接口，令 LangGraphWorker 只持有一个
memory 引用，而 workflow 节点通过语义化方法操作记忆，不直接接触
底层存储细节。
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .working_memory import WorkingMemory, MemoryStep
from .nav_graph import NavigationGraph
from .experience_pool import ExperiencePool

if TYPE_CHECKING:
    from config.settings import AgentConfig

logger = logging.getLogger(__name__)


class MemoryManager:
    """三层记忆系统管理器。

    Args:
        config: AgentConfig 实例，读取 config.memory_dir 确定持久化路径。
        working_capacity: WorkingMemory 滑窗容量，默认 10。
    """

    def __init__(
        self,
        config: "AgentConfig",
        working_capacity: int = 10,
    ) -> None:
        memory_dir = Path(config.memory_dir)
        nav_path   = str(memory_dir / "nav_graph.json")
        db_path    = str(memory_dir / "logs" / "experience.db")
        memory_dir.mkdir(parents=True, exist_ok=True)
        (memory_dir / "logs").mkdir(parents=True, exist_ok=True)

        self.working   = WorkingMemory(capacity=working_capacity)
        self.nav_graph = NavigationGraph(graph_path=nav_path)
        self.experience = ExperiencePool(db_path=db_path)
        self.memory_dir = memory_dir

        logger.info(
            "MemoryManager 已初始化: nav=%s db=%s",
            nav_path, db_path,
        )

    # ── 高层语义操作 ─────────────────────────────────────────────

    def record_step(
        self,
        step: int,
        action: str,
        params: Dict[str, Any],
        reasoning: str,
        page_hash: str,
        result_hash: str,
        page_changed: bool,
        element_label: str = "",
    ) -> None:
        """记录一步操作到 WorkingMemory。"""
        self.working.push(MemoryStep(
            step=step,
            action=action,
            params=params,
            reasoning=reasoning,
            page_hash=page_hash,
            result_hash=result_hash,
            success=page_changed,
            element_label=element_label,
        ))

    def record_transition(
        self,
        prev_hash: str,
        result_hash: str,
        action_type: str,
        elem_id: Optional[int],
        element_label: str = "",
    ) -> None:
        """添加页面转移边到 NavigationGraph。元素访问标记由 validate 节点单独调用。"""
        if result_hash and prev_hash and result_hash != prev_hash:
            self.nav_graph.add_transition(
                from_hash=prev_hash,
                to_hash=result_hash,
                action=action_type,
                element_id=elem_id if elem_id is not None else -1,
                element_label=element_label,
            )
            self.nav_graph.register_page(result_hash)

    def get_anomaly_flag(self) -> str:
        """从 WorkingMemory 获取异常标记（ABA 循环 / 卡死点击）。"""
        return self.working.get_anomaly_flag()

    def persist(self) -> None:
        """持久化 NavigationGraph 到 JSON 文件。"""
        self.nav_graph.save_json()

    def close(self) -> None:
        """关闭 SQLite 连接。"""
        self.experience.close()

    def save_successful_path(self, task: str, steps: int) -> None:
        """将成功路径保存到 ExperiencePool。"""
        self.experience.save_successful_path(
            task=task,
            steps=self.working.recent(n=steps),
            page_hashes=self.working.page_hash_sequence(),
        )

    def __repr__(self) -> str:
        return (
            f"MemoryManager(working={self.working!r}, "
            f"nav={self.nav_graph!r}, "
            f"exp={self.experience!r})"
        )
