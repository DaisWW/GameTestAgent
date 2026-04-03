from __future__ import annotations

"""上下文协议 (Context Protocol)

ContextPacket — Vision 层与 Brain 层之间的统一数据契约，
包含三层记忆系统整合后的完整上下文，供云端 LLM 决策使用。
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from core.vision.perception import detect_ui_freeze

from PIL import Image

if TYPE_CHECKING:
    from core.memory.working_memory import WorkingMemory
    from core.memory.nav_graph import NavigationGraph
    from core.memory.experience_pool import ExperiencePool

logger = logging.getLogger(__name__)


@dataclass
class ContextPacket:
    """传递给 LLMAdapter.ask() 的完整上下文包。"""
    screenshot:           Image.Image
    task_goal:            str
    current_path:         str
    action_history:       List[Dict[str, Any]]
    current_observation:  Dict[str, Any]
    anomaly_flag:         str
    relevant_experience:  List[Dict[str, Any]]
    step:                 int
    max_steps:            int

    def to_json_context(self) -> str:
        """序列化为 LLM 可读的 JSON（不含图片二进制）。"""
        payload = {
            "task_goal":            self.task_goal,
            "current_path":         self.current_path,
            "step":                 f"{self.step} / {self.max_steps}",
            "action_history":       self.action_history,
            "current_observation":  self.current_observation,
            "anomaly_flag":         self.anomaly_flag or "无",
            "relevant_experience":  self.relevant_experience,
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def has_anomaly(self) -> bool:
        return bool(self.anomaly_flag)

    def get_unvisited_ids(self) -> List[int]:
        return self.current_observation.get("unvisited_ids", [])

    def get_page_hash(self) -> str:
        return self.current_observation.get("page_hash", "")

    def get_nav_hints(self) -> List[int]:
        """当前页已探索完毕时，指向有未探索子页的元素 ID 列表（DFS 导航用）。"""
        return self.current_observation.get("nav_hints", [])


class ContextBuilder:
    """整合三层记忆，构建每步调用 LLM 所需的 ContextPacket。

    Args:
        working_memory:  WorkingMemory 实例（第一层）
        nav_graph:       NavigationGraph 实例（第二层）
        experience_pool: ExperiencePool 实例（第三层）
        max_steps:       最大步数上限
        history_window:  传入 LLM 的历史步数，默认 5
    """

    def __init__(
        self,
        working_memory: "WorkingMemory",
        nav_graph: "NavigationGraph",
        experience_pool: "ExperiencePool",
        max_steps: int = 30,
        history_window: int = 5,
    ) -> None:
        self._wm = working_memory
        self._ng = nav_graph
        self._ep = experience_pool
        self._max_steps = max_steps
        self._history_window = history_window

    def build(
        self,
        screenshot: Image.Image,
        ui_elements: List[Dict[str, Any]],
        page_hash: str,
        task: str,
        step: int,
    ) -> ContextPacket:
        all_ids        = [e["id"] for e in ui_elements]
        element_labels = [e.get("label", "") for e in ui_elements if e.get("label")]

        unvisited_ids = self._ng.get_unvisited_ids(page_hash, all_ids)
        visited_ids   = self._ng.get_visited_ids(page_hash)
        visit_count   = self._ng.get_page_visit_count(page_hash)

        hash_seq = self._wm.page_hash_sequence()[-8:]
        if page_hash not in hash_seq:
            hash_seq.append(page_hash)
        current_path = self._ng.get_breadcrumb(hash_seq)

        anomaly_flag = self._wm.get_anomaly_flag()
        if not anomaly_flag:
            anomaly_flag = detect_ui_freeze(hash_seq) or ""
        if not anomaly_flag and self._ng.is_in_aba_loop(hash_seq):
            anomaly_flag = f"ABA 循环：页面 {page_hash[:8]} 被反复访问（共 {visit_count} 次）"

        relevant_experience = self._ep.get_relevant_experience(
            task=task, page_hash=page_hash, element_labels=element_labels,
        )

        nav_hints = self._ng.get_outbound_nav_hints(page_hash)

        current_observation = {
            "page_hash":      page_hash,
            "visit_count":    visit_count,
            "omni_boxes":     ui_elements,
            "unvisited_ids":  unvisited_ids,
            "visited_ids":    visited_ids,
            "total_elements": len(ui_elements),
            "nav_hints":      nav_hints,
        }

        packet = ContextPacket(
            screenshot=screenshot,
            task_goal=task,
            current_path=current_path,
            action_history=[s.to_dict() for s in self._wm.recent(self._history_window)],
            current_observation=current_observation,
            anomaly_flag=anomaly_flag,
            relevant_experience=relevant_experience,
            step=step,
            max_steps=self._max_steps,
        )

        logger.debug(
            "ContextPacket built: step=%d hash=%s unvisited=%d anomaly=%r",
            step, page_hash[:8], len(unvisited_ids),
            anomaly_flag[:40] if anomaly_flag else "none",
        )
        return packet
