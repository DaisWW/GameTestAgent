from __future__ import annotations

"""第一层：短期工作记忆 (Working Memory)

存储在 LangGraph State 中（纯内存），不持久化。
负责维护最近 N 步的"动作-反馈"序列，并提供循环检测能力。
"""

import logging
from collections import Counter, deque
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_LOOP_WINDOW = 6


@dataclass
class MemoryStep:
    """单步操作记录，是工作记忆的最小单元。"""
    step:           int
    action:         str
    params:         Dict[str, Any]
    reasoning:      str
    page_hash:      str
    result_hash:    str
    success:        bool
    element_label:  str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class WorkingMemory:
    """短期工作记忆管理器（滚动窗口，内存内，ABA 循环检测）。

    Args:
        capacity: 最大保留步数，默认 10。
    """

    def __init__(self, capacity: int = 10) -> None:
        self._capacity = capacity
        self._steps: deque[MemoryStep] = deque(maxlen=capacity)

    def push(self, step: MemoryStep) -> None:
        self._steps.append(step)
        logger.debug(
            "WorkingMemory push [step=%d] action=%s page=%s→%s",
            step.step, step.action, step.page_hash[:8], step.result_hash[:8] or "pending",
        )

    def update_last_result(self, result_hash: str, success: bool) -> None:
        if self._steps:
            self._steps[-1].result_hash = result_hash
            self._steps[-1].success = success

    @property
    def steps(self) -> List[MemoryStep]:
        return list(self._steps)

    def recent(self, n: int = 5) -> List[MemoryStep]:
        items = list(self._steps)
        return items[-n:] if n > 0 else []

    def recent_steps(self, n: int = 5) -> List[MemoryStep]:
        """recent() 的具名别名，语义更明确。"""
        return self.recent(n=n)

    def page_hash_sequence(self) -> List[str]:
        hashes = []
        for s in self._steps:
            hashes.append(s.page_hash)
            if s.result_hash:
                hashes.append(s.result_hash)
        return hashes

    def detect_aba_loop(self) -> Optional[str]:
        hashes = self.page_hash_sequence()[-_LOOP_WINDOW:]
        if len(hashes) < 4:
            return None
        counts = Counter(hashes)
        repeated = [h for h, c in counts.items() if c >= 3]
        if repeated:
            return (
                f"检测到页面循环：Hash {repeated[0][:8]} 在最近 {_LOOP_WINDOW} 步出现了 "
                f"{counts[repeated[0]]} 次，可能陷入 ABA 死循环"
            )
        return None

    def detect_stale_click(self, consecutive_threshold: int = 3) -> Optional[str]:
        if len(self._steps) < consecutive_threshold:
            return None
        recent = list(self._steps)[-consecutive_threshold:]
        if all(not s.success for s in recent):
            actions = [f"{s.action}({s.element_label})" for s in recent]
            return (
                f"最近 {consecutive_threshold} 步操作均未引起页面变化：{actions}，"
                "请考虑换一个策略"
            )
        return None

    def get_anomaly_flag(self) -> str:
        return self.detect_aba_loop() or self.detect_stale_click() or ""

    def __len__(self) -> int:
        return len(self._steps)

    def __repr__(self) -> str:
        return f"WorkingMemory(steps={len(self._steps)}, capacity={self._capacity})"
