from __future__ import annotations

"""核心业务数据结构。

定义跨模块传递的强类型数据类，替代散落在各处的 Dict[str, Any]：

- NavStats    — NavigationGraph 统计信息
- BugRecord   — 单条 Bug 快照（与 ExperiencePool bug_observations 表行对应）
- RunResult   — worker.run() 的完整返回结果
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, TYPE_CHECKING

from core.types import BugCategory, BugSeverity, BugTag, TestStatus

if TYPE_CHECKING:
    from core.memory.working_memory import MemoryStep


@dataclass
class NavStats:
    """NavigationGraph 的运行时统计快照。"""

    pages:            int = 0
    transitions:      int = 0
    visited_elements: int = 0
    cycles:           int = 0


@dataclass
class BugRecord:
    """单条 Bug 快照，与 experience_pool.bug_observations 表一行对应。"""

    page_hash:       str
    description:     str
    category:        BugCategory
    severity:        BugSeverity
    tags:            List[BugTag]        = field(default_factory=list)
    screenshot_path: str                 = ""
    evidence:        Dict[str, Any]      = field(default_factory=dict)
    created_at:      str                 = ""
    id:              int                 = 0

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "BugRecord":
        """从 SQLite Row / dict 反序列化。"""
        tags_raw = row.get("tags", "[]") or "[]"
        try:
            raw_list = json.loads(tags_raw) if isinstance(tags_raw, str) else list(tags_raw)
            tags = [BugTag(t) for t in raw_list if t in BugTag._value2member_map_]
        except Exception:
            tags = []

        evidence_raw = row.get("evidence", "{}") or "{}"
        try:
            evidence = json.loads(evidence_raw) if isinstance(evidence_raw, str) else dict(evidence_raw)
        except Exception:
            evidence = {}

        return cls(
            id              = int(row.get("id") or 0),
            page_hash       = row.get("page_hash", ""),
            description     = row.get("description", ""),
            category        = BugCategory(row.get("category") or BugCategory.CONTENT),
            severity        = BugSeverity(row.get("severity") or BugSeverity.MINOR),
            tags            = tags,
            screenshot_path = row.get("screenshot_path", "") or "",
            evidence        = evidence,
            created_at      = row.get("created_at", "") or "",
        )


@dataclass
class RunResult:
    """worker.run() 的完整测试结果。

    Attributes:
        status:      测试终态（pass / fail / error）。
        steps:       实际执行步数。
        reason:      结束原因描述（fail/error 时非空）。
        history:     每步操作记录列表（MemoryStep）。
        nav_stats:   导航图统计。
        bug_summary: 按 category 汇总的 Bug 数量字典，key 为类别字符串，
                     "total" 键为总数。
        bugs:        本轮新增 Bug 快照列表。
    """

    status:      TestStatus
    steps:       int
    reason:      str                      = ""
    history:     List["MemoryStep"]       = field(default_factory=list)
    nav_stats:   NavStats                 = field(default_factory=NavStats)
    bug_summary: Dict[str, int]           = field(default_factory=dict)
    bugs:        List[BugRecord]          = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.status == TestStatus.PASS

    @property
    def failed(self) -> bool:
        return self.status == TestStatus.FAIL

    @property
    def errored(self) -> bool:
        return self.status == TestStatus.ERROR
