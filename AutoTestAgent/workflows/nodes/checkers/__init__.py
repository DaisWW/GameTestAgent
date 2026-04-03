"""Bug Checker 注册表。

所有启用的 Checker 在此注册。check 节点遍历 CHECKER_CLASSES 列表执行检测。
新增维度只需：1) 新建 xxx_checker.py  2) 在此列表中添加类  3) 在 settings.py CheckerConfig 加开关。

使用 get_enabled_checkers(config) 获取根据配置过滤后的 checker 实例列表。
"""

from typing import List, TYPE_CHECKING

from .base import BugChecker
from .crash_checker import CrashChecker
from .layout_checker import LayoutChecker
from .visual_checker import VisualChecker
from .freeze_checker import FreezeChecker
from .functional_checker import FunctionalChecker
from .content_checker import ContentChecker

if TYPE_CHECKING:
    from config.settings import AgentConfig

# ── 按优先级排列：critical 维度靠前 ────────────────────────────
# (checker_name, checker_class) — name 与 CheckerConfig 字段名一致
CHECKER_REGISTRY = [
    ("crash",      CrashChecker),
    ("visual",     VisualChecker),
    ("freeze",     FreezeChecker),
    ("layout",     LayoutChecker),
    ("functional", FunctionalChecker),
    ("content",    ContentChecker),
]


def get_enabled_checkers(config: "AgentConfig") -> List[BugChecker]:
    """根据 config.checker 开关返回已启用的 checker 实例列表。"""
    checkers: List[BugChecker] = []
    for name, cls in CHECKER_REGISTRY:
        if getattr(config.checker, name, True):
            checkers.append(cls())
    return checkers


__all__ = ["CHECKER_REGISTRY", "get_enabled_checkers"]
