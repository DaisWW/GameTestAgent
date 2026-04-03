from __future__ import annotations

"""核心领域枚举类型定义。

集中管理跨模块共享的字符串常量，用枚举替代散落各处的 magic string，
避免因拼写错误或遗漏修改导致运行时静默出错。

所有枚举均继承 ``str``，因此枚举值可以直接当普通字符串使用
（JSON 序列化、TypedDict 键匹配、LangGraph 路由字符串等均无需额外转换）。
"""

from enum import StrEnum


class ActionType(StrEnum):
    """LLM / SequentialDecider 输出的动作类型。"""

    TAP        = "tap"
    SWIPE      = "swipe"
    INPUT_TEXT = "input_text"
    PRESS_BACK = "press_back"
    PRESS_HOME = "press_home"
    WAIT       = "wait"
    DONE       = "done"


class BugCategory(StrEnum):
    """Bug 检测维度分类（与 CheckerConfig 字段名 / BugChecker.name 保持一致）。"""

    CRASH      = "crash"
    LAYOUT     = "layout"
    VISUAL     = "visual"
    FREEZE     = "freeze"
    FUNCTIONAL = "functional"
    CONTENT    = "content"


class BugSeverity(StrEnum):
    """Bug 严重程度（CRITICAL 自动触发恢复子流程）。"""

    CRITICAL = "critical"
    MAJOR    = "major"
    MINOR    = "minor"


class TestStatus(StrEnum):
    """测试执行结果的终态。"""

    PASS  = "pass"
    FAIL  = "fail"
    ERROR = "error"


class ElementType(StrEnum):
    """视觉 Provider 返回的 UI 元素类型。"""

    BUTTON  = "button"
    ICON    = "icon"
    TEXT    = "text"
    INPUT   = "input"
    UNKNOWN = "unknown"


class LLMProvider(StrEnum):
    """LLM 提供商标识，与 .env LLM_PROVIDER 值对应。"""

    OPENAI     = "openai"
    ANTHROPIC  = "anthropic"
    GOOGLE     = "google"
    SEQUENTIAL = "sequential"


class VisionType(StrEnum):
    """视觉 Provider 标识，与 .env VISION_TYPE 值对应。"""

    OMNI_V2        = "omni_v2"
    GROUNDING_DINO = "grounding_dino"
    MOCK           = "mock"


class SwipeDirection(StrEnum):
    """滑动方向（executor swipe 动作参数）。"""

    UP    = "up"
    DOWN  = "down"
    LEFT  = "left"
    RIGHT = "right"


class ExperienceType(StrEnum):
    """经验池条目类型（get_relevant_experience 返回列表中的 type 字段）。"""

    PATH      = "path"
    BUG       = "bug"
    KNOWLEDGE = "knowledge"


class RecoveryStrategy(StrEnum):
    """异常恢复策略（按优先级顺序尝试）。"""

    WAIT        = "wait"
    PRESS_HOME  = "press_home"
    LAUNCH_GAME = "launch_game"
    ABANDON     = "abandon"


class BugTag(StrEnum):
    """Bug 细分标签（BugReport.tags 列表中的标准值）。"""

    CRASH             = "crash"
    OVERLAP           = "overlap"
    OUT_OF_BOUNDS     = "out_of_bounds"
    TOO_DENSE         = "too_dense"
    BLACK_SCREEN      = "black_screen"
    SOLID_SCREEN      = "solid_screen"
    SOLID_BLOCK       = "solid_block"
    TEXTURE_MISSING   = "texture_missing"
    UI_FREEZE         = "ui_freeze"
    ABA_LOOP          = "aba_loop"
    ANOMALY           = "anomaly"
    NO_RESPONSE_STREAK = "no_response_streak"
    DEAD_END          = "dead_end"
    NO_BACK_BUTTON    = "no_back_button"
    TAP_NO_RESPONSE   = "tap_no_response"
    PLACEHOLDER       = "placeholder"
    UNTRANSLATED_KEY  = "untranslated_key"
    GARBLED_TEXT      = "garbled_text"
