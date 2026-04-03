from __future__ import annotations

"""维度 6：文本 / 内容异常检测 (Content Checker)

基于 ui_elements 的 label 文本分析：
- 占位符泄露：label 中出现 TODO / placeholder / test / xxx / lorem
- 未翻译 key 泄露：label 形如 btn_xxx / txt_xxx / label_xxx
- 乱码 / 异常字符：label 中包含连续特殊字符
"""

import logging
import re
from typing import List, TYPE_CHECKING

from .base import BugChecker, BugReport

if TYPE_CHECKING:
    from core.agent.worker import LangGraphWorker

logger = logging.getLogger(__name__)

# ── 占位符关键词（小写匹配）────────────────────────────────────
_PLACEHOLDER_PATTERNS = [
    r"\btodo\b",
    r"\bplaceholder\b",
    r"\blorem\b",
    r"\bfixme\b",
    r"\btest_?\d*\b",
    r"\bdebug\b",
    r"^xxx+$",
    r"^null$",
    r"^undefined$",
    r"^none$",
]

# ── 未翻译 key 的模式 ──────────────────────────────────────────
_I18N_KEY_PATTERN = re.compile(
    r"^[a-z][a-z0-9]*(_[a-z0-9]+){2,}$"  # 形如 btn_confirm_text
)

# ── 乱码检测：连续 3+ 个非常见字符 ──────────────────────────────
_GARBLED_PATTERN = re.compile(
    r"[^\w\s\u4e00-\u9fff\u3000-\u303f\uff00-\uffef.,;:!?()（）【】\-]{3,}"
)


class ContentChecker(BugChecker):
    name = "content"

    def is_applicable(self, state: dict) -> bool:
        return bool(state.get("ui_elements"))

    def check(self, state: dict, worker: "LangGraphWorker") -> List[BugReport]:
        elements = state.get("ui_elements", [])
        bugs: List[BugReport] = []

        for e in elements:
            label = (e.get("label") or "").strip()
            if not label or len(label) < 2:
                continue

            label_lower = label.lower()

            # ── 占位符检测 ──────────────────────────────────────
            for pat in _PLACEHOLDER_PATTERNS:
                if re.search(pat, label_lower):
                    bugs.append(BugReport(
                        category="content",
                        severity="minor",
                        description=(
                            f"疑似占位符泄露: [{e['id']}] \"{label}\""
                        ),
                        tags=["placeholder"],
                        evidence={"element": e, "pattern": pat},
                    ))
                    break  # 一个元素只报一次占位符

            # ── 未翻译 key 检测 ─────────────────────────────────
            if _I18N_KEY_PATTERN.match(label_lower):
                bugs.append(BugReport(
                    category="content",
                    severity="minor",
                    description=(
                        f"疑似未翻译的 i18n key: [{e['id']}] \"{label}\""
                    ),
                    tags=["untranslated_key"],
                    evidence={"element": e},
                ))

            # ── 乱码检测 ────────────────────────────────────────
            if _GARBLED_PATTERN.search(label):
                bugs.append(BugReport(
                    category="content",
                    severity="major",
                    description=(
                        f"疑似乱码: [{e['id']}] \"{label[:30]}\""
                    ),
                    tags=["garbled_text"],
                    evidence={"element": e},
                ))

        return bugs
