from __future__ import annotations

"""维度 3：画面异常检测 (Visual Checker)

基于截图像素统计分析：
- 黑屏：像素均值极低
- 白屏 / 纯色屏：像素标准差极低
- 大面积纯色区块（纹理丢失，游戏常见 bug）
"""

import logging
from typing import List, Optional, TYPE_CHECKING

import numpy as np

from .base import BugChecker, BugReport
from core.types import BugCategory, BugSeverity, BugTag

if TYPE_CHECKING:
    from core.agent.worker import LangGraphWorker

logger = logging.getLogger(__name__)

# ── 阈值配置 ────────────────────────────────────────────────────
_BLACK_MEAN_THRESHOLD = 10       # 像素均值 < 10 认为黑屏
_LOW_STD_THRESHOLD = 5           # 像素标准差 < 5 认为纯色屏
_SOLID_BLOCK_RATIO = 0.4         # 超过 40% 面积为同一纯色 → 纹理丢失

# ── 加载画面关键词（UI 元素 label 中含有这些词则认为是加载中）──
_LOADING_KEYWORDS = frozenset([
    "加载", "loading", "载入", "读取", "please wait", "connecting",
    "正在加载", "正在连接", "初始化", "initializing", "进度", "progress",
    "login", "登录中", "登陆中", "reconnecting",
])

# ── VLM 二次确认 prompt ─────────────────────────────────────────
_VLM_CONFIRM_PROMPT = (
    "这是一张手游截图。请判断画面当前是【正常的加载/过渡界面】还是【真实的渲染 Bug（如纹理丢失、颜色块异常）】。\n"
    "请仅回复以下之一：LOADING 或 BUG，不要附加任何说明。"
)


def _is_loading_screen(state: dict) -> bool:
    """根据当前 UI 元素文本判断是否处于加载/过渡画面。"""
    elements = state.get("ui_elements") or []
    for elem in elements:
        label = (elem.get("label") or "").lower()
        if any(kw in label for kw in _LOADING_KEYWORDS):
            return True
    return False


def _vlm_confirm_is_bug(screenshot, worker: "LangGraphWorker") -> Optional[bool]:
    """调用 VLM 判断截图是真实 Bug 还是加载画面。

    Returns:
        True  → VLM 确认是 Bug；
        False → VLM 认为是加载画面；
        None  → VLM 不可用或调用失败（调用方应保守处理，视为 Bug）。
    """
    try:
        reply = worker.llm.ask_vision(screenshot, _VLM_CONFIRM_PROMPT)
        if reply is None:
            return None
        reply_upper = reply.strip().upper()
        logger.debug("VLM 视觉确认回复: %r", reply[:80])
        if "LOADING" in reply_upper:
            return False
        if "BUG" in reply_upper:
            return True
        return None  # 无法解析，保守处理
    except Exception as exc:
        logger.debug("VLM 视觉确认异常: %s", exc)
        return None


class VisualChecker(BugChecker):
    name = "visual"

    def is_applicable(self, state: dict) -> bool:
        return state.get("screenshot") is not None

    def check(self, state: dict, worker: "LangGraphWorker") -> List[BugReport]:
        screenshot = state.get("screenshot")
        if screenshot is None:
            return []

        bugs: List[BugReport] = []
        arr = np.array(screenshot.convert("RGB"))

        mean_val = float(arr.mean())
        std_val = float(arr.std())

        loading = _is_loading_screen(state)

        # ── 黑屏检测 ────────────────────────────────────────────
        if mean_val < _BLACK_MEAN_THRESHOLD:
            if loading:
                logger.info("[VisualChecker] 黑屏疑似加载画面（UI 含加载关键词），跳过")
            else:
                confirmed = _vlm_confirm_is_bug(screenshot, worker)
                if confirmed is False:
                    logger.info("[VisualChecker] 黑屏经 VLM 确认为加载画面，跳过")
                else:
                    bugs.append(BugReport(
                        category=BugCategory.VISUAL,
                        severity=BugSeverity.CRITICAL,
                        description=f"疑似黑屏: 像素均值={mean_val:.1f}",
                        tags=[BugTag.BLACK_SCREEN],
                        evidence={"mean": round(mean_val, 2), "std": round(std_val, 2),
                                  "vlm_confirmed": confirmed},
                    ))
            return bugs  # 黑屏（或被过滤的加载黑屏）时无需继续其他视觉检测

        # ── 纯色屏 / 白屏检测 ──────────────────────────────────
        if std_val < _LOW_STD_THRESHOLD:
            if loading:
                logger.info("[VisualChecker] 纯色屏疑似加载画面（UI 含加载关键词），跳过")
                return bugs
            confirmed = _vlm_confirm_is_bug(screenshot, worker)
            if confirmed is False:
                logger.info("[VisualChecker] 纯色屏经 VLM 确认为加载画面，跳过")
                return bugs
            color_desc = "白屏" if mean_val > 240 else "纯色屏"
            bugs.append(BugReport(
                category=BugCategory.VISUAL,
                severity=BugSeverity.MAJOR,
                description=f"疑似{color_desc}: 像素均值={mean_val:.1f} 标准差={std_val:.1f}",
                tags=[BugTag.SOLID_SCREEN],
                evidence={"mean": round(mean_val, 2), "std": round(std_val, 2),
                          "vlm_confirmed": confirmed},
            ))
            return bugs

        # ── 大面积纯色区块检测（纹理丢失）──────────────────────
        # 将图片缩到小尺寸加速，按 8x8 区块统计
        try:
            small = screenshot.convert("RGB").resize((64, 64))
            small_arr = np.array(small)
            # 量化为 8 级，减少噪声影响
            quantized = (small_arr // 32).astype(np.uint8)
            # 将每个像素的 RGB 映射为唯一整数
            flat = quantized[:, :, 0].astype(np.int32) * 100 + \
                   quantized[:, :, 1].astype(np.int32) * 10 + \
                   quantized[:, :, 2].astype(np.int32)
            total_pixels = flat.size
            unique, counts = np.unique(flat, return_counts=True)
            max_ratio = float(counts.max()) / total_pixels
            if max_ratio > _SOLID_BLOCK_RATIO:
                if loading:
                    logger.info(
                        "[VisualChecker] 大面积色块(%.0f%%)疑似加载画面（UI 含加载关键词），跳过",
                        max_ratio * 100,
                    )
                else:
                    confirmed = _vlm_confirm_is_bug(screenshot, worker)
                    if confirmed is False:
                        logger.info(
                            "[VisualChecker] 大面积色块(%.0f%%)经 VLM 确认为加载画面，跳过",
                            max_ratio * 100,
                        )
                    else:
                        bugs.append(BugReport(
                            category=BugCategory.VISUAL,
                            severity=BugSeverity.MINOR,
                            description=(
                                f"大面积纯色区块: 最大占比={max_ratio:.0%}，"
                                f"可能为纹理丢失或渲染异常"
                            ),
                            tags=[BugTag.SOLID_BLOCK, BugTag.TEXTURE_MISSING],
                            evidence={"max_color_ratio": round(max_ratio, 3),
                                      "vlm_confirmed": confirmed},
                        ))
        except Exception as exc:
            logger.debug("纯色区块检测异常: %s", exc)

        return bugs
