from __future__ import annotations

"""维度 3：画面异常检测 (Visual Checker)

基于截图像素统计分析：
- 黑屏：像素均值极低
- 白屏 / 纯色屏：像素标准差极低
- 大面积纯色区块（纹理丢失，游戏常见 bug）
"""

import logging
from typing import List, TYPE_CHECKING

import numpy as np

from .base import BugChecker, BugReport

if TYPE_CHECKING:
    from core.agent.worker import LangGraphWorker

logger = logging.getLogger(__name__)

# ── 阈值配置 ────────────────────────────────────────────────────
_BLACK_MEAN_THRESHOLD = 10       # 像素均值 < 10 认为黑屏
_LOW_STD_THRESHOLD = 5           # 像素标准差 < 5 认为纯色屏
_SOLID_BLOCK_RATIO = 0.4         # 超过 40% 面积为同一纯色 → 纹理丢失


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

        # ── 黑屏检测 ────────────────────────────────────────────
        if mean_val < _BLACK_MEAN_THRESHOLD:
            bugs.append(BugReport(
                category="visual",
                severity="critical",
                description=f"疑似黑屏: 像素均值={mean_val:.1f}",
                tags=["black_screen"],
                evidence={"mean": round(mean_val, 2), "std": round(std_val, 2)},
            ))
            return bugs  # 黑屏时无需继续其他视觉检测

        # ── 纯色屏 / 白屏检测 ──────────────────────────────────
        if std_val < _LOW_STD_THRESHOLD:
            color_desc = "白屏" if mean_val > 240 else "纯色屏"
            bugs.append(BugReport(
                category="visual",
                severity="major",
                description=f"疑似{color_desc}: 像素均值={mean_val:.1f} 标准差={std_val:.1f}",
                tags=["solid_screen"],
                evidence={"mean": round(mean_val, 2), "std": round(std_val, 2)},
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
                bugs.append(BugReport(
                    category="visual",
                    severity="minor",
                    description=(
                        f"大面积纯色区块: 最大占比={max_ratio:.0%}，"
                        f"可能为纹理丢失或渲染异常"
                    ),
                    tags=["solid_block", "texture_missing"],
                    evidence={"max_color_ratio": round(max_ratio, 3)},
                ))
        except Exception as exc:
            logger.debug("纯色区块检测异常: %s", exc)

        return bugs
