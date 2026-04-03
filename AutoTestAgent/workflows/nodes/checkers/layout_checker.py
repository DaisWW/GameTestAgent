from __future__ import annotations

"""维度 2：UI 布局异常检测 (Layout Checker)

基于 OmniParser 输出的 ui_elements bbox 做纯数学分析：
- 元素重叠（IoU > 阈值）
- 元素越界（bbox 超出屏幕范围 0-1000）
- 可点击元素过密（间距过小，易误触）
"""

import logging
from typing import Any, Dict, List, TYPE_CHECKING

from .base import BugChecker, BugReport

if TYPE_CHECKING:
    from core.agent.worker import LangGraphWorker

logger = logging.getLogger(__name__)

# ── 阈值配置 ────────────────────────────────────────────────────
_OVERLAP_IOU_THRESHOLD = 0.3     # IoU > 30% 认为重叠
_OOB_TOLERANCE = 10              # bbox 坐标 < -10 或 > 1010 认为越界
_MIN_GAP = 15                    # 可点击元素中心距 < 15（归一化坐标）认为过密

_CLICKABLE_TYPES = {"button", "icon", "input"}


def _compute_iou(a: List[int], b: List[int]) -> float:
    """计算两个 bbox [x1,y1,x2,y2] 的 IoU。"""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _center(bbox: List[int]):
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def _distance(c1, c2) -> float:
    return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5


class LayoutChecker(BugChecker):
    name = "layout"

    def is_applicable(self, state: dict) -> bool:
        return bool(state.get("ui_elements"))

    def check(self, state: dict, worker: "LangGraphWorker") -> List[BugReport]:
        elements = state.get("ui_elements", [])
        bugs: List[BugReport] = []

        clickable = [e for e in elements if e.get("type") in _CLICKABLE_TYPES]

        # ── 重叠检测 ────────────────────────────────────────────
        for i, a in enumerate(clickable):
            for b in clickable[i + 1:]:
                iou = _compute_iou(a["bbox"], b["bbox"])
                if iou > _OVERLAP_IOU_THRESHOLD:
                    bugs.append(BugReport(
                        category="layout",
                        severity="major",
                        description=(
                            f"元素重叠: [{a['id']}]{a.get('label', '')} "
                            f"与 [{b['id']}]{b.get('label', '')} "
                            f"IoU={iou:.2f}"
                        ),
                        tags=["overlap"],
                        evidence={"elem_a": a, "elem_b": b, "iou": round(iou, 3)},
                    ))

        # ── 越界检测 ────────────────────────────────────────────
        for e in elements:
            bbox = e.get("bbox", [0, 0, 0, 0])
            oob_dirs = []
            if bbox[0] < -_OOB_TOLERANCE:
                oob_dirs.append("left")
            if bbox[1] < -_OOB_TOLERANCE:
                oob_dirs.append("top")
            if bbox[2] > 1000 + _OOB_TOLERANCE:
                oob_dirs.append("right")
            if bbox[3] > 1000 + _OOB_TOLERANCE:
                oob_dirs.append("bottom")
            if oob_dirs:
                bugs.append(BugReport(
                    category="layout",
                    severity="minor",
                    description=(
                        f"元素越界: [{e['id']}]{e.get('label', '')} "
                        f"bbox={bbox} 超出屏幕({', '.join(oob_dirs)})"
                    ),
                    tags=["out_of_bounds"],
                    evidence={"element": e, "directions": oob_dirs},
                ))

        # ── 过密检测（可点击元素间距太小）──────────────────────
        for i, a in enumerate(clickable):
            for b in clickable[i + 1:]:
                dist = _distance(_center(a["bbox"]), _center(b["bbox"]))
                if dist < _MIN_GAP:
                    bugs.append(BugReport(
                        category="layout",
                        severity="minor",
                        description=(
                            f"元素过密: [{a['id']}]{a.get('label', '')} "
                            f"与 [{b['id']}]{b.get('label', '')} "
                            f"中心距={dist:.1f} (阈值{_MIN_GAP})"
                        ),
                        tags=["too_dense"],
                        evidence={"elem_a": a, "elem_b": b, "distance": round(dist, 1)},
                    ))

        return bugs
