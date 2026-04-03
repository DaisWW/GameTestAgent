from __future__ import annotations

"""Grounding DINO 视觉 Provider。

开放词汇检测模型，通过文本提示（text prompt）在截图中检测 UI 元素。
无需启动外部服务，模型在进程内直接加载推理。

模型: IDEA-Research/grounding-dino-tiny  (~172M 参数, ~700MB)
VRAM: ~1.5-2GB
推理: ~300-800ms / 帧

配置:
    VISION_TYPE=grounding_dino
    GDINO_TEXT_PROMPT=button . icon . close . back .
    GDINO_BOX_THRESHOLD=0.25
    GDINO_TEXT_THRESHOLD=0.25
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from ..base import VisionProvider
from core.types import ElementType

logger = logging.getLogger(__name__)

_MODELS_DIR = Path(__file__).parent.parent.parent.parent / "models" / "grounding_dino" / "cache"

# Grounding DINO 检测标签 → 统一 ElementType 映射
_LABEL_MAP = {
    "button":  ElementType.BUTTON,
    "icon":    ElementType.ICON,
    "text":    ElementType.TEXT,
    "input":   ElementType.INPUT,
    "close":   ElementType.ICON,
    "back":    ElementType.ICON,
    "play":    ElementType.BUTTON,
    "settings": ElementType.ICON,
    "menu":    ElementType.ICON,
    "confirm": ElementType.BUTTON,
    "cancel":  ElementType.BUTTON,
    "ok":      ElementType.BUTTON,
    "start":   ElementType.BUTTON,
    "tab":     ElementType.BUTTON,
    "switch":  ElementType.INPUT,
    "slider":  ElementType.INPUT,
    "checkbox": ElementType.INPUT,
}

# 默认文本提示：覆盖游戏中常见的可交互 UI 元素类型
DEFAULT_TEXT_PROMPT = "button . icon . close . back . text . input ."

# 过滤 Unity 调试叠层元素
_DEBUG_LABEL_RE = re.compile(
    r'(\bfps\b|\bms\b.*(\bms\b|\d+\.)|\.\d+\s*ms|development\s*build)',
    re.IGNORECASE,
)


def _to_1000(val: float, size: int) -> int:
    """将像素坐标转换为 0-1000 归一化坐标。"""
    return max(0, min(1000, int(val / size * 1000)))


class Provider(VisionProvider):
    """Grounding DINO 视觉 Provider。

    开放词汇目标检测，通过文本引导在游戏截图中定位 UI 元素。
    模型在首次 detect() 时懒加载。

    Args:
        model_id:       HuggingFace 模型 ID 或本地路径。
        text_prompt:    检测提示词，以 " . " 分隔各类别。
        box_threshold:  bbox 置信度阈值 (0-1)。
        text_threshold: 文本匹配阈值 (0-1)。
        device:         推理设备，None 则自动选择 cuda/cpu。
    """

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-tiny",
        text_prompt: str = DEFAULT_TEXT_PROMPT,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        device: Optional[str] = None,
    ) -> None:
        self._model_id = model_id
        self._text_prompt = text_prompt
        self._box_threshold = box_threshold
        self._text_threshold = text_threshold
        self._device_name = device
        # 懒加载
        self._processor = None
        self._model = None
        self._device = None

    def _ensure_loaded(self) -> None:
        """懒加载模型和处理器。"""
        if self._model is not None:
            return

        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        # 优先使用本地缓存
        if _MODELS_DIR.exists() and (_MODELS_DIR / "config.json").exists():
            model_path = str(_MODELS_DIR)
            logger.info("从本地缓存加载 Grounding DINO: %s", model_path)
        else:
            model_path = self._model_id
            logger.info("从 HuggingFace 加载 Grounding DINO: %s", model_path)

        if self._device_name:
            self._device = torch.device(self._device_name)
        else:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("推理设备: %s", self._device)

        self._processor = AutoProcessor.from_pretrained(model_path)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_path
        ).to(self._device).eval()

        logger.info("Grounding DINO 加载完成 (%s)", self._device)

    def detect(self, image: Image.Image) -> List[Dict[str, Any]]:
        import torch

        self._ensure_loaded()

        w, h = image.size

        inputs = self._processor(
            images=image,
            text=self._text_prompt,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self._box_threshold,
            text_threshold=self._text_threshold,
            target_sizes=[(h, w)],
        )

        if not results:
            logger.debug("Grounding DINO 未检测到任何元素")
            return []

        det = results[0]  # 单张图片
        boxes  = det["boxes"]    # tensor [N, 4]  (x1, y1, x2, y2) 像素坐标
        scores = det["scores"]   # tensor [N]
        labels = det["labels"]   # list[str]

        elements = []
        for idx in range(len(boxes)):
            box = boxes[idx].tolist()
            label_text = labels[idx].strip().lower()
            score = scores[idx].item()

            # 映射到统一 ElementType
            elem_type = _LABEL_MAP.get(label_text, ElementType.UNKNOWN)

            # 归一化到 0-1000 坐标系
            bbox = [
                _to_1000(box[0], w),
                _to_1000(box[1], h),
                _to_1000(box[2], w),
                _to_1000(box[3], h),
            ]

            elements.append({
                "id":          idx,
                "bbox":        bbox,
                "label":       label_text,
                "type":        elem_type,
                "interactable": elem_type != ElementType.TEXT,
                "confidence":  round(score, 3),
            })

        # 过滤调试叠层
        before = len(elements)
        elements = [e for e in elements if not _DEBUG_LABEL_RE.search(e.get("label", ""))]
        if before != len(elements):
            logger.debug("过滤调试叠层: %d → %d 个元素", before, len(elements))

        # 按置信度降序排列，重新编号
        elements.sort(key=lambda e: e["confidence"], reverse=True)
        for i, e in enumerate(elements):
            e["id"] = i

        logger.debug("Grounding DINO 检测到 %d 个元素", len(elements))
        return elements

    def warm_up(self) -> None:
        logger.info("Grounding DINO 预热中...")
        dummy = Image.new("RGB", (100, 100), color=(128, 128, 128))
        try:
            self.detect(dummy)
        except Exception:
            pass

    def teardown(self) -> None:
        """释放 GPU 显存。"""
        if self._model is not None:
            import torch
            del self._model
            del self._processor
            self._model = None
            self._processor = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Grounding DINO 已释放显存")

    @classmethod
    def from_config(cls, config: "AgentConfig") -> "Provider":
        return cls(
            model_id=config.vision.gdino_model,
            text_prompt=config.vision.gdino_text_prompt,
            box_threshold=config.vision.gdino_box_threshold,
            text_threshold=config.vision.gdino_text_threshold,
        )

    def __repr__(self) -> str:
        return (
            f"GroundingDINOProvider("
            f"model={self._model_id!r}, "
            f"prompt={self._text_prompt!r}, "
            f"box_thr={self._box_threshold})"
        )
