from __future__ import annotations

"""OmniParser V2 视觉 Provider。

通过 HTTP 调用独立运行的 OmniParser 服务：
    python models/omni/omniparser.py
配置：OMNI_ENDPOINT=http://127.0.0.1:7861
"""

import ast
import logging
import os
import re
import tempfile
from typing import Any, Dict, List

from PIL import Image

from ..base import VisionProvider
from core.types import ElementType

logger = logging.getLogger(__name__)

_LABEL_MAP = {
    "icon":     ElementType.ICON,
    "text":     ElementType.TEXT,
    "button":   ElementType.BUTTON,
    "input":    ElementType.INPUT,
    "checkbox": ElementType.INPUT,
    "radio":    ElementType.INPUT,
}

# 匹配明显的 OCR 乱码：连续特殊符号、不可打印字符、大量无意义混合字符
_GARBLED_RE = re.compile(
    r'[\x00-\x08\x0b\x0c\x0e-\x1f]'   # 控制字符
    r'|^[^\w\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]{3,}$'  # 纯特殊符号 3 个以上
    r'|(.)\1{4,}',                       # 同一字符连续重复 5 次以上
    re.UNICODE,
)


def _clean_label(text: str) -> str:
    """清洗 OCR 产出的 label，将明显乱码替换为空串。"""
    text = text.strip()
    if not text:
        return ""
    # 含控制字符或明显乱码模式 → 清空
    if _GARBLED_RE.search(text):
        return ""
    # 可打印字符占比过低 → 视为乱码
    printable = sum(1 for c in text if c.isprintable())
    if len(text) > 2 and printable / len(text) < 0.5:
        return ""
    return text


def _to_1000(val: float, size: int) -> int:
    """将像素坐标转换为 0-1000 归一化坐标。"""
    return int(val / size * 1000)


def _normalize_bbox(bbox: List[float], w: int, h: int) -> List[int]:
    """将 OmniParser 输出的 bbox 统一转为 0-1000 归一化格式。

    OmniParser V2 输出: [x1, y1, x2, y2]，可能是绝对像素或 0-1 归一化。
    自动判断：若所有值均 <= 1.0 则认为是 0-1 归一化。
    """
    if all(0.0 <= v <= 1.0 for v in bbox) and any(v > 0 for v in bbox[2:]):
        return [
            int(bbox[0] * 1000),
            int(bbox[1] * 1000),
            int(bbox[2] * 1000),
            int(bbox[3] * 1000),
        ]
    return [
        _to_1000(bbox[0], w),
        _to_1000(bbox[1], h),
        _to_1000(bbox[2], w),
        _to_1000(bbox[3], h),
    ]


def _parse_omni_response(raw: List[Dict], w: int, h: int) -> List[Dict[str, Any]]:
    """将 OmniParser 原始输出统一化为 VisionProvider 标准格式。

    OmniParser V2 单条输出示例::

        {
            "type": "icon",
            "bbox": [0.12, 0.45, 0.28, 0.52],
            "interactable": true,
            "content": "设置",
            "source": "caption"
        }
    """
    result = []
    for idx, item in enumerate(raw):
        content = item.get("content") or item.get("label") or item.get("text") or ""
        raw_type = (item.get("type") or ElementType.UNKNOWN).lower()
        elem_type = _LABEL_MAP.get(raw_type, ElementType.UNKNOWN)

        bbox_raw = item.get("bbox", [0, 0, 0, 0])
        bbox = _normalize_bbox(bbox_raw, w, h)

        interactable = bool(item.get("interactivity", item.get("interactable", True)))
        result.append({
            "id":          idx,
            "bbox":        bbox,
            "label":       _clean_label(str(content)),
            "type":        elem_type,
            "interactable": interactable,
        })
    return result


# ─────────────────────────────────────────────────────────────
# HTTP 模式
# ─────────────────────────────────────────────────────────────

def _parse_omniparser_text(text: str) -> List[Dict[str, Any]]:
    """解析 OmniParser 文本输出为结构化列表。

    输出格式示例::

        icon 0: {'type': 'icon', 'bbox': [0.1, 0.2, 0.3, 0.4], 'interactable': True, 'content': '设置', 'source': 'caption'}
        icon 1: {'type': 'text', 'bbox': [...], ...}
    """
    items = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        colon = line.find(": {")
        if colon == -1:
            continue
        try:
            d = ast.literal_eval(line[colon + 2:])
            if isinstance(d, dict):
                items.append(d)
        except Exception:
            pass
    return items


# 过滤 Unity 调试叠层元素（FPS 计数器、Development Build 等无意义展示）
_DEBUG_LABEL_RE = re.compile(
    r'(\bfps\b|\bms\b.*(\bms\b|\d+\.)|\.\d+\s*ms|development\s*build)',
    re.IGNORECASE,
)


def _filter_debug_elements(elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Discard Unity runtime debug overlays (FPS counter, Development Build text)."""
    filtered = []
    for elem in elements:
        label = elem.get("label", "")
        if _DEBUG_LABEL_RE.search(label):
            logger.debug("过滤调试叠层元素: %r", label)
            continue
        filtered.append(elem)
    return filtered


class _HttpBackend:
    """通过 gradio_client 调用 OmniParser Gradio 服务的 /process 端点。"""

    def __init__(self, endpoint: str, timeout: int, imgsz: int = 1280, use_paddleocr: bool = False) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self.imgsz = imgsz
        self.use_paddleocr = use_paddleocr
        self._client = None

    def _get_client(self):
        if self._client is None:
            from gradio_client import Client
            self._client = Client(self.endpoint, verbose=False)
        return self._client

    def call(self, image: Image.Image) -> List[Dict]:
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()  # Windows: must close before another process can write the same path
        try:
            image.save(tmp.name)
            from gradio_client import handle_file
            for use_ocr in ([self.use_paddleocr, False] if self.use_paddleocr else [False]):
                try:
                    result = self._get_client().predict(
                        image_input=handle_file(tmp.name),
                        box_threshold=0.05,
                        iou_threshold=0.1,
                        use_paddleocr=use_ocr,
                        imgsz=self.imgsz,
                        api_name="/process",
                    )
                except Exception as exc:
                    if use_ocr:
                        logger.warning("PaddleOCR 调用失败，自动降级为 EasyOCR: %s", exc)
                        self.use_paddleocr = False
                        continue
                    raise
                else:
                    # Gradio 可能返回 None（服务端 PaddleOCR 静默崩溃）
                    if result is None or not hasattr(result, "__iter__"):
                        if use_ocr:
                            logger.warning("PaddleOCR 返回 None，自动降级为 EasyOCR")
                            self.use_paddleocr = False
                            continue
                        raise RuntimeError("OmniParser 返回空结果")
                    break  # 正常结果，退出重试循环
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

        if result is None:
            raise RuntimeError("OmniParser 返回空结果（PaddleOCR 与 EasyOCR 均失败）")
        _, text = result
        return _parse_omniparser_text(text)


# ─────────────────────────────────────────────────────────────
# Provider 主类
# ─────────────────────────────────────────────────────────────

class Provider(VisionProvider):
    """OmniParser V2 视觉 Provider。

    Args:
        endpoint: OmniParser 服务地址，默认 http://127.0.0.1:7861。
        timeout:  HTTP 请求超时（秒），默认 30。
    """

    def __init__(
        self,
        endpoint: str = "http://127.0.0.1:7861",
        timeout: int = 30,
        imgsz: int = 1280,
        use_paddleocr: bool = False,
    ) -> None:
        self._backend = _HttpBackend(endpoint, timeout, imgsz=imgsz, use_paddleocr=use_paddleocr)

    def detect(self, image: Image.Image) -> List[Dict[str, Any]]:
        w, h = image.size
        try:
            raw = self._backend.call(image)
        except Exception as exc:
            logger.error("OmniParser 检测失败: %s", exc)
            raise RuntimeError(f"OmniParser 检测失败: {exc}") from exc

        elements = _parse_omni_response(raw, w, h)
        before = len(elements)
        elements = _filter_debug_elements(elements)
        if before != len(elements):
            logger.debug("过滤调试叠层: %d → %d 个元素", before, len(elements))
        logger.debug("OmniParser 检测到 %d 个元素", len(elements))
        return elements

    def warm_up(self) -> None:
        logger.info("OmniParser V2 预热中（发送 1×1 空图）...")
        dummy = Image.new("RGB", (1, 1))
        try:
            self.detect(dummy)
        except Exception:
            pass

    def teardown(self) -> None:
        pass

    @classmethod
    def from_config(cls, config: "AgentConfig") -> "Provider":
        return cls(
            endpoint=config.vision.omni_endpoint,
            timeout=config.vision.omni_timeout,
            imgsz=config.vision.omni_imgsz,
            use_paddleocr=config.vision.omni_use_paddleocr,
        )

    def __repr__(self) -> str:
        ep = getattr(self._backend, "endpoint", "")
        return f"OmniV2Provider(endpoint={ep!r})"
