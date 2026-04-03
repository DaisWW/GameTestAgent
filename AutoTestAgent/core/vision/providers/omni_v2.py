from __future__ import annotations

"""OmniParser V2 视觉 Provider。

支持两种运行模式（通过 mode 参数切换）：

1. **local**（默认）
   直接在当前进程加载 OmniParser 模型（需运行 models/omni/omniparser.bat 下载权重）。
   适合单机一体化部署，零额外进程。
   配置：OMNI_MODE=local  OMNI_MODEL_PATH=models/omni/OmniParser/weights

2. **http**
   在本机或远端另起 OmniParser 服务：
       python models/omni/omniparser.py
   本 Provider 通过 HTTP 调用，完全不占用当前进程显存。
   配置：OMNI_MODE=http  OMNI_ENDPOINT=http://127.0.0.1:7861
"""

import base64
import io
import json
import logging
from typing import Any, Dict, List, Optional

from PIL import Image

from ..base import VisionProvider

logger = logging.getLogger(__name__)

_LABEL_MAP = {
    "icon": "icon",
    "text": "text",
    "button": "button",
    "input": "input",
    "checkbox": "input",
    "radio": "input",
}


def _to_1000(val: float, size: int) -> int:
    """将像素坐标转换为 0-1000 归一化坐标。"""
    return int(val / size * 1000)


def _normalize_bbox(bbox: List[float], w: int, h: int) -> List[int]:
    """将 OmniParser 输出的 bbox 统一转为 0-1000 归一化格式。

    OmniParser V2 输出: [x1, y1, x2, y2]，可能是绝对像素或 0-1 归一化。
    自动判断：若所有值均 <= 1.0 则认为是 0-1 归一化。
    """
    if all(v <= 1.0 for v in bbox):
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
        raw_type = item.get("type", "unknown").lower()
        elem_type = _LABEL_MAP.get(raw_type, "unknown")

        bbox_raw = item.get("bbox", [0, 0, 0, 0])
        bbox = _normalize_bbox(bbox_raw, w, h)

        result.append({
            "id": idx,
            "bbox": bbox,
            "label": str(content).strip(),
            "type": elem_type,
        })
    return result


# ─────────────────────────────────────────────────────────────
# HTTP 模式
# ─────────────────────────────────────────────────────────────

class _HttpBackend:
    """通过 HTTP 调用独立运行的 OmniParser 服务。"""

    def __init__(self, endpoint: str, timeout: int) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout

    def call(self, image: Image.Image) -> List[Dict]:
        import requests

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        payload = {"image": img_b64}
        resp = requests.post(
            f"{self.endpoint}/process",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        # 兼容两种响应结构
        if isinstance(data, list):
            return data
        return data.get("parsed_content_list") or data.get("elements") or []


# ─────────────────────────────────────────────────────────────
# Local 模式
# ─────────────────────────────────────────────────────────────

class _LocalBackend:
    """在当前进程直接加载 OmniParser V2 模型（需安装 omniparser）。"""

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self._engine: Optional[Any] = None

    def _load(self) -> None:
        try:
            from omniparser import OmniParser  # type: ignore
            self._engine = OmniParser(model_path=self.model_path)
            logger.info("OmniParser V2 本地模型加载完成")
        except ImportError as exc:
            raise ImportError(
                "本地模式需要安装 omniparser 包。\n"
                "请参考 https://github.com/microsoft/OmniParser 安装，\n"
                "或改用 HTTP 模式（OMNI_MODE=http）。"
            ) from exc

    def call(self, image: Image.Image) -> List[Dict]:
        if self._engine is None:
            self._load()
        raw = self._engine.parse(image)
        if isinstance(raw, dict):
            return raw.get("parsed_content_list") or raw.get("elements") or []
        return raw or []

    def teardown(self) -> None:
        self._engine = None
        try:
            import torch
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            logger.info("OmniParser GPU 显存已释放")
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────
# Provider 主类
# ─────────────────────────────────────────────────────────────

class Provider(VisionProvider):
    """OmniParser V2 视觉 Provider。

    Args:
        mode:       "local"（默认）或 "http"。
        endpoint:   HTTP 模式下的服务地址，默认 http://127.0.0.1:7861。
        timeout:    HTTP 请求超时（秒），默认 30。
        model_path: Local 模式下的权重父目录，默认 "models/omni/OmniParser/weights"。
    """

    def __init__(
        self,
        mode: str = "local",
        endpoint: str = "http://127.0.0.1:7861",
        timeout: int = 30,
        model_path: str = "models/omni/OmniParser/weights",
    ) -> None:
        self.mode = mode
        if mode == "local":
            self._backend: _HttpBackend | _LocalBackend = _LocalBackend(model_path)
        else:
            self._backend = _HttpBackend(endpoint, timeout)

    def detect(self, image: Image.Image) -> List[Dict[str, Any]]:
        w, h = image.size
        try:
            raw = self._backend.call(image)
        except Exception as exc:
            logger.error("OmniParser 检测失败: %s", exc)
            raise RuntimeError(f"OmniParser 检测失败: {exc}") from exc

        elements = _parse_omni_response(raw, w, h)
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
        if isinstance(self._backend, _LocalBackend):
            self._backend.teardown()

    @classmethod
    def from_config(cls, config: "AgentConfig") -> "Provider":
        import os
        mode = os.getenv("OMNI_MODE", "local")
        return cls(
            mode=mode,
            endpoint=config.omni_endpoint,
            timeout=config.omni_timeout,
            model_path=config.omni_model_path,
        )

    def __repr__(self) -> str:
        if self.mode == "http":
            ep = getattr(self._backend, "endpoint", "")
            return f"OmniV2Provider(mode=http, endpoint={ep!r})"
        return f"OmniV2Provider(mode=local)"
