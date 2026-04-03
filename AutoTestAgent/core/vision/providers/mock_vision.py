from __future__ import annotations

"""Mock 视觉 Provider，用于开发 / 单元测试。

不加载任何模型，直接返回预设数据。
使用场景：
  - 调试 LangGraph 流程逻辑，无需开启 OmniParser（节省显存）
  - CI/CD 环境中的集成测试
  - 快速验证 Brain 决策逻辑

配置示例（.env）：
    VISION_TYPE=mock
"""

import logging
from typing import Any, Dict, List, Optional

from PIL import Image

from ..base import VisionProvider

logger = logging.getLogger(__name__)

_DEFAULT_ELEMENTS: List[Dict[str, Any]] = [
    {"id": 0, "bbox": [50,  100, 300, 150], "label": "开始游戏", "type": "button"},
    {"id": 1, "bbox": [50,  200, 300, 250], "label": "设置",     "type": "button"},
    {"id": 2, "bbox": [50,  300, 300, 350], "label": "退出",     "type": "button"},
    {"id": 3, "bbox": [100, 450, 900, 500], "label": "玩家名称", "type": "input"},
    {"id": 4, "bbox": [350,  20, 650,  80], "label": "游戏标题", "type": "text"},
]


class Provider(VisionProvider):
    """Mock 视觉 Provider。

    Args:
        elements: 固定返回的元素列表。为 None 时使用内置默认元素。
        sequence: 按调用顺序返回不同帧的元素列表（循环）。
                  设置后会覆盖 elements 参数。

    使用示例::

        from core.vision.providers.mock_vision import Provider

        vision = Provider(elements=[
            {"id": 0, "bbox": [10, 20, 100, 50], "label": "登录", "type": "button"},
        ])
    """

    def __init__(
        self,
        elements: Optional[List[Dict[str, Any]]] = None,
        sequence: Optional[List[List[Dict[str, Any]]]] = None,
    ) -> None:
        self._fixed = elements if elements is not None else _DEFAULT_ELEMENTS
        self._sequence = sequence
        self._call_count = 0

    def detect(self, image: Image.Image) -> List[Dict[str, Any]]:
        self._call_count += 1

        if self._sequence:
            idx = (self._call_count - 1) % len(self._sequence)
            result = self._sequence[idx]
        else:
            result = self._fixed

        logger.debug(
            "MockVision [call=%d] 返回 %d 个元素",
            self._call_count, len(result),
        )
        return result

    @classmethod
    def from_config(cls, config: "AgentConfig") -> "Provider":
        return cls()

    def __repr__(self) -> str:
        return f"MockVisionProvider(calls={self._call_count})"
