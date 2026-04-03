from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from PIL import Image


class VisionProvider(ABC):
    """视觉层抽象基类。

    所有视觉实现（OmniParser、PaddleOCR、YOLO 等）必须继承此类并实现 detect 方法。
    统一输出格式让上层 (waterfall_flow) 与具体模型完全解耦。
    """

    @abstractmethod
    def detect(self, image: Image.Image) -> List[Dict[str, Any]]:
        """检测屏幕中的 UI 元素。

        Args:
            image: PIL Image，来自 ADB 截图。

        Returns:
            元素列表，每项格式固定为::

                {
                    "id":    0,           # int，从 0 开始的序号
                    "bbox":  [x1,y1,x2,y2],  # 归一化坐标 0-1000（与 ADBController 对齐）
                    "label": "确定",       # str，人类可读的标签/文字
                    "type":  "button",    # str，元素类型: button|text|icon|input|unknown
                }
        """

    def warm_up(self) -> None:
        """可选：预热模型（第一次推理通常较慢）。默认不操作。"""

    def teardown(self) -> None:
        """可选：释放 GPU 显存或关闭连接。默认不操作。"""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
