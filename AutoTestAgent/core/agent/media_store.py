from __future__ import annotations

"""MediaStore：截图持久化管理。

将每步截图写入 runs/<run_id>/，将页面截图按 hash 写入 memory/screenshots/。
两类截图生命周期不同：
- 步骤截图：每次 run 独立，供问题复现
- 页面截图：跨 run 共享，供导航图关联
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    from config.settings import AgentConfig

logger = logging.getLogger(__name__)


class MediaStore:
    """负责两类截图的落盘，与业务逻辑解耦。"""

    def __init__(self, config: "AgentConfig") -> None:
        self._enabled = config.output.save_screenshots
        self._run_dir    = Path(config.run_dir)
        self._memory_dir = Path(config.memory_dir)

    def save_step(self, image: Image.Image, step: int) -> str:
        """将步骤截图保存到 runs/<run_id>/step_NNN.png。

        Returns: 保存路径，未启用时返回 ""。
        """
        if not self._enabled:
            return ""
        path = self._run_dir / f"step_{step:03d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(str(path))
        logger.debug("步骤截图: %s", path)
        return str(path)

    def save_page(self, image: Image.Image, page_hash: str) -> str:
        """将页面截图按 hash 保存到 memory/screenshots/，已存在则跳过。

        Returns: 保存路径，未启用时返回 ""。
        """
        if not self._enabled:
            return ""
        path = self._memory_dir / "screenshots" / f"{page_hash}.png"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(str(path))
            logger.debug("页面截图: %s", path)
        return str(path)

    def save_annotated(
        self,
        image: Image.Image,
        ui_elements: list,
        step: int,
    ) -> str:
        """在截图副本上绘制 UI 元素 bbox + ID + label，保存为 step_XXX_annotated.png。

        bbox 格式为 0-1000 归一化坐标，自动转换为像素坐标。
        不同元素类型使用不同颜色，不影响原始 image 对象。

        Returns: 保存路径，未启用时返回 ""。
        """
        if not self._enabled:
            return ""
        from PIL import ImageDraw, ImageFont

        _TYPE_COLOR = {
            "button":  "#FF4444",
            "icon":    "#4488FF",
            "text":    "#44BB44",
            "input":   "#FF9900",
            "unknown": "#AAAAAA",
        }

        w, h   = image.size
        canvas = image.copy().convert("RGBA")
        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        draw   = ImageDraw.Draw(overlay)

        try:
            font = ImageFont.truetype("arial.ttf", max(12, w // 80))
        except Exception:
            font = ImageFont.load_default()

        for elem in ui_elements:
            bbox  = elem.get("bbox", [0, 0, 0, 0])
            eid   = elem.get("id", "?")
            label = elem.get("label", "")[:20]
            etype = elem.get("type", "unknown")
            color = _TYPE_COLOR.get(etype, "#AAAAAA")

            x1 = int(bbox[0] / 1000 * w)
            y1 = int(bbox[1] / 1000 * h)
            x2 = int(bbox[2] / 1000 * w)
            y2 = int(bbox[3] / 1000 * h)

            draw.rectangle([x1, y1, x2, y2], outline=color + "FF", width=2)
            draw.rectangle([x1, y1, x1 + 22, y1 + 16], fill=color + "CC")
            draw.text((x1 + 2, y1 + 1), str(eid), fill="white", font=font)
            if label:
                draw.text((x1 + 2, y2 + 2), label, fill=color + "FF", font=font)

        result = Image.alpha_composite(canvas, overlay).convert("RGB")
        path   = self._run_dir / f"step_{step:03d}_annotated.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        result.save(str(path))
        logger.debug("标注截图已保存: %s", path)
        return str(path)
