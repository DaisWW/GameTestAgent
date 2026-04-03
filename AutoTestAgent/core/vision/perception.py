from __future__ import annotations

"""感知工具模块 (Perception Utilities)

提供截图级别的低层感知能力，与具体视觉 Provider 解耦：
  - 感知哈希（phash）计算：对微小 UI 变化不敏感，防止误判"页面未变"
  - 动态检测：连续两帧 hash 相差多少 bit（汉明距离）
  - 卡死检测：连续多帧 hash 完全一致（UI 冻结）

设计原则：
  本模块只做图像级别的数值计算，不持有任何状态。
  所有状态管理交由 core/memory/ 负责。
"""

import io
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

_FREEZE_HAMMING_THRESHOLD = 4   # phash 差距 ≤ 4 bit 认为页面"未变化"


def compute_phash(image) -> str:
    """计算图片感知哈希（Perceptual Hash）。

    依赖 imagehash 库（列于 requirements.txt）。若未安装会直接 ImportError，
    请运行 ``pip install imagehash`` 或 ``setup.bat``。

    Args:
        image: PIL.Image.Image

    Returns:
        16 进制字符串哈希值，例如 "a1b2c3d4e5f67890"
    """
    import imagehash  # hard dependency：未安装则立即报错，避免 MD5 fallback 破坏循环检测
    return str(imagehash.phash(image))


def hamming_distance(hash_a: str, hash_b: str) -> int:
    """计算两个感知哈希之间的汉明距离（bit 差异数）。

    距离越小表示两张截图越相似。
    距离 = 0 → 完全相同；距离 > 10 → 明显页面变化。

    Args:
        hash_a: phash 字符串
        hash_b: phash 字符串

    Returns:
        整数汉明距离。如果哈希字符串为 MD5（非 phash），则退化为 0/1 比较。
    """
    if len(hash_a) != len(hash_b):
        return int(hash_a != hash_b)
    try:
        import imagehash
        return imagehash.hex_to_hash(hash_a) - imagehash.hex_to_hash(hash_b)
    except Exception:
        return int(hash_a != hash_b)


def is_page_changed(hash_before: str, hash_after: str, threshold: int = _FREEZE_HAMMING_THRESHOLD) -> bool:
    """判断两次截图之间页面是否发生了有意义的变化。

    Args:
        hash_before: 操作前的页面 phash
        hash_after:  操作后的页面 phash
        threshold:   汉明距离阈值，差异 > threshold 才算"变化"

    Returns:
        True = 页面已变化，False = 页面冻结/无变化
    """
    if not hash_before or not hash_after:
        return True
    dist = hamming_distance(hash_before, hash_after)
    changed = dist > threshold
    logger.debug(
        "PageChange: %s→%s dist=%d threshold=%d → %s",
        hash_before[:8], hash_after[:8], dist, threshold,
        "CHANGED" if changed else "FROZEN",
    )
    return changed


def detect_ui_freeze(hash_sequence: list, consecutive: int = 3, threshold: int = _FREEZE_HAMMING_THRESHOLD) -> Optional[str]:
    """检测 UI 是否卡死（连续多帧哈希几乎不变）。

    Args:
        hash_sequence: 历史 phash 列表（时序）
        consecutive:   判定卡死所需的连续相似帧数
        threshold:     汉明距离阈值

    Returns:
        卡死描述字符串；未卡死则返回 None。
    """
    if len(hash_sequence) < consecutive:
        return None

    recent = hash_sequence[-consecutive:]
    pairs  = [(recent[i], recent[i + 1]) for i in range(len(recent) - 1)]
    if all(hamming_distance(a, b) <= threshold for a, b in pairs):
        return (
            f"UI 疑似卡死：最近 {consecutive} 帧截图几乎一致（汉明距离均 ≤ {threshold}），"
            "请考虑强制重启游戏或执行恢复流程"
        )
    return None
