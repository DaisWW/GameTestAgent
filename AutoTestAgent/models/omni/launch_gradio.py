"""OmniParser 中文 OCR 启动器（不修改 OmniParser 源码）。

从 models/omni/ 目录启动，通过 monkey-patch 在 gradio_demo.py
加载前替换 EasyOCR/PaddleOCR 的语言配置。

OmniParser/ 目录保持原样，重新 clone 不受影响。

环境变量:
    OMNI_OCR_LANGS    — EasyOCR 语言列表，逗号分隔，默认 ch_sim,en
    OMNI_PADDLE_LANG  — PaddleOCR 语言代码，默认 ch
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# ── 0. 禁用 OneDNN（解决 Windows 上 PaddleOCR 崩溃）────────────
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_use_onednn", "0")
os.environ.setdefault("FLAGS_use_gpu", "0")

# ── 1. 将 OmniParser 根目录加入 sys.path ───────────────────────
_OMNI = Path(__file__).parent / "OmniParser"
if not _OMNI.exists():
    print(f"[!] OmniParser 目录不存在: {_OMNI}", flush=True)
    sys.exit(1)
# 确保 OmniParser 在 sys.path 最前，使 gradio_demo.py 中的
# `from util.utils import ...` 能正确解析
if str(_OMNI) not in sys.path:
    sys.path.insert(0, str(_OMNI))
# gradio_demo.py 默认 cwd 必须是 OmniParser/（权重路径是相对路径）
os.chdir(str(_OMNI))

# ── 2. 读取语言配置 ─────────────────────────────────────────────
_easyocr_langs = os.getenv("OMNI_OCR_LANGS", "ch_sim,en").split(",")
_paddle_lang   = os.getenv("OMNI_PADDLE_LANG", "ch")

# ── 3. Monkey-patch OCR 实例 ────────────────────────────────────
import easyocr

print(f"[OCR-patch] EasyOCR langs = {_easyocr_langs}", flush=True)
_reader = easyocr.Reader(_easyocr_langs)

_paddle = None
try:
    from paddleocr import PaddleOCR
    print(f"[OCR-patch] PaddleOCR lang = {_paddle_lang}", flush=True)
    _paddle = PaddleOCR(
        lang=_paddle_lang,
        use_angle_cls=False,
        use_gpu=False,
        show_log=False,
        max_batch_size=1024,
        use_dilation=True,
        det_db_score_mode="slow",
        rec_batch_num=1024,
    )
    print("[OCR-patch] PaddleOCR OK", flush=True)
except Exception as exc:
    print(f"[OCR-patch] PaddleOCR 不可用（仅使用 EasyOCR）: {exc}", flush=True)

# import utils 触发原始初始化，随后覆盖全局变量
from util import utils as _utils
_utils.reader = _reader
if _paddle is not None:
    _utils.paddle_ocr = _paddle

# ── 4. 修复 OmniParser 源码的空结果 bug ─────────────────────────
#   check_ocr_box: PaddleOCR 对空白图返回 [None]，迭代 None 崩溃
#   get_som_labeled_img: ocr_bbox=[] 被转为 None，zip(None,...) 崩溃
_orig_check_ocr_box = _utils.check_ocr_box

def _safe_check_ocr_box(*args, **kwargs):
    """包装 check_ocr_box，确保返回值永远是可迭代的列表。"""
    try:
        (text, bb), goal = _orig_check_ocr_box(*args, **kwargs)
    except (TypeError, AttributeError):
        # PaddleOCR 返回 None 导致内部崩溃，回退空结果
        text, bb, goal = [], [], None
    if text is None:
        text = []
    if bb is None:
        bb = []
    return (text, bb), goal

_utils.check_ocr_box = _safe_check_ocr_box
print("[OCR-patch] 中文 OCR 已就绪 ✓", flush=True)

# ── 5. 启动 gradio_demo（从 OmniParser/ 目录）──────────────────
import gradio_demo  # noqa: F401
