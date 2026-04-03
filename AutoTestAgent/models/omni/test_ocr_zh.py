"""
测试中文 OCR 识别效果。

功能：
  1. 生成一张含中英文文本的测试图片
  2. 直接调用 EasyOCR / PaddleOCR 本地引擎对比（不依赖 Gradio 服务）
  3. 分别测试 英文模式(原版) 和 中文模式(patch后) 的识别效果

用法:
  cd models/omni/OmniParser
  python ../test_ocr_zh.py

或直接:
  python models/omni/test_ocr_zh.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# ── 禁用 OneDNN（解决 Windows 上 PaddleOCR 崩溃）────────────
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_use_gpu", "0")

# ── 确保能 import OmniParser 的 util ──────────────────────
_OMNI = Path(__file__).parent / "OmniParser"
if str(_OMNI) not in sys.path:
    sys.path.insert(0, str(_OMNI))

# ── 生成测试图片 ─────────────────────────────────────────────
from PIL import Image, ImageDraw, ImageFont

_TEST_TEXTS = [
    "开始游戏",
    "设置",
    "退出",
    "Hello World",
    "确认购买",
    "返回主页",
    "Loading...",
    "等级 Lv.99",
]


def _make_test_image(save_path: str = "") -> Image.Image:
    """生成一张 720x1280 的模拟游戏 UI 截图，包含中英文文本。"""
    w, h = 720, 1280
    img = Image.new("RGB", (w, h), color=(30, 30, 50))
    draw = ImageDraw.Draw(img)

    # 尝试加载中文字体
    font = None
    for name in ("msyh.ttc", "msyhbd.ttc", "simsun.ttc", "simhei.ttf", "arial.ttf"):
        try:
            font = ImageFont.truetype(name, 36)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()

    # 绘制文本按钮
    y_offset = 80
    for i, text in enumerate(_TEST_TEXTS):
        x = 100
        y = y_offset + i * 140
        # 按钮背景
        draw.rounded_rectangle([x, y, x + 520, y + 80], radius=12, fill=(60, 60, 100))
        # 文本
        draw.text((x + 30, y + 18), text, fill="white", font=font)

    if save_path:
        img.save(save_path)
        print(f"[测试图片] 已保存: {save_path}")
    return img


# ── OCR 测试 ─────────────────────────────────────────────────

def _test_easyocr(img_path: str, langs: list[str], label: str):
    """测试 EasyOCR 指定语言的识别效果。"""
    import easyocr
    import numpy as np

    print(f"\n{'='*56}")
    print(f"  EasyOCR  langs={langs}  ({label})")
    print(f"{'='*56}")

    t0 = time.time()
    reader = easyocr.Reader(langs, verbose=False)
    img = np.array(Image.open(img_path).convert("RGB"))
    results = reader.readtext(img, paragraph=False, text_threshold=0.5)
    elapsed = time.time() - t0

    print(f"  耗时: {elapsed:.1f}s  |  识别到 {len(results)} 段文本\n")
    for i, (bbox, text, conf) in enumerate(results):
        status = "✓" if any(ch in text for ch in "游戏设置退出确认购买返回等级") or text.strip() in [t.strip() for t in _TEST_TEXTS] else "?"
        print(f"  [{i:2d}] conf={conf:.2f}  {status}  \"{text}\"")

    # 统计命中（严格匹配：识别结果必须包含期望文本的至少 60% 字符）
    recognized = set()
    for _, text, _ in results:
        text_clean = text.strip()
        for expected in _TEST_TEXTS:
            # 英文匹配忽略大小写，中文精确匹配
            if expected.lower() == text_clean.lower():
                recognized.add(expected)
            elif len(expected) >= 2:
                matched_chars = sum(1 for c in expected if c in text_clean)
                if matched_chars / len(expected) >= 0.6:
                    recognized.add(expected)
    hit = len(recognized)
    total = len(_TEST_TEXTS)
    print(f"\n  命中: {hit}/{total}  ({hit/total*100:.0f}%)")
    return hit, total


def _test_paddleocr(img_path: str, lang: str, label: str):
    """测试 PaddleOCR 指定语言的识别效果。"""
    print(f"\n{'='*56}")
    print(f"  PaddleOCR  lang={lang}  ({label})")
    print(f"{'='*56}")

    try:
        from paddleocr import PaddleOCR
        import numpy as np
    except ImportError:
        print("  [跳过] paddleocr 未安装")
        return 0, len(_TEST_TEXTS)

    t0 = time.time()
    try:
        ocr = PaddleOCR(lang=lang, use_angle_cls=False, use_gpu=False, show_log=False)
        img = np.array(Image.open(img_path).convert("RGB"))
        result = ocr.ocr(img, cls=False)
        result = result[0] if result else []
    except Exception as e:
        print(f"  [失败] {e}")
        return 0, len(_TEST_TEXTS)

    elapsed = time.time() - t0
    if not result:
        print(f"  耗时: {elapsed:.1f}s  |  未识别到文本")
        return 0, len(_TEST_TEXTS)

    print(f"  耗时: {elapsed:.1f}s  |  识别到 {len(result)} 段文本\n")
    for i, item in enumerate(result):
        text, conf = item[1][0], item[1][1]
        status = "✓" if any(ch in text for ch in "游戏设置退出确认购买返回等级") or text.strip() in [t.strip() for t in _TEST_TEXTS] else "?"
        print(f"  [{i:2d}] conf={conf:.2f}  {status}  \"{text}\"")

    recognized = set()
    for item in result:
        text_clean = item[1][0].strip()
        for expected in _TEST_TEXTS:
            if expected.lower() == text_clean.lower():
                recognized.add(expected)
            elif len(expected) >= 2:
                matched_chars = sum(1 for c in expected if c in text_clean)
                if matched_chars / len(expected) >= 0.6:
                    recognized.add(expected)
    hit = len(recognized)
    total = len(_TEST_TEXTS)
    print(f"\n  命中: {hit}/{total}  ({hit/total*100:.0f}%)")
    return hit, total


# ── 主入口 ───────────────────────────────────────────────────

def main():
    test_dir = Path(__file__).parent
    img_path = str(test_dir / "test_ocr_zh.png")

    print("\n" + "=" * 56)
    print("  中文 OCR 识别效果测试")
    print("=" * 56)
    print(f"  期望识别文本: {_TEST_TEXTS}\n")

    _make_test_image(img_path)

    results = {}

    # ── EasyOCR 对比 ──────────────────────────────────────
    try:
        h1, t1 = _test_easyocr(img_path, ["en"], "原版-仅英文")
        results["EasyOCR(en)"] = f"{h1}/{t1}"
    except Exception as e:
        print(f"  [EasyOCR en 失败] {e}")

    try:
        h2, t2 = _test_easyocr(img_path, ["ch_sim", "en"], "Patch后-中英文")
        results["EasyOCR(ch_sim+en)"] = f"{h2}/{t2}"
    except Exception as e:
        print(f"  [EasyOCR ch_sim+en 失败] {e}")

    # ── PaddleOCR 对比 ────────────────────────────────────
    h3, t3 = _test_paddleocr(img_path, "en", "原版-仅英文")
    results["PaddleOCR(en)"] = f"{h3}/{t3}"

    h4, t4 = _test_paddleocr(img_path, "ch", "Patch后-中文")
    results["PaddleOCR(ch)"] = f"{h4}/{t4}"

    # ── 汇总 ─────────────────────────────────────────────
    print(f"\n{'='*56}")
    print("  汇总")
    print(f"{'='*56}")
    for engine, score in results.items():
        print(f"  {engine:25s}  命中 {score}")
    print(f"{'='*56}\n")

    # 清理测试图片
    # os.unlink(img_path)


if __name__ == "__main__":
    main()
