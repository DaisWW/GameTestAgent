"""
Grounding DINO Provider 集成测试。

测试:
  1. 模型加载与推理（dummy 图片）
  2. VisionProvider 接口正确性（输出格式校验）
  3. 真实游戏 UI 模拟图检测
  4. 自定义 text_prompt 切换
  5. teardown 释放显存

用法:
  python models/grounding_dino/test_grounding_dino.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# ── 确保项目根目录在 sys.path 中 ──────────────────────────────
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PIL import Image, ImageDraw, ImageFont

# ─────────────────────── 日志 ────────────────────────────────
_PASS = 0
_FAIL = 0

def ok(name: str, msg: str = ""):
    global _PASS
    _PASS += 1
    print(f"  [✓] {name}  {msg}", flush=True)

def fail(name: str, msg: str = ""):
    global _FAIL
    _FAIL += 1
    print(f"  [✗] {name}  {msg}", flush=True)

def sep(t: str):
    print(f"\n{'─'*52}\n  {t}\n{'─'*52}", flush=True)


# ─────────────────────── 辅助：生成模拟游戏 UI 截图 ──────────
def make_game_ui_image() -> Image.Image:
    """生成 720x1280 模拟游戏 UI，包含按钮、图标区域。"""
    w, h = 720, 1280
    img = Image.new("RGB", (w, h), color=(20, 20, 40))
    draw = ImageDraw.Draw(img)

    # 尝试加载字体，失败则用默认字体
    try:
        font = ImageFont.truetype("arial.ttf", 32)
        font_small = ImageFont.truetype("arial.ttf", 20)
    except (IOError, OSError):
        font = ImageFont.load_default()
        font_small = font

    # 顶部标题栏
    draw.rectangle([0, 0, w, 80], fill=(40, 40, 80))
    draw.text((20, 25), "Game Title", fill="white", font=font)

    # 几个按钮
    buttons = [
        (160, 300, 560, 380, "Start Game"),
        (160, 420, 560, 500, "Settings"),
        (160, 540, 560, 620, "Exit"),
        (160, 700, 560, 780, "Shop"),
    ]
    for x1, y1, x2, y2, label in buttons:
        draw.rounded_rectangle([x1, y1, x2, y2], radius=15, fill=(60, 120, 200))
        tw = draw.textlength(label, font=font) if hasattr(draw, "textlength") else len(label) * 16
        tx = x1 + (x2 - x1 - tw) / 2
        ty = y1 + 12
        draw.text((tx, ty), label, fill="white", font=font)

    # 右上角关闭按钮（小圆）
    draw.ellipse([w - 60, 15, w - 15, 60], fill=(200, 50, 50))
    draw.text((w - 48, 22), "X", fill="white", font=font_small)

    # 底部导航图标区域
    icon_y = h - 120
    for i, label in enumerate(["Home", "Quest", "Bag", "Team"]):
        cx = 90 + i * 180
        draw.rounded_rectangle([cx - 40, icon_y, cx + 40, icon_y + 60], radius=8, fill=(80, 80, 120))
        draw.text((cx - 20, icon_y + 18), label, fill="white", font=font_small)

    return img


# ─────────────────────── 测试用例 ────────────────────────────

def test_1_model_load_and_inference():
    """测试模型能否加载并对 dummy 图片推理。"""
    sep("Test 1: 模型加载与 dummy 推理")
    from core.vision.providers.grounding_dino import Provider

    provider = Provider()
    dummy = Image.new("RGB", (200, 200), color=(128, 128, 128))

    t0 = time.time()
    elements = provider.detect(dummy)
    elapsed = time.time() - t0

    if isinstance(elements, list):
        ok("模型加载+推理成功", f"耗时 {elapsed:.2f}s, 检出 {len(elements)} 个元素")
    else:
        fail("返回类型错误", f"期望 list, 得到 {type(elements)}")

    return provider  # 复用，避免重复加载


def test_2_output_format(provider):
    """验证输出格式符合 VisionProvider 规范。"""
    sep("Test 2: 输出格式校验")
    img = make_game_ui_image()
    elements = provider.detect(img)

    if not elements:
        fail("未检出任何元素", "游戏 UI 模拟图应该能检出元素")
        return

    ok("检出元素", f"共 {len(elements)} 个")

    # 检查每个元素的必要字段
    required_keys = {"id", "bbox", "label", "type"}
    for i, elem in enumerate(elements):
        missing = required_keys - set(elem.keys())
        if missing:
            fail(f"元素[{i}] 缺少字段", f"{missing}")
            return

    ok("字段完整", f"所有元素都包含 {required_keys}")

    # 检查 bbox 格式
    for i, elem in enumerate(elements):
        bbox = elem["bbox"]
        if not (isinstance(bbox, list) and len(bbox) == 4):
            fail(f"元素[{i}] bbox 格式错误", f"{bbox}")
            return
        if not all(0 <= v <= 1000 for v in bbox):
            fail(f"元素[{i}] bbox 超范围", f"{bbox} (应为 0-1000)")
            return

    ok("bbox 格式正确", "所有坐标在 0-1000 范围内")

    # 检查 id 连续性
    ids = [e["id"] for e in elements]
    if ids == list(range(len(elements))):
        ok("id 连续", f"0..{len(elements)-1}")
    else:
        fail("id 不连续", f"{ids}")

    # 打印检出结果
    print(f"\n  {'ID':>3}  {'Label':<12} {'Type':<10} {'Conf':>5}  {'BBox'}")
    print(f"  {'─'*3}  {'─'*12} {'─'*10} {'─'*5}  {'─'*20}")
    for e in elements:
        conf = e.get("confidence", "?")
        print(f"  {e['id']:>3}  {e['label']:<12} {e['type']:<10} {conf:>5}  {e['bbox']}")


def test_3_game_ui_detection(provider):
    """测试对模拟游戏 UI 的检测效果。"""
    sep("Test 3: 游戏 UI 检测效果")
    img = make_game_ui_image()

    t0 = time.time()
    elements = provider.detect(img)
    elapsed = time.time() - t0

    ok("推理速度", f"{elapsed*1000:.0f}ms")

    # 应该至少检出一些元素（按钮/图标）
    if len(elements) >= 2:
        ok("检出数量合理", f"{len(elements)} 个元素 (≥2)")
    else:
        fail("检出过少", f"仅 {len(elements)} 个元素，期望 ≥2")

    # 检查是否有 button 类型
    types = set(e["type"] for e in elements)
    if "button" in types:
        ok("检出 button 类型")
    else:
        fail("未检出 button 类型", f"类型: {types}")


def test_4_custom_prompt(provider):
    """测试自定义 text_prompt 切换。"""
    sep("Test 4: 自定义 text_prompt")
    from core.vision.providers.grounding_dino import Provider

    # 创建一个只找 button 的 provider（复用已加载的模型不行，新建一个改 prompt）
    custom = Provider(text_prompt="button .", box_threshold=0.2)
    img = make_game_ui_image()
    elements = custom.detect(img)

    ok("自定义 prompt 推理成功", f"prompt='button .' → {len(elements)} 个元素")

    # 对比全量 prompt
    all_elements = provider.detect(img)
    print(f"  默认 prompt: {len(all_elements)} 个元素")
    print(f"  仅 button:   {len(elements)} 个元素")

    custom.teardown()


def test_5_teardown(provider):
    """测试显存释放。"""
    sep("Test 5: teardown 释放显存")
    provider.teardown()

    if provider._model is None and provider._processor is None:
        ok("模型和处理器已释放")
    else:
        fail("释放不完全", f"model={provider._model is not None}, processor={provider._processor is not None}")


def test_6_from_config():
    """测试 from_config 工厂方法。"""
    sep("Test 6: from_config 工厂方法")
    from config.settings import load_config

    config = load_config()
    # 临时切换 vision_type
    config.vision.vision_type = "grounding_dino"

    from core.vision.providers.grounding_dino import Provider
    provider = Provider.from_config(config)

    if provider._model_id == config.vision.gdino_model:
        ok("from_config model_id 正确", f"{provider._model_id}")
    else:
        fail("from_config model_id 错误", f"{provider._model_id} != {config.vision.gdino_model}")

    if provider._text_prompt == config.vision.gdino_text_prompt:
        ok("from_config text_prompt 正确")
    else:
        fail("from_config text_prompt 错误")

    if provider._box_threshold == config.vision.gdino_box_threshold:
        ok("from_config box_threshold 正确")
    else:
        fail("from_config box_threshold 错误")


# ─────────────────────── 主入口 ──────────────────────────────
def main() -> int:
    if sys.platform == "win32":
        try:
            import ctypes; ctypes.windll.kernel32.SetConsoleOutputCP(65001)
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    print(f"\n{'='*52}")
    print(f"  Grounding DINO Provider  集成测试")
    print(f"{'='*52}")

    try:
        provider = test_1_model_load_and_inference()
        test_2_output_format(provider)
        test_3_game_ui_detection(provider)
        test_4_custom_prompt(provider)
        test_6_from_config()
        test_5_teardown(provider)
    except Exception as e:
        import traceback
        traceback.print_exc()
        fail("异常", str(e))

    # 汇总
    print(f"\n{'='*52}")
    total = _PASS + _FAIL
    print(f"  结果: {_PASS}/{total} 通过, {_FAIL}/{total} 失败")
    print(f"{'='*52}\n")

    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
