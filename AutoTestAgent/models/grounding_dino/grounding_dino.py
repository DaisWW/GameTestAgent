"""
Grounding DINO  —  安装 + 验证  (单文件)

用法:
    python grounding_dino.py          # 自动安装(如需)并验证
    python grounding_dino.py --check  # 仅查看状态

功能: 开放词汇 UI 元素检测，替代 OmniParser 用于游戏 UI
模型: IDEA-Research/grounding-dino-tiny  (Swin-T backbone, ~172M 参数)
特点: 无需启动服务，in-process 推理，文本引导检测
"""
from __future__ import annotations
import argparse, os, socket, subprocess, sys, time
from pathlib import Path

# ─────────────────────── 配置 ────────────────────────────────
DIR       = Path(__file__).parent
CACHE_DIR = DIR / "cache"              # 模型缓存目录
MODEL_ID  = "IDEA-Research/grounding-dino-tiny"
PROXY     = "http://127.0.0.1:7890"
HF_MIRROR = "https://hf-mirror.com"

# ─────────────────────── 日志 ────────────────────────────────
def _p(tag, msg): print(f"{tag} {msg}", flush=True)
def ok(m):   _p("[✓]", m)
def info(m): _p("[→]", m)
def warn(m): _p("[!]", m)
def err(m):  _p("[✗]", m)
def sep(t):  print(f"\n{'─'*52}\n  {t}\n{'─'*52}", flush=True)

# ─────────────────────── 状态检测 ────────────────────────────
def proxy_up() -> bool:
    try:
        h, p = PROXY.rsplit(":", 1)
        with socket.socket() as s:
            s.settimeout(1)
            return s.connect_ex((h.split("//")[-1], int(p))) == 0
    except Exception:
        return False

def model_cached() -> bool:
    """检查模型权重是否已下载。"""
    config_file = CACHE_DIR / "config.json"
    model_file  = CACHE_DIR / "model.safetensors"
    if not config_file.exists():
        return False
    if not model_file.exists():
        # 也检查 pytorch_model.bin 格式
        alt = CACHE_DIR / "pytorch_model.bin"
        if not alt.exists():
            return False
    return True

# ─────────────────────── 代理环境 ────────────────────────────
def mk_env(proxy: str | None) -> dict:
    e = {**os.environ}
    if proxy:
        e["HTTPS_PROXY"] = e["HTTP_PROXY"] = e["ALL_PROXY"] = proxy
    return e

def best_proxy() -> str | None:
    return PROXY if proxy_up() else None

# ─────────────────────── 安装步骤 ────────────────────────────
def step_deps():
    sep("第 1/2 步  检查 Python 依赖")
    pip = [sys.executable, "-m", "pip", "install"]
    px = best_proxy()
    env = mk_env(px)

    # 检查 torch
    has_torch = subprocess.run(
        [sys.executable, "-c", "import torch"],
        capture_output=True,
    ).returncode == 0
    if not has_torch:
        info("安装 PyTorch CUDA 12.1 (~2-3 GB)...")
        info("无 GPU 请先手动运行: pip install torch --index-url https://download.pytorch.org/whl/cpu")
        if subprocess.run(
            pip + ["torch", "torchvision",
                   "--index-url", "https://download.pytorch.org/whl/cu121"],
            env=env,
        ).returncode != 0:
            raise RuntimeError("PyTorch 安装失败")

    # 检查 transformers + pillow
    missing = []
    for pkg, imp in [("transformers>=4.40", "transformers"), ("Pillow", "PIL")]:
        rc = subprocess.run([sys.executable, "-c", f"import {imp}"], capture_output=True).returncode
        if rc != 0:
            missing.append(pkg)
    if missing:
        info(f"安装: {', '.join(missing)}")
        if subprocess.run(pip + missing + ["-q"], env=env).returncode != 0:
            raise RuntimeError(f"依赖安装失败: {missing}")

    ok("依赖就绪")

def step_download_model():
    sep("第 2/2 步  下载 Grounding DINO Tiny 模型权重")
    if model_cached():
        ok("模型已缓存，跳过")
        return

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # 尝试各种下载源
    endpoints = [
        ("https://huggingface.co", "HuggingFace", None),
        ("https://huggingface.co", "HF+代理",     PROXY if proxy_up() else None),
        (HF_MIRROR,                "hf-mirror",   None),
        (HF_MIRROR,                "hf-mirror+代理", PROXY if proxy_up() else None),
    ]
    for endpoint, label, eproxy in endpoints:
        if eproxy and not proxy_up():
            continue
        info(f"  [{label}]  {endpoint}")
        env = {**mk_env(eproxy), "HF_ENDPOINT": endpoint}
        rc = subprocess.run(
            [sys.executable, "-c",
             f"from huggingface_hub import snapshot_download;"
             f"snapshot_download('{MODEL_ID}', local_dir=r'{CACHE_DIR}')"],
            env=env,
        ).returncode
        if rc == 0 and model_cached():
            ok(f"模型下载完成 [{label}]")
            return
        warn(f"  [{label}] 失败")

    # 最后尝试 transformers 自带下载
    info("尝试 transformers 自动下载...")
    rc = subprocess.run(
        [sys.executable, "-c",
         f"from transformers import AutoModelForZeroShotObjectDetection;"
         f"AutoModelForZeroShotObjectDetection.from_pretrained('{MODEL_ID}', cache_dir=r'{CACHE_DIR}')"],
        env=mk_env(best_proxy()),
    ).returncode
    if rc == 0:
        ok("模型下载完成 (transformers 缓存)")
        return

    warn("所有渠道失败，请手动下载:\n"
         f"  huggingface-cli download {MODEL_ID} --local-dir {CACHE_DIR}")

def install():
    step_deps()
    step_download_model()

# ─────────────────────── 验证 ────────────────────────────────
def verify():
    """加载模型并做一次推理验证。"""
    sep("验证模型推理")
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    model_path = str(CACHE_DIR) if model_cached() else MODEL_ID
    device = "cuda" if torch.cuda.is_available() else "cpu"
    info(f"加载模型: {model_path}  设备: {device}")

    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_path).to(device)
    ok("模型加载成功")

    # 用 100x100 测试图做验证
    dummy = Image.new("RGB", (100, 100), color=(128, 128, 128))
    text = "button ."
    inputs = processor(images=dummy, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    ok(f"推理验证通过  (设备: {device})")
    return True

# ─────────────────────── 主入口 ──────────────────────────────
def status():
    print(f"\n{'='*52}")
    print(f"  Grounding DINO Tiny  状态")
    print(f"{'='*52}")
    print(f"  模型缓存  : {'是' if model_cached() else '否'}  {CACHE_DIR}")
    print(f"  代理可用  : {'是' if proxy_up()     else '否'}  {PROXY}")
    print(f"{'='*52}")

def main() -> int:
    if sys.platform == "win32":
        try:
            import ctypes; ctypes.windll.kernel32.SetConsoleOutputCP(65001)
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception: pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="仅查看状态")
    args = ap.parse_args()

    print(f"\n{'='*52}\n  Grounding DINO Tiny  —  安装 & 验证器\n{'='*52}")

    if args.check:
        status(); return 0

    if not model_cached():
        try: install()
        except Exception as e: err(f"安装失败: {e}"); return 1

    try:
        verify()
    except Exception as e:
        err(f"验证失败: {e}")
        return 1

    ok("Grounding DINO 就绪，无需启动服务，Agent 会在进程内直接加载模型")
    return 0

if __name__ == "__main__":
    sys.exit(main())
