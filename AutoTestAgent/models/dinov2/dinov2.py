"""
DINOv2-Small  —  安装 + 验证  (单文件)

用法:
    python dinov2.py          # 自动安装(如需)并验证
    python dinov2.py --check  # 仅查看状态

功能: 页面截图嵌入，用于页面身份识别
模型: facebook/dinov2-small  (ViT-S/14, 22M 参数, 384 维嵌入)
特点: 无需启动服务，由 Grounding DINO Provider 在进程内同时加载
"""
from __future__ import annotations
import argparse, os, socket, subprocess, sys, time
from pathlib import Path

# ─────────────────────── 配置 ────────────────────────────────
DIR       = Path(__file__).parent
CACHE_DIR = DIR / "cache"              # 模型缓存目录
PROXY     = "http://127.0.0.1:7890"
HF_MIRROR = "https://hf-mirror.com"
MODEL_ID  = "facebook/dinov2-small"
EMBED_DIM = 384

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

def _clean_incomplete(cache: Path):
    """清理 huggingface_hub 残留的 .incomplete 和 .lock 文件。"""
    hf_cache = cache / ".cache"
    if not hf_cache.exists():
        return
    for f in hf_cache.rglob("*.incomplete"):
        try: f.unlink(); info(f"  清理残留: {f.name}")
        except OSError: pass
    for f in hf_cache.rglob("*.lock"):
        try: f.unlink()
        except OSError: pass

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
    for pkg, imp in [("transformers", "transformers"), ("Pillow", "PIL")]:
        rc = subprocess.run([sys.executable, "-c", f"import {imp}"], capture_output=True).returncode
        if rc != 0:
            missing.append(pkg)
    if missing:
        info(f"安装: {', '.join(missing)}")
        if subprocess.run(pip + missing + ["-q"], env=env).returncode != 0:
            raise RuntimeError(f"依赖安装失败: {missing}")

    ok("依赖就绪")

def step_download_model():
    sep("第 2/2 步  下载 DINOv2-Small 模型权重")
    if model_cached():
        ok("模型已缓存，跳过")
        return

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _clean_incomplete(CACHE_DIR)

    # 尝试各种下载源
    endpoints = [
        ("https://huggingface.co", "HF+代理",         PROXY if proxy_up() else None),
        (HF_MIRROR,                "hf-mirror",       None),
        (HF_MIRROR,                "hf-mirror+代理",  PROXY if proxy_up() else None),
        ("https://huggingface.co", "HuggingFace",     None),
    ]
    for endpoint, label, eproxy in endpoints:
        if eproxy and not proxy_up():
            continue
        info(f"  [{label}]  {endpoint}")
        env = {**mk_env(eproxy), "HF_ENDPOINT": endpoint}
        try:
            rc = subprocess.run(
                [sys.executable, "-c",
                 f"from huggingface_hub import snapshot_download;"
                 f"snapshot_download('{MODEL_ID}', local_dir=r'{CACHE_DIR}')"],
                env=env,
                timeout=600,
            ).returncode
        except subprocess.TimeoutExpired:
            warn(f"  [{label}] 超时（600s）")
            _clean_incomplete(CACHE_DIR)
            continue
        if rc == 0 and model_cached():
            ok(f"模型下载完成 [{label}]")
            return
        warn(f"  [{label}] 失败")
        _clean_incomplete(CACHE_DIR)

    warn("所有渠道失败，请手动下载:\n"
         f"  huggingface-cli download {MODEL_ID} --local-dir {CACHE_DIR}")

def install():
    step_deps()
    step_download_model()

# ─────────────────────── 验证 ────────────────────────────────
def verify():
    """加载模型并做一次嵌入推理验证。"""
    sep("验证模型推理")
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    model_path = str(CACHE_DIR) if model_cached() else MODEL_ID
    device = "cuda" if torch.cuda.is_available() else "cpu"
    info(f"加载模型: {model_path}  设备: {device}")

    processor = AutoImageProcessor.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device).eval()
    ok("模型加载成功")

    # 用 224x224 测试图做验证
    dummy = Image.new("RGB", (224, 224), color=(128, 128, 128))
    inputs = processor(images=dummy, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    cls = outputs.last_hidden_state[:, 0, :]
    cls = cls / cls.norm(dim=-1, keepdim=True)

    ok(f"推理验证通过  (设备: {device}, 嵌入维度: {cls.shape[-1]})")
    return True

# ─────────────────────── 主入口 ──────────────────────────────
def status():
    print(f"\n{'='*52}")
    print(f"  DINOv2-Small  状态")
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

    print(f"\n{'='*52}\n  DINOv2-Small  —  安装 & 验证器\n{'='*52}")

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

    ok("DINOv2 就绪，无需启动服务，Agent 会在进程内直接加载模型")
    return 0

if __name__ == "__main__":
    sys.exit(main())
