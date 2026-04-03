"""
DINOv2-Small  —  安装 + 启动 + 验证  (单文件)

用法:
    python dinov2.py          # 自动安装(如需)并启动
    python dinov2.py --check  # 仅查看状态

功能: 提供截图嵌入服务，用于页面身份识别
模型: facebook/dinov2-small  (ViT-S/14, 22M 参数, 384 维嵌入)
API:
    POST /embed       — 上传图片，返回 384 维归一化向量
    POST /similarity  — 上传两张图片，返回余弦相似度
    GET  /health      — 健康检查
"""
from __future__ import annotations
import argparse, io, json, os, socket, subprocess, sys, time
from pathlib import Path

# ─────────────────────── 配置 ────────────────────────────────
DIR       = Path(__file__).parent
CACHE_DIR = DIR / "cache"              # 模型缓存目录
PORT      = 7862
URL       = f"http://127.0.0.1:{PORT}"
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
def port_open(port=PORT) -> bool:
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0

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

    # 检查 transformers + flask + pillow
    missing = []
    for pkg, imp in [("transformers", "transformers"), ("flask", "flask"), ("Pillow", "PIL")]:
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
         f"from transformers import AutoModel;"
         f"AutoModel.from_pretrained('{MODEL_ID}', cache_dir=r'{CACHE_DIR}')"],
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

# ─────────────────────── HTTP 服务 ───────────────────────────
def create_app():
    """创建 Flask 应用，加载模型。"""
    import torch
    from flask import Flask, request, jsonify
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    app = Flask(__name__)

    # 加载模型
    info("加载 DINOv2-Small 模型...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    info(f"推理设备: {device}")

    if model_cached() and (CACHE_DIR / "config.json").exists():
        model_path = str(CACHE_DIR)
    else:
        model_path = MODEL_ID

    processor = AutoImageProcessor.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device).eval()
    ok(f"模型加载完成 ({EMBED_DIM} 维嵌入)")

    def _embed_image(pil_image: Image.Image) -> list:
        """提取单张图片的归一化嵌入向量。"""
        inputs = processor(images=pil_image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        # CLS token = last_hidden_state[:, 0, :]
        cls = outputs.last_hidden_state[:, 0, :]
        cls = cls / cls.norm(dim=-1, keepdim=True)
        return cls[0].cpu().tolist()

    def _read_image(file) -> Image.Image:
        """从 Flask request file 读取 PIL Image。"""
        return Image.open(io.BytesIO(file.read())).convert("RGB")

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "model": MODEL_ID, "device": device, "dim": EMBED_DIM})

    @app.route("/embed", methods=["POST"])
    def embed():
        if "image" not in request.files:
            return jsonify({"error": "缺少 image 字段"}), 400
        img = _read_image(request.files["image"])
        vec = _embed_image(img)
        return jsonify({"embedding": vec, "dim": len(vec)})

    @app.route("/similarity", methods=["POST"])
    def similarity():
        if "image1" not in request.files or "image2" not in request.files:
            return jsonify({"error": "缺少 image1 或 image2 字段"}), 400
        img1 = _read_image(request.files["image1"])
        img2 = _read_image(request.files["image2"])
        vec1 = _embed_image(img1)
        vec2 = _embed_image(img2)
        # 余弦相似度（向量已归一化，直接点积）
        sim = sum(a * b for a, b in zip(vec1, vec2))
        return jsonify({"similarity": sim, "is_same_page": sim > 0.92})

    return app

# ─────────────────────── 启动 & 验证 ─────────────────────────
def launch():
    sep("启动 DINOv2 嵌入服务")
    app = create_app()
    ok(f"服务地址: {URL}")
    print(f"\n{'='*52}\n  {URL}  (Ctrl+C 停止)\n{'='*52}\n", flush=True)
    app.run(host="127.0.0.1", port=PORT, threaded=False)

def wait_ready(timeout=60) -> bool:
    sep("验证服务就绪")
    deadline = time.time() + timeout
    while time.time() < deadline:
        if port_open():
            print(flush=True)
            ok(f"服务就绪！  {URL}")
            return True
        used = int(time.time() - (deadline - timeout))
        print(f"\r  等待中... {used}s/{timeout}s", end="", flush=True)
        time.sleep(1)
    print(flush=True)
    return False

# ─────────────────────── 主入口 ──────────────────────────────
def status():
    print(f"\n{'='*52}")
    print(f"  DINOv2-Small  状态")
    print(f"{'='*52}")
    print(f"  模型缓存  : {'是' if model_cached() else '否'}  {CACHE_DIR}")
    print(f"  服务运行  : {'是' if port_open()    else '否'}  {URL}")
    print(f"  代理可用  : {'是' if proxy_up()     else '否'}  {PROXY}")
    print(f"{'='*52}")

def main() -> int:
    if sys.platform == "win32":
        try:
            import ctypes; ctypes.windll.kernel32.SetConsoleOutputCP(65001)
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception: pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--check",   action="store_true", help="仅查看状态")
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--port",    type=int, default=PORT)
    args = ap.parse_args()

    if args.port != PORT:
        # 需要更新模块级变量供其他函数使用
        import dinov2 as _self
        _self.PORT = args.port
        _self.URL  = f"http://127.0.0.1:{args.port}"

    print(f"\n{'='*52}\n  DINOv2-Small  —  安装 & 启动器\n{'='*52}")

    if args.check:
        status(); return 0

    if port_open():
        ok(f"服务已在运行  {URL}"); return 0

    if not model_cached():
        try: install()
        except Exception as e: err(f"安装失败: {e}"); return 1

    try:
        launch()
    except KeyboardInterrupt:
        info("停止中...")
    return 0

if __name__ == "__main__":
    sys.exit(main())
