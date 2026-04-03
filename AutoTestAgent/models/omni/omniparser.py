"""
OmniParser V2  —  安装 + 启动 + 验证  (单文件)

用法:
    python omniparser.py          # 自动安装(如需)并启动
    python omniparser.py --check  # 仅查看状态

安装流程: clone → pip → 下载权重 → 预热easyocr → 启动gradio → 验证端口
代理策略: 直连 | GitHub+代理 | hf-mirror | hf-mirror+代理  (按顺序尝试)
"""
from __future__ import annotations
import argparse, os, shutil, socket, stat, subprocess, sys, threading, time
from pathlib import Path

# ─────────────────────── 配置 ────────────────────────────────
DIR      = Path(__file__).parent
OMNI     = DIR / "OmniParser"          # clone 目标
WEIGHTS  = OMNI / "weights"
PORT     = 7861
URL      = f"http://127.0.0.1:{PORT}"
PROXY    = "http://127.0.0.1:7890"    # 本地代理
HF       = "https://hf-mirror.com"    # HuggingFace 国内镜像
MIN_MB   = 50                          # 权重最小体积(MB)，防止部分下载

CLONE_SOURCES = [
    ("https://github.com/microsoft/OmniParser.git", "GitHub",       None),
    ("https://github.com/microsoft/OmniParser.git", "GitHub+代理",  PROXY),
    (f"{HF}/spaces/microsoft/OmniParser",           "hf-mirror",    None),
    (f"{HF}/spaces/microsoft/OmniParser",           "hf-mirror+代理", PROXY),
]
HF_ENDPOINTS = [
    ("https://huggingface.co", "HuggingFace", None),
    ("https://huggingface.co", "HF+代理",     PROXY),
    (HF,                       "hf-mirror",   None),
    (HF,                       "hf-mirror+代理", PROXY),
]

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

def cloned()  -> bool: return (OMNI / "gradio_demo.py").exists()
def dir_mb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6

def weights_ok() -> bool:
    """
    icon_detect/model.pt       — YOLO 小模型 (~38MB)
    icon_caption_florence/     — Florence-2 大模型 (>500MB)
    """
    model_pt = WEIGHTS / "icon_detect" / "model.pt"
    if not model_pt.exists() or model_pt.stat().st_size < 1024 * 1024:
        return False
    cap = WEIGHTS / "icon_caption_florence"
    if not cap.exists() or dir_mb(cap) < MIN_MB:
        return False
    return True

# ─────────────────────── 代理环境 ────────────────────────────
def mk_env(proxy: str | None) -> dict:
    e = {**os.environ}
    if proxy:
        e["HTTPS_PROXY"] = e["HTTP_PROXY"] = e["ALL_PROXY"] = proxy
    return e

def best_proxy() -> str | None:
    """返回可用代理，不可用则返回 None。"""
    return PROXY if proxy_up() else None

# ─────────────────────── 删目录（Windows 文件锁）────────────
def rmdir(p: Path):
    if not p.exists(): return
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/F", "/IM", "git.exe"], capture_output=True)
        time.sleep(0.3)
        subprocess.run(["cmd", "/c", "rd", "/s", "/q", str(p)], capture_output=True)
    else:
        def fix(fn, fp, _): os.chmod(fp, stat.S_IWRITE); fn(fp)
        shutil.rmtree(p, onerror=fix)
    if p.exists():
        raise RuntimeError(f"无法删除: {p}  (请手动删除后重试)")

# ─────────────────────── 安装步骤 ────────────────────────────
def _run(cmd, env=None, cwd=None) -> int:
    return subprocess.run(cmd, env=env, cwd=cwd).returncode

def step_clone():
    sep("第 1/4 步  克隆 OmniParser")
    if cloned(): ok("已克隆，跳过"); return
    if OMNI.exists(): warn("检测到残留目录，清理中..."); rmdir(OMNI)
    for url, label, proxy in CLONE_SOURCES:
        if proxy and not proxy_up(): info(f"  [{label}] 代理未开启，跳过"); continue
        info(f"  尝试 [{label}]  {url[:55]}")
        rc = _run(["git", "clone", "--progress", url, str(OMNI)], env=mk_env(proxy))
        if rc == 0 and cloned(): ok(f"克隆成功 [{label}]"); return
        warn(f"  [{label}] 失败，清理中..."); rmdir(OMNI)
    raise RuntimeError("所有克隆源均失败，请手动 clone 到 models/omni/OmniParser/")

def step_deps():
    sep("第 2/4 步  安装 Python 依赖")
    px = best_proxy()
    pip = [sys.executable, "-m", "pip", "install"]
    env = mk_env(px)
    has_torch = subprocess.run([sys.executable, "-c", "import torch"],
                               capture_output=True).returncode == 0
    if not has_torch:
        info("安装 PyTorch CUDA 12.1 (~2-3 GB)...")
        info("无 GPU 请先手动运行: pip install torch --index-url https://download.pytorch.org/whl/cpu")
        if _run(pip + ["torch", "torchvision",
                       "--index-url", "https://download.pytorch.org/whl/cu121"], env=env) != 0:
            raise RuntimeError("PyTorch 安装失败")
    info("安装 OmniParser requirements...")
    if _run(pip + ["-r", str(OMNI / "requirements.txt")], env=env) != 0:
        raise RuntimeError("requirements.txt 安装失败")
    _run(pip + ["huggingface_hub>=0.23", "-q"], env=env)
    ok("依赖安装完成")

def step_weights():
    """
    microsoft/OmniParser-v2.0 包含两个子目录:
      icon_detect/model.pt   —— YOLO 检测模型
      icon_caption/          —— Florence-2 描述模型
    直接下载整个 repo 到 weights/ 即可，无需分开下载。
    """
    sep("第 3/4 步  下载模型权重")
    if weights_ok(): ok("权重已就绪，跳过"); return

    WEIGHTS.mkdir(parents=True, exist_ok=True)
    info("下载 microsoft/OmniParser-v2.0 (含 icon_detect + icon_caption)...")

    for endpoint, label, eproxy in HF_ENDPOINTS:
        if eproxy and not proxy_up(): continue
        info(f"  [{label}]  {endpoint}")
        env = {**mk_env(eproxy), "HF_ENDPOINT": endpoint}
        rc = subprocess.run(
            [sys.executable, "-c",
             f"from huggingface_hub import snapshot_download;"
             f"snapshot_download('microsoft/OmniParser-v2.0', local_dir=r'{WEIGHTS}')"],
            env=env,
        ).returncode
        if rc == 0:
            # OmniParser-v2.0 下载到 weights/，会创建 icon_caption/ 和 icon_detect/
            # 但 gradio_demo.py 期望的是 icon_caption_florence/，需要改名
            src = WEIGHTS / "icon_caption"
            dst = WEIGHTS / "icon_caption_florence"
            if src.exists() and not dst.exists():
                src.rename(dst)
                info("  已将 icon_caption 改名为 icon_caption_florence")
        if rc == 0 and weights_ok():
            ok(f"权重下载完成 [{label}]")
            return
        if rc == 0:
            info(f"  下载完毕但体积检查未通过, icon_detect={dir_mb(WEIGHTS/'icon_detect'):.0f}MB, icon_caption_florence={dir_mb(WEIGHTS/'icon_caption_florence'):.0f}MB")
        warn(f"  [{label}] 失败")

    warn("所有渠道失败，请手动下载:\n"
         f"  huggingface-cli download microsoft/OmniParser-v2.0 --local-dir {WEIGHTS}")

def step_warm_easyocr():
    """预热 easyocr：带代理提前下载模型缓存，避免 gradio 启动时卡死。"""
    sep("第 4/4 步  预热 easyocr 模型")
    px = best_proxy()
    info(f"初始化 easyocr.Reader (代理={px or '直连'})...")
    rc = subprocess.run(
        [sys.executable, "-c",
         "import easyocr; easyocr.Reader(['en'], verbose=True)"],
        env=mk_env(px),
        cwd=str(OMNI),
    ).returncode
    if rc == 0:
        ok("easyocr 预热完成，模型已缓存")
    else:
        warn("easyocr 预热失败，启动时可能再次尝试下载（需网络）")

def install():
    step_clone()
    step_deps()
    step_weights()
    step_warm_easyocr()

# ─────────────────────── 启动 & 验证 ─────────────────────────
def launch() -> subprocess.Popen:
    sep("启动 OmniParser V2")
    px = best_proxy()
    info(f"代理: {px or '直连'}")
    info("模型加载约 1-3 分钟，请等待...")
    env = mk_env(px)
    # 禁止 transformers 在运行时下载新版本配置文件（防止 flash_attn 等不兼容依赖）
    env["HF_HUB_OFFLINE"]       = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    # 代理时排除 localhost，防止 gradio 调用自身端口时被代理拦截
    env["NO_PROXY"]  = "127.0.0.1,localhost,::1"
    env["no_proxy"]  = "127.0.0.1,localhost,::1"
    # 优先使用中文 OCR wrapper（不改 OmniParser 源码），回退到原版
    _gradio_script = "gradio_demo_zh.py" if (OMNI / "gradio_demo_zh.py").exists() else "gradio_demo.py"
    info(f"启动脚本: {_gradio_script}")
    proc = subprocess.Popen(
        [sys.executable, _gradio_script],
        cwd=str(OMNI),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    info(f"PID: {proc.pid}")
    def _fwd():
        for line in proc.stdout:
            print(f"  [gradio] {line}", end="", flush=True)
    threading.Thread(target=_fwd, daemon=True).start()
    return proc

def wait_ready(proc: subprocess.Popen, timeout=300) -> bool:
    sep("验证服务就绪")
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            err(f"gradio 意外退出 (code={proc.returncode})")
            return False
        if port_open():
            print(flush=True)
            ok(f"服务就绪！  {URL}")
            return True
        used = int(time.time() - (deadline - timeout))
        print(f"\r  等待中... {used}s/{timeout}s", end="", flush=True)
        time.sleep(2)
    print(flush=True)
    return False

# ─────────────────────── 主入口 ──────────────────────────────
def status():
    print(f"\n{'='*52}")
    print(f"  OmniParser V2  状态")
    print(f"{'='*52}")
    print(f"  已克隆    : {'是' if cloned()     else '否'}  {OMNI}")
    print(f"  权重就绪  : {'是' if weights_ok() else '否'}  {WEIGHTS}")
    print(f"  服务运行  : {'是' if port_open()  else '否'}  {URL}")
    print(f"  代理可用  : {'是' if proxy_up()   else '否'}  {PROXY}")
    print(f"{'='*52}")

def main() -> int:
    if sys.platform == "win32":
        try:
            import ctypes; ctypes.windll.kernel32.SetConsoleOutputCP(65001)
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception: pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--check",   action="store_true")
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args()

    print(f"\n{'='*52}\n  OmniParser V2  —  安装 & 启动器\n{'='*52}")

    if args.check:
        status(); return 0

    if port_open():
        ok(f"服务已在运行  {URL}"); return 0

    if not cloned() or not weights_ok():
        try: install()
        except Exception as e: err(f"安装失败: {e}"); return 1

    proc = launch()
    if not wait_ready(proc, args.timeout):
        err(f"服务 {args.timeout}s 内未就绪")
        proc.terminate(); return 1

    print(f"\n{'='*52}\n  {URL}  (Ctrl+C 停止)\n{'='*52}\n")
    try: proc.wait()
    except KeyboardInterrupt: info("停止中..."); proc.terminate()
    return 0

if __name__ == "__main__":
    sys.exit(main())
