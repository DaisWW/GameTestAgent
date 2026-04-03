"""
OmniParser V2 — 安装 + 启动 + 验证一体化脚本

用法：
    python omniparser.py            # 自动安装（如有需要）并启动服务
    python omniparser.py --check    # 仅查看当前状态
    python omniparser.py --timeout 180  # 自定义等待超时（秒）
"""
from __future__ import annotations

import argparse
import os
import shutil
import socket
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

# ── 路径常量 ────────────────────────────────────────────────
SCRIPTS_DIR = Path(__file__).parent
OMNI_DIR    = SCRIPTS_DIR / "OmniParser"
WEIGHTS_DIR = OMNI_DIR / "weights"
PORT        = 7861
SERVICE_URL = f"http://127.0.0.1:{PORT}"

HF_MIRROR         = "https://hf-mirror.com"
ICON_CAPTION_REPO = "microsoft/OmniParser-v2.0"
ICON_DETECT_REPO  = "microsoft/Florence-2-base-ft"
GITHUB_URL        = "https://github.com/microsoft/OmniParser.git"
LOCAL_PROXY       = "http://127.0.0.1:7890"

# clone 尝试顺序：(url, 描述, proxy)
CLONE_SOURCES = [
    (GITHUB_URL,                                          "GitHub",          None),
    (GITHUB_URL,                                          "GitHub+代理",     LOCAL_PROXY),
    (f"{HF_MIRROR}/spaces/microsoft/OmniParser",          "hf-mirror.com",   None),
    (f"{HF_MIRROR}/spaces/microsoft/OmniParser",          "hf-mirror+代理", LOCAL_PROXY),
]

# 权重下载 HF_ENDPOINT 尝试顺序：(端点, 描述, 是否透过代理)
HF_ENDPOINTS = [
    ("https://huggingface.co",  "HuggingFace官网",  None),
    ("https://huggingface.co",  "HF+代理",         LOCAL_PROXY),
    (HF_MIRROR,                 "hf-mirror.com",  None),
    (HF_MIRROR,                 "hf-mirror+代理", LOCAL_PROXY),
]


# ══════════════════════════════════════════════════════════════
# 日志工具（支持中文 + 颜色）
# ══════════════════════════════════════════════════════════════

# Windows Terminal / VSCode 终端支持 ANSI；cmd 默认不支持
_USE_COLOR = sys.stdout.isatty() and (
    os.environ.get("WT_SESSION")          # Windows Terminal
    or os.environ.get("TERM_PROGRAM")     # iTerm2 / VSCode
    or sys.platform != "win32"
)

def _color(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text

def ok(msg: str):
    print(_color("32", f"[✓] {msg}"), flush=True)

def info(msg: str):
    print(f"[→] {msg}", flush=True)

def warn(msg: str):
    print(_color("33", f"[!] {msg}"), flush=True)

def err(msg: str):
    print(_color("31", f"[✗] {msg}"), file=sys.stderr, flush=True)

def section(title: str):
    line = "─" * 50
    print(f"\n{_color('36', line)}", flush=True)
    print(_color("1", f"  {title}"), flush=True)
    print(f"{_color('36', line)}", flush=True)


# ══════════════════════════════════════════════════════════════
# 旋转进度指示器（子线程）
# ══════════════════════════════════════════════════════════════

class Spinner:
    """在等待耗时命令时显示旋转动画。"""
    _CHARS = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, label: str = ""):
        self._label = label
        self._stop  = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> "Spinner":
        self._stop.clear()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def stop(self, final: str = ""):
        self._stop.set()
        if self._thread:
            self._thread.join()
        if final:
            print(f"\r{final}                    ", flush=True)
        else:
            print("\r", end="", flush=True)

    def _spin(self):
        chars = self._CHARS if _USE_COLOR else "|/-\\"
        i = 0
        while not self._stop.is_set():
            c = chars[i % len(chars)]
            print(f"\r  {c} {self._label}", end="", flush=True)
            i += 1
            time.sleep(0.1)


# ══════════════════════════════════════════════════════════════
# 状态检测
# ══════════════════════════════════════════════════════════════

def is_port_listening(port: int = PORT) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0

def is_installed() -> bool:
    return (OMNI_DIR / "gradio_demo.py").exists()

def weights_ready() -> bool:
    cap = WEIGHTS_DIR / "icon_caption"
    det = WEIGHTS_DIR / "icon_detect"
    return cap.exists() and any(cap.iterdir()) and det.exists() and any(det.iterdir())


# ══════════════════════════════════════════════════════════════
# 安装步骤
# ══════════════════════════════════════════════════════════════

def _run(cmd: list[str], cwd=None, env=None) -> int:
    return subprocess.run(cmd, cwd=cwd, env=env).returncode


def _kill_git() -> None:
    """Kill any lingering git.exe processes (Windows only)."""
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/IM", "git.exe"],
            capture_output=True
        )
        time.sleep(0.5)


def _force_rmtree(path: Path) -> None:
    """Windows 上 git 对象文件只读且可能被锁定，用系统命令强制删除。"""
    if not path.exists():
        return
    if sys.platform == "win32":
        rc = subprocess.run(
            ["cmd", "/c", "rd", "/s", "/q", str(path)],
            capture_output=True
        ).returncode
        if rc == 0 and not path.exists():
            return
        # 仍然失败则 kill git 后再试
        _kill_git()
        subprocess.run(["cmd", "/c", "rd", "/s", "/q", str(path)], capture_output=True)
    else:
        def _on_error(func, fpath, _exc):
            try:
                os.chmod(fpath, stat.S_IWRITE)
                func(fpath)
            except Exception:
                pass
        shutil.rmtree(path, onerror=_on_error)
    if path.exists():
        raise RuntimeError(f"无法删除目录: {path}")


def _proxy_reachable(proxy: str) -> bool:
    """Check if local proxy port is open."""
    try:
        host, port = proxy.rstrip("/").rsplit(":", 1)
        host = host.split("//")[-1]
        with socket.socket() as s:
            s.settimeout(1)
            return s.connect_ex((host, int(port))) == 0
    except Exception:
        return False


def clone_repo() -> None:
    section("第 1 步 / 3  克隆 OmniParser 仓库")
    if is_installed():
        ok("已克隆，跳过。")
        return

    if OMNI_DIR.exists():
        warn("检测到不完整的克隆目录，正在清理...")
        _force_rmtree(OMNI_DIR)

    for url, label, proxy in CLONE_SOURCES:
        if proxy and not _proxy_reachable(proxy):
            info(f"  [{label}] 代理端口未开放，跳过")
            continue
        info(f"  尝试: {label}  ({url[:60]})")
        env = {**os.environ}
        if proxy:
            env["HTTPS_PROXY"] = proxy
            env["HTTP_PROXY"]  = proxy
        rc = subprocess.run(
            ["git", "clone", "--progress", url, str(OMNI_DIR)],
            env=env
        ).returncode
        if rc == 0 and is_installed():
            ok(f"克隆成功 [{label}]")
            return
        warn(f"  [{label}] 失败，清理并尝试下一个...")
        if OMNI_DIR.exists():
            _force_rmtree(OMNI_DIR)

    raise RuntimeError(
        "所有镜像均失败。\n"
        f"  请手动克隆到: {OMNI_DIR}"
    )


def install_deps() -> None:
    section("第 2 步 / 3  安装 Python 依赖")

    # torch 检查（避免重复下载 2+ GB）
    has_torch = subprocess.run(
        [sys.executable, "-c", "import torch"],
        capture_output=True
    ).returncode == 0

    if not has_torch:
        info("未检测到 PyTorch，正在安装 CUDA 12.1 版本（约 2-3 GB）...")
        info("若无 NVIDIA GPU，请先手动执行：")
        info("  pip install torch --index-url https://download.pytorch.org/whl/cpu")
        rc = _run([sys.executable, "-m", "pip", "install",
                   "torch", "torchvision",
                   "--index-url", "https://download.pytorch.org/whl/cu121"])
        if rc != 0:
            raise RuntimeError("PyTorch 安装失败，请检查网络或改用 CPU 版。")

    info("安装 OmniParser 依赖包...")
    rc = _run([sys.executable, "-m", "pip", "install",
               "-r", str(OMNI_DIR / "requirements.txt")])
    if rc != 0:
        raise RuntimeError("requirements.txt 安装失败。")

    _run([sys.executable, "-m", "pip", "install", "huggingface_hub>=0.23", "-q"])
    ok("依赖安装完成。")


def _download_one(repo_id: str, target: Path, env: dict) -> bool:
    rc = subprocess.run(
        [sys.executable, "-c",
         f"from huggingface_hub import snapshot_download; "
         f"snapshot_download('{repo_id}', local_dir=r'{target}')"],
        env=env,
    ).returncode
    return rc == 0 and target.exists() and any(target.iterdir())


def download_weights() -> None:
    section("第 3 步 / 3  下载模型权重")
    if weights_ready():
        ok("权重已存在，跳过下载。")
        return

    for repo_id, local_name, desc in [
        (ICON_CAPTION_REPO, "icon_caption", "图标描述模型 (OmniParser-v2.0)"),
        (ICON_DETECT_REPO,  "icon_detect",  "图标检测模型 (Florence-2)"),
    ]:
        target = WEIGHTS_DIR / local_name
        if target.exists() and any(target.iterdir()):
            ok(f"{desc} 已下载，跳过。")
            continue

        target.mkdir(parents=True, exist_ok=True)
        info(f"下载 {desc} ...")
        success = False

        for endpoint, label, proxy in HF_ENDPOINTS:
            if proxy and not _proxy_reachable(proxy):
                info(f"  [{label}] 代理端口未开放，跳过")
                continue
            info(f"  尝试: {label} ({endpoint})")
            env = {**os.environ, "HF_ENDPOINT": endpoint}
            if proxy:
                env["HTTPS_PROXY"] = proxy
                env["HTTP_PROXY"]  = proxy
            if _download_one(repo_id, target, env):
                ok(f"  {desc} 下载完成 [{label}]")
                success = True
                break
            warn(f"  [{label}] 失败")

        if not success:
            warn(f"{desc} 所有渠道均失败，请手动下载到: {target}")

    ok("权重步骤完成。")


def install() -> None:
    clone_repo()
    install_deps()
    download_weights()


# ══════════════════════════════════════════════════════════════
# 启动 & 验证
# ══════════════════════════════════════════════════════════════

def start_service() -> subprocess.Popen:
    section("启动 OmniParser V2 服务")
    proc = subprocess.Popen(
        [sys.executable, "gradio_demo.py"],
        cwd=str(OMNI_DIR),
    )
    info(f"进程已启动，PID: {proc.pid}")
    return proc


def wait_for_service(proc: subprocess.Popen, timeout: int = 120) -> bool:
    """轮询端口直到服务就绪，同时监控子进程是否提前退出。"""
    section("验证服务是否就绪")
    info(f"等待 {SERVICE_URL} 响应（最长 {timeout} 秒）...")

    deadline = time.time() + timeout
    interval = 2.0
    elapsed  = 0

    while time.time() < deadline:
        # 检查进程是否意外退出
        if proc.poll() is not None:
            err(f"gradio_demo.py 意外退出，退出码: {proc.returncode}")
            return False

        if is_port_listening():
            print(flush=True)
            ok(f"服务已就绪！地址: {SERVICE_URL}")
            return True

        elapsed = int(time.time() - (deadline - timeout))
        remaining = timeout - elapsed
        print(f"\r  等待中... 已用 {elapsed}s / 剩余 {remaining}s", end="", flush=True)
        time.sleep(interval)

    print(flush=True)
    return False


# ══════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════

def check_status() -> None:
    installed = is_installed()
    weights   = weights_ready()
    running   = is_port_listening()

    print()
    print("=" * 52)
    print("  OmniParser V2  状态检查")
    print("=" * 52)
    print(f"  已克隆   : {'是' if installed else '否'}  ({OMNI_DIR})")
    print(f"  权重就绪 : {'是' if weights   else '否'}  ({WEIGHTS_DIR})")
    print(f"  服务运行 : {'是' if running    else '否'}  ({SERVICE_URL})")
    print("=" * 52)


def main() -> int:
    # Windows 控制台启用 UTF-8
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="OmniParser V2 安装 & 启动")
    parser.add_argument("--check",   action="store_true", help="仅检查状态，不安装不启动")
    parser.add_argument("--timeout", type=int, default=120, help="等待服务就绪超时秒数（默认 120）")
    args = parser.parse_args()

    print()
    print("=" * 52)
    print("  OmniParser V2  —  安装 & 启动器")
    print("=" * 52)

    if args.check:
        check_status()
        return 0

    # 已在运行？
    if is_port_listening():
        ok(f"服务已在运行，端口 {PORT}：{SERVICE_URL}")
        return 0

    # 需要安装？
    if not is_installed() or not weights_ready():
        try:
            install()
        except Exception as exc:
            err(f"安装失败：{exc}")
            return 1

    # 启动
    proc = start_service()

    # 验证
    success = wait_for_service(proc, timeout=args.timeout)
    if not success:
        err(f"服务在 {args.timeout} 秒内未就绪，请查看 gradio_demo.py 输出")
        proc.terminate()
        return 1

    print()
    print("=" * 52)
    print(f"  服务地址: {SERVICE_URL}")
    print(f"  PID: {proc.pid}   按 Ctrl+C 停止")
    print("=" * 52)
    print()

    try:
        proc.wait()
    except KeyboardInterrupt:
        info("正在停止服务...")
        proc.terminate()

    return 0


if __name__ == "__main__":
    sys.exit(main())
