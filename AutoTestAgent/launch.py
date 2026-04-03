"""
AutoTestAgent 一键启动器。

负责：
  1. 按配置自动判断是否需要先启动 OmniParser 服务
     优先读取 .env，不存在时 fallback 到 .env.example
  2. 等待 OmniParser 就绪后启动 Agent

用法（通常通过 run.bat 调用，也可直接用）:
  python launch.py --task "测试登录流程"
  python launch.py --task "调试" --vision mock
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).parent

_DEFAULT_OMNI_PORT = 7861


# ── 读 .env（不依赖 python-dotenv）──────────────────────────────
def _read_env(env_file: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    if not env_file.exists():
        return result
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        v = v.strip()
        # 去掉引号内的值直接保留，不处理内联注释
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            v = v[1:-1]
        else:
            # 无引号时去掉行尾 # 注释
            v = v.split("#")[0].strip() if "#" in v else v
        result[k.strip()] = v
    return result


# ── 端口检测 ─────────────────────────────────────────────────────
def _port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex((host, port)) == 0


# ── OmniParser 自动启动 ─────────────────────────────────────────
def _start_omniparser(port: int = _DEFAULT_OMNI_PORT) -> subprocess.Popen | None:
    """
    以后台进程调用 models/omni/omniparser.py。
    omniparser.py 自身处理：检查已安装 → 下载权重 → 启动 gradio → 等待就绪。
    本函数只负责：启动它，然后轮询端口直到就绪（不设上限，由 omniparser 自己控制）。
    """
    omni_py = ROOT / "models" / "omni" / "omniparser.py"
    if not omni_py.exists():
        print(f"[!] 找不到 {omni_py}，跳过 OmniParser 自动启动", flush=True)
        return None

    # 输出直接打到当前终端，让用户看到 omniparser 的进度
    proc = subprocess.Popen(
        [sys.executable, str(omni_py)],
    )

    # 轮询端口；omniparser.py 本身会在就绪后返回，所以进程结束 = 服务已起或失败
    while True:
        if _port_open(port):
            return proc
        code = proc.poll()
        if code is not None:
            if code == 0 and _port_open(port):
                return proc          # 已在运行（omniparser 发现端口已开直接 exit 0）
            print(f"[✗] OmniParser 启动失败 (exit={code})", flush=True)
            return None
        time.sleep(2)


# ── 主逻辑 ──────────────────────────────────────────────────────
def main() -> int:
    # ── 读取配置（.env 优先，否则 fallback 到 .env.example）──
    env_file = ROOT / ".env" if (ROOT / ".env").exists() else ROOT / ".env.example"
    cfg = _read_env(env_file)
    vision = cfg.get("VISION_TYPE", "omni_v2")

    # ── 判断是否需要自动启动 OmniParser ──────────
    omni_proc: subprocess.Popen | None = None

    from core.types import VisionType  # noqa: PLC0415
    if vision == VisionType.OMNI_V2:
        # 从 OMNI_ENDPOINT 提取端口，避免硬编码
        endpoint = cfg.get("OMNI_ENDPOINT", f"http://127.0.0.1:{_DEFAULT_OMNI_PORT}")
        try:
            omni_port = urlparse(endpoint).port or _DEFAULT_OMNI_PORT
        except Exception:
            omni_port = _DEFAULT_OMNI_PORT

        if _port_open(omni_port):
            print(f"[✓] OmniParser 服务已在运行 (:{omni_port})", flush=True)
        else:
            omni_proc = _start_omniparser(omni_port)
            if omni_proc is None:
                return 1

    # ── 启动 Agent ───────────────────────────────────
    import main as agent_main  # noqa: PLC0415
    result = agent_main.main()

    # OmniParser 服务故意保留运行，下次测试可直接复用
    if omni_proc and omni_proc.poll() is None:
        print(f"[→] OmniParser 服务保持运行  {cfg.get('OMNI_ENDPOINT', 'http://127.0.0.1:' + str(_DEFAULT_OMNI_PORT))}", flush=True)

    return result


if __name__ == "__main__":
    # 透传所有参数给 main.py（sys.argv[1:] 由 main._parse_args() 直接读取）
    sys.exit(main())
