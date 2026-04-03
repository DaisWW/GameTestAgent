import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass  # pytest 捕获模式、IDE 内嵌终端或文件重定向时， stdout 无 reconfigure 方法


def info(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


def ok(msg: str) -> None:
    print(f"[ OK ] {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"[WARN] {msg}", flush=True)


def err(msg: str) -> None:
    print(f"[ERR ] {msg}", file=sys.stderr, flush=True)


@dataclass
class RunnerSettings:
    game_package: str = ""
    game_activity: str = ""
    game_launch_wait: float = 5.0
    crash_log_lines: int = 100
