import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

_logger = logging.getLogger("tools")


def configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass  # pytest 捕获模式、IDE 内嵌终端或文件重定向时， stdout 无 reconfigure 方法


def info(msg: str) -> None:
    _logger.info(msg)


def ok(msg: str) -> None:
    _logger.info("[ OK ] %s", msg)


def warn(msg: str) -> None:
    _logger.warning(msg)


def err(msg: str) -> None:
    _logger.error(msg)


@dataclass
class RunnerSettings:
    game_package: str = ""
    game_activity: str = ""
    game_launch_wait: float = 5.0
    crash_log_lines: int = 100
