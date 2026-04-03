import io
import logging
import os
import re
import subprocess
import time
from typing import List, Optional

from .adb_setup import ensure_adb
from .core import RunnerSettings
from .exceptions import ADBError, CommandFailedError, DeviceNotFoundError
from .screen_recorder import ScreenRecorder
from .utils import escape_for_adb_input
from core.types import SwipeDirection

logger = logging.getLogger(__name__)


class ADBController:
    def __init__(self, settings: RunnerSettings, serial: str = ""):
        self.settings = settings
        self.serial = serial
        self.dev_w = 0
        self.dev_h = 0
        self._adb = ensure_adb()
        self._base = [self._adb]
        if serial:
            self._base += ["-s", serial]
        self._recorder: Optional[ScreenRecorder] = None

    # ── 内部辅助 ──────────────────────────────────────────────────

    def _run(self, args, **kwargs):
        return subprocess.run(self._base + list(args), **kwargs)

    def _result_detail(self, result) -> str:
        parts = [str(v).strip() for v in (getattr(result, "stdout", ""), getattr(result, "stderr", "")) if v and str(v).strip()]
        return "\n".join(parts)

    def _run_checked(self, args, error_message: str):
        result = self._run(args, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode != 0:
            detail = self._result_detail(result)
            raise CommandFailedError(
                f"{error_message}: {detail}" if detail else error_message,
                returncode=result.returncode,
                detail=detail,
            )
        return result

    def _norm_to_px(self, nx: float, ny: float) -> tuple[int, int]:
        """将归一化坐标 (0-999) 转为像素坐标。调用前须先 connect()。"""
        if self.dev_w == 0 or self.dev_h == 0:
            raise ADBError("屏幕分辨率未知，请先调用 connect()")
        nx = max(0.0, min(999.0, nx))
        ny = max(0.0, min(999.0, ny))
        return int(nx / 1000 * self.dev_w), int(ny / 1000 * self.dev_h)

    # ── 设备连接 ──────────────────────────────────────────────────

    def connect(self) -> None:
        result = subprocess.run(
            [self._adb, "devices"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if result.returncode != 0:
            detail = self._result_detail(result) or "unknown error"
            logger.error("ADB devices 失败：%s", detail)
            raise CommandFailedError(f"adb devices 失败: {detail}", returncode=result.returncode)

        lines = [l.strip() for l in result.stdout.splitlines() if "\tdevice" in l]
        if not lines:
            logger.error("未检测到已连接的 ADB 设备")
            logger.error("请确认：① USB 调试已开启  ② adb devices 能看到设备")
            raise DeviceNotFoundError("未检测到已连接的 ADB 设备")

        if self.serial and all(not l.startswith(f"{self.serial}\t") for l in lines):
            logger.error("未找到指定设备：%s", self.serial)
            logger.error("当前可用设备：%s", [l.split()[0] for l in lines])
            raise DeviceNotFoundError(f"未找到指定设备：{self.serial}")

        if not self.serial and len(lines) > 1:
            logger.warning("检测到多个设备，默认使用第一个：%s", [l.split()[0] for l in lines])

        if not self.serial:
            self.serial = lines[0].split()[0]
            self._base = [self._adb, "-s", self.serial]

        logger.info("ADB 设备：%s", self.serial)

        res = self._run(["shell", "wm", "size"], capture_output=True, text=True)
        match = (
            re.search(r"Override size:\s*(\d+)x(\d+)", res.stdout)
            or re.search(r"Physical size:\s*(\d+)x(\d+)", res.stdout)
            or re.search(r"(\d+)x(\d+)", res.stdout)
        )
        if match:
            self.dev_w, self.dev_h = int(match.group(1)), int(match.group(2))
            logger.info("屏幕分辨率：%dx%d", self.dev_w, self.dev_h)
        else:
            logger.warning("无法获取屏幕分辨率，使用默认 1080×2400")
            self.dev_w, self.dev_h = 1080, 2400

    # ── 截图 ──────────────────────────────────────────────────────

    def screenshot(self):
        from PIL import Image

        for attempt in range(3):
            result = self._run(["exec-out", "screencap", "-p"], capture_output=True)
            if result.returncode == 0 and result.stdout:
                image = Image.open(io.BytesIO(result.stdout))
                self.dev_w, self.dev_h = image.width, image.height
                return image
            # 失败：先关键盘/弹窗，再唤屏，然后重试
            logger.warning("screencap 第 %d 次失败，尝试关键盘/唤屏后重试...", attempt + 1)
            self._run(["shell", "input", "keyevent", "KEYCODE_BACK"],  check=False)
            time.sleep(0.3)
            self._run(["shell", "input", "keyevent", "KEYCODE_WAKEUP"], check=False)
            time.sleep(0.5)

        raise CommandFailedError("screencap 失败（重试 3 次）", returncode=-1)

    # ── 手势操作 ──────────────────────────────────────────────────

    def tap(self, nx: float, ny: float) -> None:
        x, y = self._norm_to_px(nx, ny)
        self._run_checked(["shell", "input", "tap", str(x), str(y)], "点击失败")
        time.sleep(0.3)

    def swipe(self, nx: float, ny: float, direction: str, distance_pct: float = 0.3) -> None:
        """direction: up | down | left | right"""
        _DIRS = {SwipeDirection.UP, SwipeDirection.DOWN, SwipeDirection.LEFT, SwipeDirection.RIGHT}
        if direction not in _DIRS:
            raise ValueError(f"非法方向 {direction!r}，可选：{_DIRS}")
        x, y = self._norm_to_px(nx, ny)
        dist_v = int(self.dev_h * distance_pct)
        dist_h = int(self.dev_w * distance_pct)
        dx, dy = {SwipeDirection.UP: (0, -dist_v), SwipeDirection.DOWN: (0, dist_v), SwipeDirection.LEFT: (-dist_h, 0), SwipeDirection.RIGHT: (dist_h, 0)}[direction]
        x2 = max(0, min(self.dev_w - 1, x + dx))
        y2 = max(0, min(self.dev_h - 1, y + dy))
        self._run_checked(["shell", "input", "swipe", str(x), str(y), str(x2), str(y2), "400"], "滑动失败")
        time.sleep(0.3)

    # ── 输入 ──────────────────────────────────────────────────────

    def input_text(self, text: str) -> None:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
        if not normalized:
            return
        non_ascii_warned = False
        segments = normalized.split("\n")
        for index, segment in enumerate(segments):
            safe_text, has_non_ascii = escape_for_adb_input(segment)
            if has_non_ascii and not non_ascii_warned:
                logger.warning("输入文本含非 ASCII 字符，部分设备可能无法正确输入")
                non_ascii_warned = True
            if safe_text:
                self._run_checked(["shell", "input", "text", safe_text], "输入文本失败")
            if index < len(segments) - 1:
                self.press_enter()

    def press_key(self, key: str) -> None:
        code = {"back": "4", "home": "3", "enter": "66"}.get(key)
        if not code:
            raise ValueError(f"unsupported key: {key}")
        self._run_checked(["shell", "input", "keyevent", code], "按键失败")

    def press_back(self) -> None:
        self.press_key("back")

    def press_home(self) -> None:
        self.press_key("home")

    def press_enter(self) -> None:
        self.press_key("enter")

    # ── 游戏控制 ──────────────────────────────────────────────────

    def launch_game(self) -> None:
        package = self.settings.game_package
        activity = self.settings.game_activity
        self._run(["shell", "am", "force-stop", package], check=False)
        time.sleep(1.0)
        result = self._run_checked(
            ["shell", "am", "start", "-W", "-n", f"{package}/{activity}"],
            "启动游戏失败",
        )
        detail = self._result_detail(result)
        if any(m in detail.lower() for m in ("error:", "exception", "does not exist", "unable to resolve")):
            raise CommandFailedError(f"启动游戏失败: {detail}")
        time.sleep(self.settings.game_launch_wait)
        # 清空 logcat 缓冲区，防止 check_crash 读到上一轮的崩溃日志导致误报
        self._run(["logcat", "-c"], capture_output=True)

    # ── 录屏（委托给 ScreenRecorder）──────────────────────────

    def start_recording(self, remote_dir: str = "/sdcard", segment_secs: int = 170) -> None:
        """启动分段录屏，每段 segment_secs 秒自动续接，绕过设备 180s 硬限。"""
        if self._recorder and self._recorder.is_recording:
            logger.warning("录屏已在进行中，先停止旧录屏再启动")
            self.stop_recording()
        self._recorder = ScreenRecorder(self._base, remote_dir=remote_dir, segment_secs=segment_secs)
        self._recorder.start()

    def stop_recording(self) -> List[str]:
        """停止录屏，返回设备上的分段文件路径列表。"""
        if not self._recorder:
            return []
        paths = self._recorder.stop()
        self._recorder = None
        return paths

    def pull_recording(self, remote_paths: List[str], local_dir: str = ".") -> List[str]:
        """拉取所有分段到本地，返回本地路径列表。"""
        recorder = ScreenRecorder(self._base)
        return recorder.pull(remote_paths, local_dir)

    def pull_file(self, remote: str, local: str) -> None:
        self._run_checked(["pull", remote, local], f"拉取文件失败：{remote}")
        self._run(["shell", "rm", "-f", remote], check=False)

    # ── 日志 & 崩溃检测 ──────────────────────────────────────────

    def _get_pids(self, package: str) -> set[str]:
        """获取包名对应的所有进程 PID（支持多进程游戏）。"""
        result = self._run(
            ["shell", "pidof", package],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        return set(result.stdout.strip().split()) if result.returncode == 0 and result.stdout.strip() else set()

    def logcat(self, lines: Optional[int] = None, package: Optional[str] = None) -> str:
        """拉取 logcat。

        package: 过滤指定包名的进程日志；不传则使用 settings.game_package；
                 传入空字符串 "" 则返回全量日志。
        """
        line_count = lines if lines is not None else self.settings.crash_log_lines
        pkg = self.settings.game_package if package is None else package

        result = self._run(
            ["logcat", "-d", "-t", str(line_count), "-v", "threadtime"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )

        if not pkg:
            return result.stdout

        pids = self._get_pids(pkg)
        filtered = []
        for line in result.stdout.splitlines(keepends=True):
            parts = line.split()
            # threadtime 格式：date time PID TID level tag: msg
            pid_match = len(parts) >= 3 and parts[2] in pids
            if pid_match or pkg in line:
                filtered.append(line)
        return "".join(filtered)

    def check_crash(self, package: Optional[str] = None) -> bool:
        output = self.logcat(lines=200, package=package)
        patterns = [
            "FATAL EXCEPTION", "ANR in",
            "java.lang.RuntimeException", "NullPointerException",
            "Force finishing activity",
            "NullReferenceException", "IndexOutOfRangeException",
            "StackOverflowException", "OutOfMemoryException", "ExecutionEngineException",
            "Fatal signal", "SIGABRT", "SIGSEGV",
            "*** *** *** *** *** ***", "backtrace:", "native crash",
        ]
        return any(p in output for p in patterns)
