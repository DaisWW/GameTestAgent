"""ScreenRecorder：封装分段录屏逻辑，独立于 ADBController。

支持用法：
    recorder = ScreenRecorder(base_cmd=[adb, "-s", serial])
    recorder.start()
    ...
    paths = recorder.stop()          # 返回设备上的 mp4 路径列表
    local = recorder.pull(local_dir) # 拉取到本机
"""
from __future__ import annotations

import os
import subprocess
import threading
import time
from typing import List

from .exceptions import CommandFailedError, RecordingError


class ScreenRecorder:
    """分段屏幕录制器，绕过 Android 180s 硬限制。

    Args:
        base_cmd:      adb 基础命令列表，例如 ["adb", "-s", "emulator-5554"]。
        remote_dir:    视频保存在设备上的目录，默认 /sdcard。
        segment_secs:  每段录制时长（秒），应 < 180，默认 170。
    """

    def __init__(
        self,
        base_cmd: List[str],
        remote_dir: str = "/sdcard",
        segment_secs: int = 170,
    ) -> None:
        self._base      = base_cmd
        self._remote_dir = remote_dir.rstrip("/")
        self._seg_secs  = segment_secs
        self._stop_evt: threading.Event | None = None
        self._thread:   threading.Thread | None = None
        self._paths:    List[str] = []
        self._idx = 0

    # ── 公共接口 ──────────────────────────────────────────────

    @property
    def is_recording(self) -> bool:
        return self._stop_evt is not None and not self._stop_evt.is_set()

    def start(self) -> None:
        """启动分段录屏线程。重复调用抛出 RecordingError。"""
        if self.is_recording:
            raise RecordingError("录屏已在进行中，请先调用 stop()")
        self._stop_evt = threading.Event()
        self._paths    = []
        self._idx      = 0
        self._thread   = threading.Thread(target=self._loop, daemon=True, name="ScreenRecorder")
        self._thread.start()

    def stop(self) -> List[str]:
        """停止录屏，返回设备上各分段的路径列表。

        未调用 start() 直接 stop() 返回空列表（不抛异常）。
        """
        if self._stop_evt is None:
            return []
        self._stop_evt.set()
        # 终止设备上还在录制的进程
        subprocess.run(
            self._base + ["shell", "pkill", "-f", "screenrecord"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if self._thread:
            self._thread.join(timeout=10)
        paths = list(self._paths)
        self._stop_evt = None
        self._thread   = None
        return paths

    def pull(self, remote_paths: List[str], local_dir: str = ".") -> List[str]:
        """将设备上的录屏文件拉取到本机，同时从设备删除原文件。

        Args:
            remote_paths: stop() 返回的路径列表。
            local_dir:    本机目标目录。

        Returns:
            本机文件路径列表。
        """
        os.makedirs(local_dir, exist_ok=True)
        local_paths = []
        for remote in remote_paths:
            local = os.path.join(local_dir, os.path.basename(remote))
            try:
                subprocess.run(
                    self._base + ["pull", remote, local],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            except subprocess.CalledProcessError as exc:
                raise CommandFailedError(
                    f"拉取录屏文件失败: {remote}",
                    returncode=exc.returncode,
                ) from exc
            subprocess.run(
                self._base + ["shell", "rm", "-f", remote],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            local_paths.append(local)
        return local_paths

    # ── 内部 ──────────────────────────────────────────────────

    def _loop(self) -> None:
        while not self._stop_evt.is_set():
            path = f"{self._remote_dir}/rec_{self._idx:04d}.mp4"
            self._paths.append(path)
            self._idx += 1
            proc = subprocess.Popen(
                self._base + ["shell", "screenrecord", "--time-limit", str(self._seg_secs), path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            while proc.poll() is None and not self._stop_evt.is_set():
                time.sleep(0.5)
            if proc.poll() is None:
                proc.terminate()

    def __repr__(self) -> str:
        state = "recording" if self.is_recording else "idle"
        return f"ScreenRecorder(state={state}, segments={len(self._paths)})"
