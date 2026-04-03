"""AutoTestAgent ADB 工具层自定义异常体系。

层级::

    ADBError
    ├── DeviceNotFoundError    # 设备未连接或 serial 不匹配
    ├── CommandFailedError     # adb 命令返回非 0
    └── RecordingError         # 录屏相关异常
"""


class ADBError(RuntimeError):
    """ADB 工具层基础异常，调用方可 catch 此类捕获所有 ADB 相关错误。"""


class DeviceNotFoundError(ADBError):
    """设备未找到或指定 serial 不在线。"""


class CommandFailedError(ADBError):
    """adb 命令执行失败（returncode != 0）。

    Attributes:
        returncode: 子进程返回码。
        detail:     stdout/stderr 合并内容。
    """

    def __init__(self, message: str, returncode: int = -1, detail: str = "") -> None:
        super().__init__(message)
        self.returncode = returncode
        self.detail = detail


class RecordingError(ADBError):
    """录屏生命周期相关异常（重复启动、未启动就停止等）。"""
