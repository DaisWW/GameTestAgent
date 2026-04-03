from __future__ import annotations

"""ActionExecutor：将 LLM 决策字典翻译为 ADB 动作。

与 LangGraphWorker 解耦：仅持有 ADBController 引用，
不依赖 Graph State、记忆系统或视觉层，便于单独测试。
"""

import logging
import time
from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from tools.adb_controller import ADBController

logger = logging.getLogger(__name__)

# 注册表：action 名 → 处理函数 (executor, action_dict) → None
_HANDLERS: Dict[str, Any] = {}


def _register(name: str):
    def decorator(fn):
        _HANDLERS[name] = fn
        return fn
    return decorator


@_register("tap")
def _tap(adb: "ADBController", action: Dict[str, Any]) -> None:
    bbox = action.get("_bbox")
    if not bbox:
        logger.warning("tap 动作缺少 _bbox，跳过")
        return
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    adb.tap(cx, cy)


@_register("swipe")
def _swipe(adb: "ADBController", action: Dict[str, Any]) -> None:
    bbox      = action.get("_bbox")
    direction = action.get("params", {}).get("direction", "up")
    if not bbox:
        logger.warning("swipe 动作缺少 _bbox，跳过")
        return
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    adb.swipe(cx, cy, direction)


@_register("input_text")
def _input_text(adb: "ADBController", action: Dict[str, Any]) -> None:
    adb.input_text(action.get("params", {}).get("text", ""))


@_register("press_back")
def _press_back(adb: "ADBController", action: Dict[str, Any]) -> None:
    adb.press_back()


@_register("press_home")
def _press_home(adb: "ADBController", action: Dict[str, Any]) -> None:
    adb.press_home()


@_register("wait")
def _wait(adb: "ADBController", action: Dict[str, Any]) -> None:
    time.sleep(float(action.get("params", {}).get("seconds", 2)))


@_register("done")
def _done(adb: "ADBController", action: Dict[str, Any]) -> None:
    pass  # 任务完成标记，无需 ADB 操作


class ActionExecutor:
    """将 LLM ActionDecision 字典翻译为 ADB 设备操作。

    Args:
        adb: ADBController 实例。
    """

    def __init__(self, adb: "ADBController") -> None:
        self._adb = adb

    def execute(self, action: Dict[str, Any]) -> None:
        """执行单条 ActionDecision。

        Args:
            action: LLM 输出的决策字典，格式为::

                {
                    "action":    "tap" | "swipe" | ...,
                    "params":    {...},
                    "_bbox":     [x1, y1, x2, y2],   # cognition 节点注入
                    "reasoning": "...",
                }

        Raises:
            tools.exceptions.ADBError: ADB 命令执行失败时向上传播。
        """
        act = action.get("action", "wait")
        handler = _HANDLERS.get(act)
        if handler is None:
            logger.warning("未知动作类型: %s，跳过", act)
            return
        logger.debug("执行动作: %s  params=%s", act, action.get("params"))
        handler(self._adb, action)

    def __repr__(self) -> str:
        return f"ActionExecutor(adb={self._adb!r})"
