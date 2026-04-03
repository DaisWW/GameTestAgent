from __future__ import annotations

"""异常恢复工作流 (Recovery Flow)

当主流程（waterfall_flow）检测到以下情况时触发本工作流：
  - UI 卡死（连续多帧 Hash 无变化）
  - ABA 循环（Agent 在两个页面间来回跳转）
  - 游戏崩溃（由主流程自动保存 Bug 快照后移交）
  - 执行动作连续失败

恢复策略（优先级从高到低）：
  1. wait(3)      — 等待 UI 自行恢复（网络加载、动画等）
  2. press_home   — 退出到桌面
  3. launch_game  — 重新启动游戏（ADB am start）
  4. abandon      — 放弃本次测试，报告 fail

架构说明：
  本文件是独立的 LangGraph 子图，由主流程通过
  ``worker.run_recovery(reason)`` 调用，恢复后交还控制权。
  当前为 v1 实现（基于规则），后续可扩展为 LLM 驱动的恢复。
"""

import logging
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from langgraph.graph import StateGraph, END
from core.types import RecoveryStrategy
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from core.agent.worker import LangGraphWorker

logger = logging.getLogger(__name__)

_MAX_RECOVERY_STEPS = 5


class RecoveryState(TypedDict, total=False):
    reason:         str     # 触发恢复的原因描述
    step:           int     # 恢复步数计数
    recovered:      bool    # True = 恢复成功，可重回主流程
    give_up:        bool    # True = 无法恢复，终止测试
    strategy_index: int     # 当前尝试的恢复策略索引


_STRATEGIES = [
    RecoveryStrategy.WAIT,
    RecoveryStrategy.PRESS_HOME,
    RecoveryStrategy.LAUNCH_GAME,
    RecoveryStrategy.ABANDON,
]


def _make_recover_node(worker: "LangGraphWorker"):
    def recover_node(state: RecoveryState) -> RecoveryState:
        step  = state.get("step", 0)
        idx   = state.get("strategy_index", 0)
        reason = state.get("reason", "未知异常")

        if step >= _MAX_RECOVERY_STEPS or idx >= len(_STRATEGIES):
            logger.error("[Recovery] 已达最大恢复步数，放弃测试。原因: %s", reason)
            return {**state, "give_up": True}

        strategy = _STRATEGIES[idx]
        logger.info("[Recovery Step %d] 策略: %s  原因: %s", step, strategy, reason[:60])

        adb = worker._get_adb()

        try:
            if strategy == RecoveryStrategy.WAIT:
                time.sleep(3)
            elif strategy == RecoveryStrategy.PRESS_HOME:
                adb.press_home()
                time.sleep(2)
            elif strategy == RecoveryStrategy.LAUNCH_GAME:
                adb.launch_game()
                time.sleep(worker.config.adb.game_launch_wait)
            elif strategy == RecoveryStrategy.ABANDON:
                return {**state, "give_up": True}
        except Exception as exc:
            logger.warning("[Recovery] 策略 %s 执行失败: %s", strategy, exc)

        return {**state, "step": step + 1, "strategy_index": idx + 1}
    return recover_node


def _make_verify_node(worker: "LangGraphWorker"):
    def verify_node(state: RecoveryState) -> RecoveryState:
        if state.get("give_up"):
            return state

        try:
            screenshot = worker.capture()
            elements   = worker.detect(screenshot)
            if elements:
                logger.info("[Recovery] 检测到 %d 个 UI 元素，判断为已恢复", len(elements))
                return {**state, "recovered": True}
            logger.warning("[Recovery] 截图成功但未检测到 UI 元素，继续尝试下一策略")
            return {**state, "recovered": False}
        except Exception as exc:
            logger.warning("[Recovery] 截图/检测失败，继续尝试下一策略: %s", exc)
            return {**state, "recovered": False}
    return verify_node


def _route(state: RecoveryState) -> str:
    if state.get("give_up") or state.get("recovered"):
        return END
    return "recover"


def build_recovery_graph(worker: "LangGraphWorker"):
    """构建恢复子图。

    Returns:
        已编译的 CompiledGraph，调用 .invoke({"reason": "..."}) 启动。
        结束后通过 state["recovered"] / state["give_up"] 判断结果。
    """
    graph = StateGraph(RecoveryState)

    graph.add_node("recover", _make_recover_node(worker))
    graph.add_node("verify",  _make_verify_node(worker))

    graph.set_entry_point("recover")
    graph.add_edge("recover", "verify")
    graph.add_conditional_edges("verify", _route, {"recover": "recover", END: END})

    return graph.compile()
