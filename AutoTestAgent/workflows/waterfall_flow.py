from __future__ import annotations

"""LangGraph 瀑布流主工作流。

节点顺序：perception → cognition → execute → validate → check → (loop or end)

节点职责：
    perception  截图 + OmniParser + ImageHash + 导航图查询 + 上下文构建
    cognition   调用 LLMAdapter，输入完整 ContextPacket，输出决策
    execute     ADB 执行动作
    validate    再次截图，计算 Hash 变化，更新三层记忆
    check       崩溃检测 + 超步终止 + Bug 快照保存
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

from core.vision.perception import compute_phash, is_page_changed
from core.memory.working_memory import MemoryStep

if TYPE_CHECKING:
    from core.agent.worker import LangGraphWorker

logger = logging.getLogger(__name__)


# ── State schema ─────────────────────────────────────────────────

class AgentState(TypedDict, total=False):
    task:           str
    screenshot:     Any
    ui_elements:    List[Dict[str, Any]]
    page_hash:      str
    prev_hash:      str
    context_packet: Any
    current_action: Optional[Dict[str, Any]]
    step:           int
    done:           bool
    result:         str
    output_dir:     str


# ── 节点工厂 ──────────────────────────────────────────────────────

def _make_perception_node(worker: "LangGraphWorker"):
    def perception_node(state: AgentState) -> AgentState:
        step = state.get("step", 0)
        logger.info("[Step %02d] ── PERCEPTION ──────────────────────", step)

        screenshot  = worker.capture()
        worker.save_screenshot(screenshot, step, state.get("output_dir", "data"))

        ui_elements = worker.detect(screenshot)
        logger.info("  OmniParser: %d 个元素", len(ui_elements))

        page_hash = compute_phash(screenshot)
        logger.info("  PageHash: %s", page_hash[:8])

        shot_path = worker.save_page_screenshot(screenshot, page_hash, state.get("output_dir", "data"))
        worker.nav_graph.register_page(page_hash, elements=ui_elements, screenshot_path=shot_path)

        packet = worker.context_builder.build(
            screenshot=screenshot,
            ui_elements=ui_elements,
            page_hash=page_hash,
            task=state["task"],
            step=step,
        )

        if packet.anomaly_flag:
            logger.warning("  [ANOMALY] %s", packet.anomaly_flag)

        return {"screenshot": screenshot, "ui_elements": ui_elements,
                "page_hash": page_hash, "context_packet": packet}
    return perception_node


def _make_cognition_node(worker: "LangGraphWorker"):
    def cognition_node(state: AgentState) -> AgentState:
        step = state.get("step", 0)
        logger.info("[Step %02d] ── COGNITION ───────────────────────", step)

        if step >= worker.config.max_steps:
            logger.warning("  已达最大步数 %d", worker.config.max_steps)
            return {"done": True, "result": f"超出最大步数 {worker.config.max_steps}"}

        packet   = state["context_packet"]
        decision = worker.llm.ask(packet)

        elem_id = decision.get("params", {}).get("id")
        if elem_id is not None:
            elem_map = {e["id"]: e for e in state.get("ui_elements", [])}
            elem = elem_map.get(elem_id)
            if elem:
                decision["_bbox"]          = elem["bbox"]
                decision["_element_label"] = elem.get("label", "")
            else:
                logger.warning("  Brain 引用了不存在的元素 id=%s", elem_id)

        logger.info("  决策: action=%-12s params=%s", decision.get("action"), decision.get("params"))
        logger.info("  推理: %s", decision.get("reasoning", "")[:80])

        return {"current_action": decision,
                "done": bool(decision.get("done", False)),
                "result": decision.get("result", "")}
    return cognition_node


def _make_execute_node(worker: "LangGraphWorker"):
    def execute_node(state: AgentState) -> AgentState:
        action = state.get("current_action")
        if not action or state.get("done"):
            return {}

        step = state.get("step", 0)
        logger.info("[Step %02d] ── EXECUTE: %s %s", step, action.get("action"), action.get("params"))

        try:
            worker.execute(action)
        except Exception as exc:
            logger.error("  执行失败: %s", exc)
            return {"done": True, "result": f"执行失败: {exc}"}

        return {}
    return execute_node


def _make_validate_node(worker: "LangGraphWorker"):
    def validate_node(state: AgentState) -> AgentState:
        if state.get("done"):
            return {}

        step   = state.get("step", 0)
        action = state.get("current_action", {})
        logger.info("[Step %02d] ── VALIDATE ─────────────────────────", step)

        if worker.config.step_delay > 0:
            time.sleep(worker.config.step_delay)

        after_shot  = worker.capture()
        result_hash = compute_phash(after_shot)
        prev_hash   = state.get("page_hash", "")
        page_changed = is_page_changed(prev_hash, result_hash)

        logger.info(
            "  Hash: %s → %s  [%s]",
            prev_hash[:8], result_hash[:8],
            "CHANGED" if page_changed else "NO_CHANGE",
        )

        # ── 第一层：工作记忆 ────────────────────────────
        worker.working_memory.push(MemoryStep(
            step=step,
            action=action.get("action", "unknown"),
            params=action.get("params", {}),
            reasoning=action.get("reasoning", ""),
            page_hash=prev_hash,
            result_hash=result_hash,
            success=page_changed,
            element_label=action.get("_element_label", ""),
        ))

        # ── 第二层：导航图 ────────────────────────────────────────
        ng      = worker.nav_graph
        act_type = action.get("action", "")
        elem_id  = action.get("params", {}).get("id")

        if act_type == "tap" and elem_id is not None:
            ng.mark_visited(prev_hash, elem_id)

        if page_changed and prev_hash:
            ng.add_transition(
                from_hash=prev_hash, to_hash=result_hash,
                action=act_type, element_id=elem_id or -1,
                element_label=action.get("_element_label", ""),
            )
            ng.register_page(result_hash)

        return {"prev_hash": prev_hash, "page_hash": result_hash, "step": step + 1}
    return validate_node


def _make_check_node(worker: "LangGraphWorker"):
    def check_node(state: AgentState) -> AgentState:
        if state.get("done"):
            return {}

        try:
            if worker._get_adb().check_crash():
                logger.error("[CRASH] 检测到游戏崩溃！")
                page_hash = state.get("page_hash", "")
                shot_path = ""
                screenshot = state.get("screenshot")
                if page_hash and screenshot:
                    shot_path = os.path.join(
                        state.get("output_dir", "data"),
                        "screenshots",
                        f"crash_{page_hash[:8]}_step{state.get('step',0):03d}.png",
                    )
                    os.makedirs(os.path.dirname(shot_path), exist_ok=True)
                    screenshot.save(shot_path)
                if page_hash:
                    worker.experience_pool.save_bug(
                        page_hash=page_hash,
                        description=f"崩溃，任务: {state.get('task','')[:60]}",
                        screenshot_path=shot_path,
                        tags=["crash"],
                    )
                return {"done": True, "result": "游戏崩溃"}
        except Exception as exc:
            logger.warning("崩溃检测失败（忽略）: %s", exc)

        return {}
    return check_node


# ── 条件路由 ─────────────────────────────────────────────────────

def _should_continue(state: AgentState) -> str:
    if state.get("done"):
        logger.info("=== 测试结束: %s ===", state.get("result", "未知"))
        return END
    return "perception"


# ── 图构建入口 ────────────────────────────────────────────────────

def build_graph(worker: "LangGraphWorker"):
    """构建并编译瀑布流主工作流图。"""
    graph = StateGraph(AgentState)

    graph.add_node("perception", _make_perception_node(worker))
    graph.add_node("cognition",  _make_cognition_node(worker))
    graph.add_node("execute",    _make_execute_node(worker))
    graph.add_node("validate",   _make_validate_node(worker))
    graph.add_node("check",      _make_check_node(worker))

    graph.set_entry_point("perception")
    graph.add_edge("perception", "cognition")
    graph.add_edge("cognition",  "execute")
    graph.add_edge("execute",    "validate")
    graph.add_edge("validate",   "check")
    graph.add_conditional_edges("check", _should_continue, {"perception": "perception", END: END})

    return graph.compile()
