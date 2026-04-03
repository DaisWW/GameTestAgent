from __future__ import annotations

"""LangGraph 瀑布流主工作流 —— 只负责图的连线。

节点逻辑分别在 workflows/nodes/ 下的独立文件中。
"""

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

from workflows.nodes import perception, cognition, execute, validate, check

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
    run_dir:        str
    memory_dir:     str


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

    graph.add_node("perception", perception(worker))
    graph.add_node("cognition",  cognition(worker))
    graph.add_node("execute",    execute(worker))
    graph.add_node("validate",   validate(worker))
    graph.add_node("check",      check(worker))

    graph.set_entry_point("perception")
    graph.add_edge("perception", "cognition")
    graph.add_edge("cognition",  "execute")
    graph.add_edge("execute",    "validate")
    graph.add_edge("validate",   "check")
    graph.add_conditional_edges("check", _should_continue, {"perception": "perception", END: END})

    return graph.compile()
