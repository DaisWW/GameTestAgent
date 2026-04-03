"""Unit tests for the three-layer memory system."""
from __future__ import annotations

import pytest

from core.memory.working_memory import WorkingMemory, MemoryStep
from core.memory.nav_graph import NavigationGraph
from core.memory.experience_pool import ExperiencePool
from core.types import ActionType, BugTag, ElementType


# ── WorkingMemory ─────────────────────────────────────────────

def _make_step(step: int, action: str = ActionType.TAP, page_hash: str = "aaaa",
               result_hash: str = "bbbb", success: bool = True) -> MemoryStep:
    return MemoryStep(
        step=step, action=action, params={"id": step},
        reasoning="test", page_hash=page_hash,
        result_hash=result_hash, success=success, element_label="btn",
    )


def test_working_memory_capacity():
    wm = WorkingMemory(capacity=3)
    for i in range(5):
        wm.push(_make_step(i))
    assert len(wm) == 3


def test_working_memory_aba_detection():
    wm = WorkingMemory()
    for _ in range(3):
        wm.push(_make_step(0, page_hash="A", result_hash="B"))
        wm.push(_make_step(1, page_hash="B", result_hash="A"))
    assert wm.detect_aba_loop() is not None


def test_working_memory_stale_click():
    wm = WorkingMemory()
    for i in range(3):
        wm.push(_make_step(i, success=False))
    assert wm.detect_stale_click() is not None


# ── NavigationGraph ───────────────────────────────────────────

def test_nav_graph_register_and_visit():
    ng = NavigationGraph()
    elements = [{"id": 1, "bbox": [0,0,100,50], "label": "btn1", "type": ElementType.BUTTON},
                {"id": 2, "bbox": [0,60,100,110], "label": "btn2", "type": ElementType.BUTTON}]
    ng.register_page("page_A", elements=elements)
    ng.mark_visited("page_A", 1)
    assert ng.get_visited_ids("page_A") == [1]
    assert ng.get_unvisited_ids("page_A", [1, 2]) == [2]


def test_nav_graph_transition():
    ng = NavigationGraph()
    ng.register_page("page_A")
    ng.register_page("page_B")
    ng.add_transition("page_A", "page_B", action=ActionType.TAP, element_id=1)
    stats = ng.stats()
    assert stats.pages == 2
    assert stats.transitions == 1


def test_nav_graph_aba_loop_detection():
    """Regression: [A,B,A,B] was incorrectly returning False before fix."""
    ng = NavigationGraph()
    assert ng.is_in_aba_loop(["A", "B", "A", "B"]) is True
    assert ng.is_in_aba_loop(["A", "B", "C", "D"]) is False
    assert ng.is_in_aba_loop(["A", "A", "A", "A"]) is False  # same page, not ABA
    assert ng.is_in_aba_loop(["A", "B", "A"]) is False        # too short


def test_nav_graph_save_load(tmp_path):
    path = str(tmp_path / "graph.json")
    ng = NavigationGraph(graph_path=path)
    ng.register_page("p1")
    ng.register_page("p2")
    ng.add_transition("p1", "p2", action=ActionType.TAP, element_id=0)
    ng.mark_visited("p1", 0)
    ng.save_json()

    ng2 = NavigationGraph(graph_path=path)
    assert ng2.stats().pages == 2
    assert ng2.stats().transitions == 1
    assert ng2.get_visited_ids("p1") == []  # visited_map 不跨 run 持久化


# ── ExperiencePool ────────────────────────────────────────────

def test_experience_pool_in_memory():
    ep = ExperiencePool(db_path=":memory:")
    rid = ep.save_successful_path("test task", [{"step": 1}])
    assert rid > 0
    results = ep.query_similar_paths("test")
    assert len(results) == 1


def test_experience_pool_bug_snapshot():
    ep = ExperiencePool(db_path=":memory:")
    ep.save_bug("deadbeef", "游戏崩溃", tags=[BugTag.CRASH])
    desc = ep.is_known_bug_page("deadbeef")
    assert desc == "游戏崩溃"
