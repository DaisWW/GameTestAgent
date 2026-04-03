"""Unit tests for LLMAdapter using a mock LangChain model."""
from __future__ import annotations

from unittest.mock import MagicMock, patch
from PIL import Image

from core.llm.adapter import LLMAdapter
from core.context.protocol import ContextPacket


def _make_packet() -> ContextPacket:
    img = Image.new("RGB", (100, 100), color=(0, 0, 0))
    return ContextPacket(
        screenshot=img,
        task_goal="test task",
        current_path="主界面",
        action_history=[],
        current_observation={"page_hash": "abc123", "omni_boxes": [], "unvisited_ids": [], "visited_ids": [], "visit_count": 1, "total_elements": 0},
        anomaly_flag="",
        relevant_experience=[],
        step=1,
        max_steps=30,
    )


def test_adapter_returns_action():
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content='{"action": "tap", "params": {"id": 1}, "reasoning": "ok", "done": false, "result": ""}')

    with patch("core.llm.adapter._load_system_prompt", return_value="You are a test agent."):
        adapter = LLMAdapter(llm=mock_llm)

    decision = adapter.ask(_make_packet())
    assert decision["action"] == "tap"


def test_adapter_fallback_on_error():
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = RuntimeError("network error")

    with patch("core.llm.adapter._load_system_prompt", return_value=""):
        adapter = LLMAdapter(llm=mock_llm, max_retries=0)

    decision = adapter.ask(_make_packet())
    assert decision["action"] == "wait"
