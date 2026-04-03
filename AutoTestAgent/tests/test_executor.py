"""Unit tests for ActionExecutor."""
from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from core.agent.executor import ActionExecutor


def _make_adb():
    adb = MagicMock()
    adb.tap = MagicMock()
    adb.swipe = MagicMock()
    adb.input_text = MagicMock()
    adb.press_home = MagicMock()
    return adb


def test_tap_dispatched():
    adb = _make_adb()
    ex = ActionExecutor(adb)
    ex.execute({"action": "tap", "_bbox": [0, 0, 200, 100], "params": {}})
    adb.tap.assert_called_once_with(100.0, 50.0)


def test_swipe_dispatched():
    adb = _make_adb()
    ex = ActionExecutor(adb)
    ex.execute({"action": "swipe", "_bbox": [0, 0, 200, 100], "params": {"direction": "up"}})
    adb.swipe.assert_called_once_with(100.0, 50.0, "up")


def test_input_text_dispatched():
    adb = _make_adb()
    ex = ActionExecutor(adb)
    ex.execute({"action": "input_text", "params": {"text": "hello"}})
    adb.input_text.assert_called_once_with("hello")

def test_unknown_action_logs_warning():
    adb = _make_adb()
    ex = ActionExecutor(adb)
    ex.execute({"action": "fly_to_moon", "params": {}})
    adb.tap.assert_not_called()


def test_tap_without_bbox_skips():
    adb = _make_adb()
    ex = ActionExecutor(adb)
    ex.execute({"action": "tap", "params": {}})
    adb.tap.assert_not_called()


def test_wait_sleeps(monkeypatch):
    adb = _make_adb()
    ex = ActionExecutor(adb)
    slept = []
    monkeypatch.setattr("core.agent.executor.time.sleep", lambda s: slept.append(s))
    ex.execute({"action": "wait", "params": {"seconds": 3}})
    assert slept == [3.0]
