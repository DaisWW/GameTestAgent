"""Unit tests for ActionExecutor."""
from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from core.agent.executor import ActionExecutor
from core.types import ActionType, SwipeDirection


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
    ex.execute({"action": ActionType.TAP, "_bbox": [0, 0, 200, 100], "params": {}})
    adb.tap.assert_called_once_with(100.0, 50.0)


def test_swipe_dispatched():
    adb = _make_adb()
    ex = ActionExecutor(adb)
    ex.execute({"action": ActionType.SWIPE, "_bbox": [0, 0, 200, 100], "params": {"direction": SwipeDirection.UP}})
    adb.swipe.assert_called_once_with(100.0, 50.0, SwipeDirection.UP)


def test_input_text_dispatched():
    adb = _make_adb()
    ex = ActionExecutor(adb)
    ex.execute({"action": ActionType.INPUT_TEXT, "params": {"text": "hello"}})
    adb.input_text.assert_called_once_with("hello")

def test_unknown_action_logs_warning():
    adb = _make_adb()
    ex = ActionExecutor(adb)
    ex.execute({"action": "fly_to_moon", "params": {}})
    adb.tap.assert_not_called()


def test_tap_without_bbox_taps_center():
    adb = _make_adb()
    ex = ActionExecutor(adb)
    ex.execute({"action": ActionType.TAP, "params": {}})
    adb.tap.assert_called_once_with(500.0, 500.0)


def test_tap_with_xy_params():
    adb = _make_adb()
    ex = ActionExecutor(adb)
    ex.execute({"action": ActionType.TAP, "params": {"x": 300, "y": 700}})
    adb.tap.assert_called_once_with(300.0, 700.0)


def test_wait_sleeps(monkeypatch):
    adb = _make_adb()
    ex = ActionExecutor(adb)
    slept = []
    monkeypatch.setattr("core.agent.executor.time.sleep", lambda s: slept.append(s))
    ex.execute({"action": ActionType.WAIT, "params": {"seconds": 3}})
    assert slept == [3.0]
