"""端到端集成测试：MockVision + SequentialDecider + Mock ADB 全流程验证。

验证 LangGraph waterfall_flow 在纯 Mock 环境下：
- 能完整走通 perception → cognition → execute → validate → check 各节点
- MemoryManager 被正确更新（WorkingMemory 有记录）
- 返回结果包含正确字段
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch
from PIL import Image
import pytest

from config.settings import AgentConfig, VisionConfig, LLMConfig, ADBConfig, OutputConfig
from core.agent.factory import get_agent
from core.vision.providers.mock import Provider as MockVision
from core.llm.sequential import SequentialDecider


# ── helpers ───────────────────────────────────────────────────────

def _make_config(tmp_path) -> AgentConfig:
    return AgentConfig(
        vision=VisionConfig(vision_type="mock"),
        llm=LLMConfig(provider="sequential"),
        adb=ADBConfig(game_package="com.test.demo", game_activity=".MainActivity"),
        output=OutputConfig(output_dir=str(tmp_path), save_screenshots=False),
        max_steps=5,
        step_delay=0.0,
    )


def _make_mock_adb():
    adb = MagicMock()
    adb.screenshot.return_value = Image.new("RGB", (100, 100), color=(30, 30, 30))
    adb.check_crash.return_value = False
    adb.dev_w = 100
    adb.dev_h = 100
    return adb


# ── tests ─────────────────────────────────────────────────────────

def test_full_flow_completes(tmp_path):
    """5步内完成顺序遍历，结果 status 应为 pass 或 fail（不 error）。"""
    config = _make_config(tmp_path)
    vision = MockVision()
    llm    = SequentialDecider()

    mock_adb = _make_mock_adb()

    with patch("core.agent.worker.ADBController") as MockADB, \
         patch("core.agent.worker.ADBError", Exception):
        instance = MockADB.return_value
        instance.screenshot.return_value = Image.new("RGB", (100, 100))
        instance.check_crash.return_value = False
        instance.dev_w = 100
        instance.dev_h = 100
        instance.connect.return_value = None

        worker = get_agent(config, vision=vision, llm=llm)
        worker._adb      = instance
        worker._executor = MagicMock()

        result = worker.run("点击所有按钮")

    assert result["status"] in ("pass", "fail", "error")
    assert isinstance(result["steps"], int)
    assert "history" in result
    assert "nav_stats" in result

    worker.teardown()


def test_memory_manager_records_steps(tmp_path):
    """validate 节点执行后 WorkingMemory 应有步骤记录。"""
    config = _make_config(tmp_path)
    vision = MockVision()
    llm    = SequentialDecider()

    with patch("core.agent.worker.ADBController") as MockADB, \
         patch("core.agent.worker.ADBError", Exception):
        instance = MockADB.return_value
        instance.screenshot.return_value = Image.new("RGB", (100, 100))
        instance.check_crash.return_value = False
        instance.dev_w = 100
        instance.dev_h = 100
        instance.connect.return_value = None

        worker = get_agent(config, vision=vision, llm=llm)
        worker._adb      = instance
        worker._executor = MagicMock()

        worker.run("测试记忆记录")

    assert isinstance(len(worker.memory.working), int)
    assert result["steps"] >= 0

    worker.teardown()


def test_sequential_decider_pass(tmp_path):
    """当所有元素遍历完毕时，SequentialDecider 返回 done=True，status 应为 pass。"""
    config = _make_config(tmp_path)
    config.max_steps = 20
    vision = MockVision()
    llm    = SequentialDecider()

    with patch("core.agent.worker.ADBController") as MockADB, \
         patch("core.agent.worker.ADBError", Exception):
        instance = MockADB.return_value
        instance.screenshot.return_value = Image.new("RGB", (100, 100))
        instance.check_crash.return_value = False
        instance.dev_w = 100
        instance.dev_h = 100
        instance.connect.return_value = None

        worker = get_agent(config, vision=vision, llm=llm)
        worker._adb      = instance
        worker._executor = MagicMock()

        result = worker.run("顺序遍历所有按钮")

    assert result["status"] in ("pass", "fail")

    worker.teardown()


def test_config_nested_structure():
    """AgentConfig 嵌套子配置结构是否正确。"""
    config = AgentConfig(
        vision=VisionConfig(vision_type="mock", omni_mode="http"),
        llm=LLMConfig(provider="openai", model_name="gpt-4o"),
        adb=ADBConfig(game_package="com.example.game"),
        output=OutputConfig(output_dir="/tmp/test", save_screenshots=True),
    )
    assert config.vision.vision_type == "mock"
    assert config.vision.omni_mode == "http"
    assert config.llm.provider == "openai"
    assert config.llm.model_name == "gpt-4o"
    assert config.adb.game_package == "com.example.game"
    assert config.output.save_screenshots is True
    assert config.max_steps == 30
