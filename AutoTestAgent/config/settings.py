from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

_ROOT = Path(__file__).parent.parent


def load_config(env_file: Optional[str] = None) -> "AgentConfig":
    """从 .env 文件加载配置，env_file 默认为项目根目录的 .env。"""
    target = env_file or str(_ROOT / ".env")
    load_dotenv(target, override=False)
    return AgentConfig(
        # ── Vision ──────────────────────────────────────────────────
        vision_type=os.getenv("VISION_TYPE", "omni_v2"),
        omni_endpoint=os.getenv("OMNI_ENDPOINT", "http://127.0.0.1:7861"),
        omni_timeout=int(os.getenv("OMNI_TIMEOUT", "30")),
        omni_model_path=os.getenv("OMNI_MODEL_PATH", "scripts/OmniParser/weights"),
        # ── LLM ─────────────────────────────────────────────────────
        llm_provider=os.getenv("LLM_PROVIDER", "openai"),
        api_key=os.getenv("LLM_API_KEY", ""),
        api_base=os.getenv("LLM_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        model_name=os.getenv("LLM_MODEL", "qwen-vl-max"),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1024")),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
        llm_template_key=os.getenv("LLM_TEMPLATE_KEY", "default"),
        # ── Agent Loop ──────────────────────────────────────────────
        max_steps=int(os.getenv("MAX_STEPS", "30")),
        step_delay=float(os.getenv("STEP_DELAY", "1.0")),
        # ── ADB ─────────────────────────────────────────────────────
        adb_serial=os.getenv("ADB_SERIAL", ""),
        game_package=os.getenv("GAME_PACKAGE", ""),
        game_activity=os.getenv("GAME_ACTIVITY", ""),
        game_launch_wait=float(os.getenv("GAME_LAUNCH_WAIT", "5.0")),
        # ── Output ──────────────────────────────────────────────────
        output_dir=os.getenv("OUTPUT_DIR", ""),
        save_screenshots=os.getenv("SAVE_SCREENSHOTS", "true").lower() == "true",
    )


@dataclass
class AgentConfig:
    # ── Vision provider ─────────────────────────────────────────────
    vision_type: str = "omni_v2"
    """可选值: omni_v2 | mock | paddle_ocr (未来扩展)"""
    omni_endpoint: str = "http://127.0.0.1:7861"
    """OmniParser V2 服务地址（HTTP 模式）"""
    omni_timeout: int = 30
    """HTTP 请求超时（秒）"""
    omni_model_path: str = "scripts/OmniParser/weights"
    """Local 模式下的模型权重目录（icon_caption / icon_detect 的父目录）"""

    # ── LLM provider ────────────────────────────────────────────────
    llm_provider: str = "openai"
    """可选值: openai | anthropic | google
    openai 兼容所有 OpenAI-format API（Qwen / DeepSeek / Moonshot / Ollama 等）"""
    api_key: str = ""
    api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model_name: str = "qwen-vl-max"
    max_tokens: int = 1024
    temperature: float = 0.1
    llm_template_key: str = "default"
    """prompt_template.json 中的模板键名"""

    # ── Agent loop ──────────────────────────────────────────────────
    max_steps: int = 30
    step_delay: float = 1.0
    """每步执行后等待时间（秒），让 UI 稳定"""

    # ── ADB ─────────────────────────────────────────────────────────
    adb_serial: str = ""
    game_package: str = ""
    game_activity: str = ""
    game_launch_wait: float = 5.0

    # ── Output ──────────────────────────────────────────────────────
    output_dir: str = ""
    save_screenshots: bool = True

    # ── 额外透传参数（供子类或新 provider 使用）────────────────────
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.output_dir:
            if self.game_package:
                self.output_dir = str(_ROOT / "data" / self.game_package)
            else:
                self.output_dir = str(_ROOT / "data")
