from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

_ROOT = Path(__file__).parent.parent


def load_config(env_file: Optional[str] = None) -> "AgentConfig":
    """从 .env 文件加载配置，env_file 默认为项目根目录的 .env。"""
    target = env_file or str(_ROOT / ".env")
    load_dotenv(target, override=False)
    return AgentConfig(
        vision=VisionConfig(
            vision_type=os.getenv("VISION_TYPE", "omni_v2"),
            omni_endpoint=os.getenv("OMNI_ENDPOINT", "http://127.0.0.1:7861"),
            omni_timeout=int(os.getenv("OMNI_TIMEOUT", "30")),
        ),
        llm=LLMConfig(
            provider=os.getenv("LLM_PROVIDER", "openai"),
            api_key=os.getenv("LLM_API_KEY", ""),
            api_base=os.getenv("LLM_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            model_name=os.getenv("LLM_MODEL", "qwen-vl-max"),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1024")),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
            template_key=os.getenv("LLM_TEMPLATE_KEY", "default"),
        ),
        adb=ADBConfig(
            serial=os.getenv("ADB_SERIAL", ""),
            game_package=os.getenv("GAME_PACKAGE", ""),
            game_activity=os.getenv("GAME_ACTIVITY", ""),
            game_launch_wait=float(os.getenv("GAME_LAUNCH_WAIT", "5.0")),
        ),
        output=OutputConfig(
            output_dir=os.getenv("OUTPUT_DIR", ""),
            save_screenshots=os.getenv("SAVE_SCREENSHOTS", "true").lower() == "true",
        ),
        max_steps=int(os.getenv("MAX_STEPS", "30")),
        step_delay=float(os.getenv("STEP_DELAY", "1.0")),
    )


# ── 子配置 dataclass ─────────────────────────────────────────────


@dataclass
class VisionConfig:
    """视觉层配置。"""

    vision_type: str = "omni_v2"
    """可选值: omni_v2 | mock | paddle_ocr (未来扩展)"""
    omni_endpoint: str = "http://127.0.0.1:7861"
    """OmniParser 服务地址"""
    omni_timeout: int = 30
    """HTTP 请求超时（秒）"""


@dataclass
class LLMConfig:
    """LLM 层配置。"""

    provider: str = "openai"
    """可选值: openai | anthropic | google | sequential
    openai 兼容所有 OpenAI-format API（Qwen / DeepSeek / Moonshot / Ollama 等）"""
    api_key: str = ""
    api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model_name: str = "qwen-vl-max"
    max_tokens: int = 1024
    temperature: float = 0.1
    template_key: str = "default"
    """prompt_template.json 中的模板键名"""


@dataclass
class ADBConfig:
    """ADB 设备配置。"""

    serial: str = ""
    game_package: str = ""
    game_activity: str = ""
    game_launch_wait: float = 5.0


@dataclass
class OutputConfig:
    """输出路径配置。"""

    output_dir: str = ""
    save_screenshots: bool = True


# ── 顶层配置 ─────────────────────────────────────────────────────


@dataclass
class AgentConfig:
    """Agent 全局配置（嵌套子配置）。"""

    vision: VisionConfig = field(default_factory=VisionConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    adb: ADBConfig = field(default_factory=ADBConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # ── Agent loop ──────────────────────────────────────────────────
    max_steps: int = 30
    step_delay: float = 1.0
    """每步执行后等待时间（秒），让 UI 稳定"""

    # ── 额外透传参数（供子类或新 provider 使用）────────────────────
    extra: Dict[str, Any] = field(default_factory=dict)

    run_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))

    def __post_init__(self) -> None:
        if not self.output.output_dir:
            if self.adb.game_package:
                self.output.output_dir = str(_ROOT / "data" / self.adb.game_package)
            else:
                self.output.output_dir = str(_ROOT / "data")

    @property
    def memory_dir(self) -> str:
        return str(Path(self.output.output_dir) / "memory")

    @property
    def run_dir(self) -> str:
        return str(Path(self.output.output_dir) / "runs" / self.run_id)

    # ── 向后兼容 shim（供 main.py / factory.py / worker.py 等使用）────
    @property
    def vision_type(self) -> str:       return self.vision.vision_type
    @vision_type.setter
    def vision_type(self, v: str):      self.vision.vision_type = v
    @property
    def omni_endpoint(self) -> str:     return self.vision.omni_endpoint
    @property
    def omni_timeout(self) -> int:      return self.vision.omni_timeout

    @property
    def llm_provider(self) -> str:      return self.llm.provider
    @llm_provider.setter
    def llm_provider(self, v: str):     self.llm.provider = v
    @property
    def api_key(self) -> str:           return self.llm.api_key
    @property
    def api_base(self) -> str:          return self.llm.api_base
    @property
    def model_name(self) -> str:        return self.llm.model_name
    @property
    def llm_template_key(self) -> str:  return self.llm.template_key

    @property
    def game_package(self) -> str:      return self.adb.game_package
    @property
    def game_activity(self) -> str:     return self.adb.game_activity
    @property
    def game_launch_wait(self) -> float: return self.adb.game_launch_wait
    @property
    def adb_serial(self) -> str:        return self.adb.serial

    @property
    def output_dir(self) -> str:        return self.output.output_dir
    @property
    def save_screenshots(self) -> bool: return self.output.save_screenshots
