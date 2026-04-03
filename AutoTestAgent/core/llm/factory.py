from __future__ import annotations

"""LLM 工厂：根据 llm_provider 动态创建 LangChain BaseChatModel。

支持的 provider：
    openai    — ChatOpenAI，兼容所有 OpenAI-format API
                （Qwen / DeepSeek / Moonshot / 本地 Ollama 等只需改 base_url）
    anthropic — ChatAnthropic（Claude 系列）
    google    — ChatGoogleGenerativeAI（Gemini 系列）

新增 provider 只需修改本文件，无需改动其他任何代码：
    1. 在此处添加一个 _make_<provider> 函数
    2. 将其注册到 _MAKERS 字典
    3. 在 .env 中设置 LLM_PROVIDER=<provider>
"""

import logging
from typing import TYPE_CHECKING

from langchain_core.language_models import BaseChatModel

if TYPE_CHECKING:
    from config.settings import AgentConfig

logger = logging.getLogger(__name__)


def create_llm(config: "AgentConfig") -> BaseChatModel:
    """根据 config.llm_provider 创建对应的 LangChain 模型实例。

    Returns:
        实现了 BaseChatModel 接口的 LLM 实例，可直接 .invoke() / .stream()。
    """
    provider = config.llm_provider
    maker = _MAKERS.get(provider)
    if maker is None:
        raise ValueError(
            f"不支持的 llm_provider: {provider!r}，"
            f"可选值: {list(_MAKERS)}"
        )
    llm = maker(config)
    logger.info("LLM 已创建: provider=%s model=%s", provider, config.model_name)
    return llm


# ── 各 provider 构造函数 ─────────────────────────────────────────


def _make_openai(config: "AgentConfig") -> BaseChatModel:
    """OpenAI 兼容接口（含 Qwen / DeepSeek / Moonshot / Ollama 等）。

    只需在 .env 中设置正确的 LLM_API_BASE 即可切换到任意兼容后端：
        Qwen:     https://dashscope.aliyuncs.com/compatible-mode/v1
        DeepSeek: https://api.deepseek.com
        Ollama:   http://localhost:11434/v1
    """
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        api_key=config.api_key or "dummy",
        base_url=config.api_base or None,
        model=config.model_name,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
    )


def _make_anthropic(config: "AgentConfig") -> BaseChatModel:
    """Anthropic Claude 系列（claude-3-5-sonnet 等）。"""
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(
        api_key=config.api_key,
        model=config.model_name,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
    )


def _make_google(config: "AgentConfig") -> BaseChatModel:
    """Google Gemini 系列（gemini-1.5-pro 等）。"""
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        google_api_key=config.api_key,
        model=config.model_name,
        max_output_tokens=config.max_tokens,
        temperature=config.temperature,
    )


# ── 注册表（provider name → 构造函数）────────────────────────────

_MAKERS = {
    "openai":    _make_openai,
    "anthropic": _make_anthropic,
    "google":    _make_google,
}
