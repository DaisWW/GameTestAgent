from __future__ import annotations

"""工厂函数：根据 AgentConfig 动态加载 Vision/Brain Provider 并组装 Agent。

设计原则：
- 所有 Provider 约定类名均为 ``Provider``，放在对应模块的顶层。
- 新增 Provider 只需在 providers/ 下新建文件，在 .env 中写上名字即可，无需改工厂。
"""

import importlib
import logging
from typing import Optional

from config.settings import AgentConfig
from core.llm.factory import create_llm
from core.llm.adapter import LLMAdapter
from core.vision.base import VisionProvider

from .worker import LangGraphWorker

logger = logging.getLogger(__name__)

_VISION_PKG = "core.vision.providers"


def _load_vision(config: AgentConfig) -> VisionProvider:
    module_name = f"{_VISION_PKG}.{config.vision_type}"
    logger.info("加载视觉 Provider: %s", module_name)
    try:
        mod = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ImportError(
            f"找不到视觉 Provider: {module_name}\n"
            f"请在 core/vision/providers/ 下创建 {config.vision_type}.py 并定义 Provider 类。"
        ) from exc

    cls = getattr(mod, "Provider", None)
    if cls is None:
        raise AttributeError(f"{module_name} 中未定义 'Provider' 类")

    # 优先调用 from_config(config)；旧 Provider 回退到 extra 字典传参
    if hasattr(cls, "from_config"):
        provider = cls.from_config(config)
    else:
        provider = cls(**config.extra.get("vision_kwargs", {}))

    logger.info("视觉 Provider 初始化完成: %r", provider)
    return provider


def _create_llm_adapter(config: AgentConfig):
    """使用 LangChain 工厂创建 LLMAdapter，或返回 SequentialDecider（无需 LLM）。

    LLM_PROVIDER=sequential 时跳过真实模型调用，按元素 ID 顺序遍历。
    """
    if config.llm_provider == "sequential":
        from core.llm.sequential import SequentialDecider
        decider = SequentialDecider()
        logger.info("LLM: SequentialDecider（顺序遍历，无需 API Key）")
        return decider

    llm = create_llm(config)
    adapter = LLMAdapter(
        llm=llm,
        template_key=config.llm_template_key,
    )
    logger.info("LLMAdapter 初始化完成: %r", adapter)
    return adapter


def get_agent(
    config: AgentConfig,
    vision: Optional[VisionProvider] = None,
    llm: Optional[LLMAdapter] = None,
) -> LangGraphWorker:
    """根据配置构建 LangGraphWorker。

    Args:
        config: AgentConfig 实例（来自 load_config()）。
        vision: 可选，直接传入 VisionProvider 实例（测试时注入 Mock）。
        llm:    可选，直接传入 LLMAdapter 实例（测试时注入 Mock）。

    Returns:
        已初始化的 LangGraphWorker，可直接调用 run()。
    """
    v           = vision or _load_vision(config)
    llm_adapter = llm    or _create_llm_adapter(config)
    worker = LangGraphWorker(vision=v, llm=llm_adapter, config=config)
    logger.info("Agent 组装完成: vision=%r, llm=%r", v, llm_adapter)
    return worker
