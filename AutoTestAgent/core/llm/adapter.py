from __future__ import annotations

"""LLM 适配器：将 LangChain BaseChatModel 包装成统一的 ask(ContextPacket) 接口。

Chain 结构（LangChain Expression Language）：
    messages → BaseChatModel → JsonOutputParser → ActionDecision dict

设计要点：
- 任何实现了 BaseChatModel 的模型均可直接传入，无需改动此文件
- 图片以 base64 内嵌在 HumanMessage 中，兼容所有支持多模态的 provider
- 系统提示从 config/prompt_template.json 读取，与代码解耦
- 内置重试逻辑和 fallback 决策，防止单次 LLM 失败导致 Agent 崩溃
"""

import base64
import io
import json
import logging
from pathlib import Path
from typing import Any, Dict, TYPE_CHECKING

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser

from core.llm.base import BrainProvider

if TYPE_CHECKING:
    from core.context.protocol import ContextPacket

logger = logging.getLogger(__name__)

_TEMPLATE_PATH = Path(__file__).parent.parent.parent / "config" / "prompt_template.json"

_FALLBACK_ACTION: Dict[str, Any] = {
    "action":    "wait",
    "params":    {"seconds": 2},
    "reasoning": "LLM 调用失败，等待重试",
    "done":      False,
    "result":    "",
}


def _load_system_prompt(template_key: str) -> str:
    """从 config/prompt_template.json 加载系统提示词。"""
    try:
        templates = json.loads(_TEMPLATE_PATH.read_text(encoding="utf-8"))
        entry = templates.get(template_key) or templates.get("default", "")
        # 支持两种格式：纯字符串 或 {"role": "system", "content": "..."}
        if isinstance(entry, dict):
            return entry.get("content", "")
        return entry or ""
    except Exception as exc:
        logger.warning("加载 prompt_template.json 失败: %s，使用空系统提示", exc)
        return ""


def _encode_image(image) -> str:
    """将 PIL Image 编码为 base64 JPEG 字符串。"""
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


class LLMAdapter(BrainProvider):
    """统一 LLM 调用接口。

    通过 LangChain BaseChatModel 抽象层屏蔽所有 provider 差异，
    对上层（Worker / Workflow）始终暴露同一个 ask(ContextPacket) → dict 接口。

    切换模型只需在工厂函数中传入不同的 BaseChatModel 实例，此类零改动。

    Args:
        llm:          任意 LangChain BaseChatModel 实例。
        template_key: prompt_template.json 中的模板键名。
        max_retries:  LLM 调用失败时最大重试次数。
    """

    def __init__(
        self,
        llm: BaseChatModel,
        template_key: str = "default",
        max_retries: int = 2,
    ) -> None:
        self._llm           = llm
        self._system_prompt = _load_system_prompt(template_key)
        self._parser        = JsonOutputParser()
        self._max_retries   = max_retries

    # ── 公共接口 ─────────────────────────────────────────────────

    def ask(self, packet: "ContextPacket") -> Dict[str, Any]:
        """接收 ContextPacket，调用 LLM，返回 ActionDecision 字典。

        ActionDecision 格式::

            {
                "action":    "tap" | "swipe" | "input_text"
                             | "press_home" | "wait" | "done",
                "params":    {...},          # 动作参数，tap 时含 id
                "reasoning": "...",          # LLM 推理说明
                "done":      false,          # true 表示任务完成
                "result":    "pass" | ""     # done=true 时填写
            }
        """
        messages = self._build_messages(packet)

        for attempt in range(self._max_retries + 1):
            try:
                response = self._llm.invoke(messages)
                decision = self._parser.invoke(response)
                self._validate(decision)
                logger.debug(
                    "LLM 决策: action=%s params=%s reasoning=%s",
                    decision.get("action"), decision.get("params"),
                    str(decision.get("reasoning", ""))[:60],
                )
                return decision
            except Exception as exc:
                if attempt < self._max_retries:
                    logger.warning(
                        "LLM 调用失败（第 %d 次），重试: %s", attempt + 1, exc
                    )
                else:
                    logger.error("LLM 调用最终失败，使用 fallback: %s", exc)
                    return dict(_FALLBACK_ACTION)

    # ── 内部辅助 ─────────────────────────────────────────────────

    def _build_messages(self, packet: "ContextPacket") -> list:
        """构建 LangChain 消息列表（SystemMessage + 多模态 HumanMessage）。"""
        img_b64      = _encode_image(packet.screenshot)
        context_json = packet.to_json_context()

        human_content = [
            {
                "type":      "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
            },
            {
                "type": "text",
                "text": (
                    f"当前上下文：\n{context_json}\n\n"
                    "请严格按 JSON 格式输出操作决策，不要包含任何额外文字或 Markdown 代码块。"
                ),
            },
        ]

        messages = []
        if self._system_prompt:
            messages.append(SystemMessage(content=self._system_prompt))
        messages.append(HumanMessage(content=human_content))
        return messages

    @staticmethod
    def _validate(decision: Any) -> None:
        """简单校验 LLM 输出格式，不合规则抛出 ValueError 触发重试。"""
        if not isinstance(decision, dict):
            raise ValueError(f"LLM 输出不是 dict: {type(decision)}")
        if "action" not in decision:
            raise ValueError(f"LLM 输出缺少 'action' 字段: {decision}")

    def __repr__(self) -> str:
        model_id = getattr(self._llm, "model_name", None) or getattr(
            self._llm, "model", "unknown"
        )
        return f"LLMAdapter(llm={self._llm.__class__.__name__}, model={model_id})"
