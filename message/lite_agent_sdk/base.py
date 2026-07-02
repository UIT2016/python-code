from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from lite_agent.client import LLMConfig, OpenAIClient


@dataclass
class MessageBatch:
    batch_id: int
    text: str
    message_count: int
    room_titles: List[str] = field(default_factory=list)


@dataclass
class ExtractedItem:
    term: str
    category: str = "keyword"
    count: int = 1
    sentiment: str = ""
    note: str = ""
    rooms: List[str] = field(default_factory=list)


@dataclass
class SkillContext:
    date_str: str
    source_data: Dict[str, Any]
    batch_size: int = 8000
    batches: List[MessageBatch] = field(default_factory=list)
    extract_results: List[List[ExtractedItem]] = field(default_factory=list)
    aggregated: Dict[str, ExtractedItem] = field(default_factory=dict)
    hot_terms: List[ExtractedItem] = field(default_factory=list)
    report: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


class BaseSkill(ABC):
    name: str = "BaseSkill"

    @abstractmethod
    async def execute(self, ctx: SkillContext) -> None:
        raise NotImplementedError


def create_llm_client(
    *,
    model: str,
    api_key: str,
    api_base: str,
    temperature: float = 0.3,
) -> OpenAIClient:
    """创建 OpenAI 兼容 LLM 客户端（DeepSeek / Qwen 等）。"""
    return OpenAIClient(
        model=model,
        api_key=api_key,
        api_base=api_base,
        llm_config=LLMConfig(temperature=temperature),
    )


def create_llm_client_from_cfg(cfg: Dict[str, Any], provider: Optional[str] = None, *, temperature: float = 0.3) -> OpenAIClient:
    """根据 config_loader 返回的配置创建 LLM 客户端。"""
    name = (provider or cfg.get("llm_provider") or "deepseek").strip().lower()
    block = (cfg.get("llm") or {}).get(name)
    if not block:
        block = {
            "api_key": cfg["api_key"],
            "api_url": cfg["api_url"],
            "model": cfg["model"],
        }
    return create_llm_client(
        model=block["model"],
        api_key=block["api_key"],
        api_base=block["api_url"],
        temperature=temperature,
    )


async def llm_chat(client: OpenAIClient, system: str, user: str) -> str:
    """纯文本对话，绕过 lite_agent.completion 默认的 tool_choice（DeepSeek/Qwen 无 tools 时会 400）。"""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    params: Dict[str, Any] = {
        "model": client.model,
        "messages": messages,
        "stream": False,
    }
    llm_config = client.llm_config
    if llm_config.temperature is not None:
        params["temperature"] = llm_config.temperature
    if llm_config.max_tokens is not None:
        params["max_tokens"] = llm_config.max_tokens
    if llm_config.top_p is not None:
        params["top_p"] = llm_config.top_p
    if llm_config.frequency_penalty is not None:
        params["frequency_penalty"] = llm_config.frequency_penalty
    if llm_config.presence_penalty is not None:
        params["presence_penalty"] = llm_config.presence_penalty
    if llm_config.stop is not None:
        params["stop"] = llm_config.stop
    resp = await client._client.chat.completions.create(**params)
    content = resp.choices[0].message.content
    return content or ""
