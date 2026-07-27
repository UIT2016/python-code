"""Cursor Agent 会话封装。"""

from __future__ import annotations

from typing import Any, Iterator, Optional

from cursor_chat.chat.client_helper import close_agent, create_agent, create_client_context
from cursor_chat.chat.streaming import collect_assistant_text, iter_assistant_text, wait_run_result
from cursor_chat.config import ChatConfig
from cursor_chat.exceptions import ChatStartupError
from cursor_chat.knowledge.base import KnowledgeRetriever
from cursor_chat.prompt_builder import build_user_message, format_knowledge_context


class ChatSession:
    """维护单个 Cursor Agent 会话，多轮 send 自动保留官方 SDK 上下文。"""

    def __init__(self, config: ChatConfig):
        self.config = config
        self.system_prompt: Optional[str] = config.system_prompt
        self._client_ctx = create_client_context(config)
        self._client: Any | None = None
        self._agent: Any | None = None
        self._knowledge: KnowledgeRetriever | None = None

    @property
    def agent_id(self) -> Optional[str]:
        if self._agent is None:
            return None
        return getattr(self._agent, "agent_id", None)

    def attach_knowledge(self, retriever: KnowledgeRetriever | None) -> None:
        self._knowledge = retriever

    def set_system_prompt(self, prompt: Optional[str]) -> None:
        self.system_prompt = prompt

    def new_session(self) -> str:
        """关闭当前 Agent 并创建新会话（清空官方 SDK 侧对话历史）。"""
        self._close_agent()
        agent = self._ensure_agent()
        return getattr(agent, "agent_id", "") or ""

    def clear_history(self) -> str:
        """清空历史等价于创建新 Agent 会话。"""
        return self.new_session()

    def chat(self, message: str, *, stream: bool = False, use_knowledge: bool = False) -> str | Iterator[str]:
        agent = self._ensure_agent()
        prompt = self._build_prompt(message, use_knowledge=use_knowledge)
        run = self._send(agent, prompt)
        if stream:
            return self._stream_response(run)
        return self._blocking_response(run)

    def close(self) -> None:
        self._close_agent()
        if self._client_ctx is not None:
            self._client_ctx.__exit__(None, None, None)
            self._client_ctx = None
            self._client = None

    def _ensure_agent(self) -> Any:
        if self._agent is not None:
            return self._agent
        if self._client_ctx is not None and self._client is None:
            self._client = self._client_ctx.__enter__()
        self._agent = create_agent(self.config, client=self._client)
        # Cloud REST Agent 在首次 chat 时才真正创建远端 agent_id
        from cursor_chat.chat.cloud_rest import CloudRestAgent

        if isinstance(self._agent, CloudRestAgent):
            return self._agent
        agent_id = getattr(self._agent, "agent_id", None)
        if not agent_id:
            raise ChatStartupError("Agent 创建失败：未返回 agent_id")
        return self._agent

    def _close_agent(self) -> None:
        close_agent(self._agent)
        self._agent = None

    def _build_prompt(self, message: str, *, use_knowledge: bool) -> str:
        knowledge_context = None
        if use_knowledge and self._knowledge is not None:
            chunks = self._knowledge.retrieve(message, top_k=self.config.knowledge.top_k)
            knowledge_context = format_knowledge_context(chunks)
        return build_user_message(
            message,
            system_prompt=self.system_prompt,
            knowledge_context=knowledge_context,
        )

    def _send(self, agent: Any, prompt: str) -> Any:
        try:
            return agent.send(prompt)
        except Exception as exc:
            from cursor_chat.chat.client_helper import _raise_startup_error

            _raise_startup_error(exc)

    def _blocking_response(self, run: Any) -> str:
        text = collect_assistant_text(run)
        wait_run_result(run)
        return text

    def _stream_response(self, run: Any) -> Iterator[str]:
        def _generator() -> Iterator[str]:
            for chunk in iter_assistant_text(run):
                yield chunk
            wait_run_result(run)

        return _generator()

    def __enter__(self) -> "ChatSession":
        self._ensure_agent()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
