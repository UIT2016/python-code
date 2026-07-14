"""对外聊天服务 API。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

from cursor_chat.chat.session import ChatSession
from cursor_chat.config import ChatConfig, load_chat_config
from cursor_chat.knowledge.base import KnowledgeRetriever
from cursor_chat.knowledge.local_retriever import LocalDirectoryRetriever
from cursor_chat.prompts.presets import DEFAULT_MODE, get_preset


@dataclass
class ChatResponse:
    text: str
    agent_id: str
    run_id: str = ""


class CursorChatService:
    """基于 Cursor 官方 SDK 的聊天服务（业务层仅管理会话、配置与知识库）。"""

    def __init__(self, config: ChatConfig):
        self.config = config
        self._session = ChatSession(config)
        self._knowledge: KnowledgeRetriever | None = None
        self.task_mode: str = DEFAULT_MODE
        self._init_knowledge_from_config()
        # 用预设覆盖默认 system（创建会话后还可再改）
        preset = get_preset(self.task_mode)
        self.set_system_prompt(preset.system_prompt)

    def set_task_mode(self, mode_id: str, *, renew_session: bool = True) -> str:
        """切换任务模式，默认新建会话避免上下文串味。"""
        preset = get_preset(mode_id)
        self.task_mode = preset.id
        self.set_system_prompt(preset.system_prompt)
        if renew_session:
            return self.new_session()
        return self.agent_id or ""

    @classmethod
    def from_config_file(cls, base_dir: Optional[str] = None) -> "CursorChatService":
        from pathlib import Path

        config = load_chat_config(Path(base_dir) if base_dir else None)
        return cls(config)

    def _init_knowledge_from_config(self) -> None:
        kb = self.config.knowledge
        if kb.directories:
            retriever = LocalDirectoryRetriever(
                kb.directories,
                max_chunk_chars=kb.max_chunk_chars,
            )
            self.attach_knowledge(retriever)

    def attach_knowledge(self, retriever: KnowledgeRetriever | None) -> None:
        self._knowledge = retriever
        self._session.attach_knowledge(retriever)

    def refresh_knowledge(self) -> None:
        if self._knowledge is not None:
            self._knowledge.refresh()

    def set_system_prompt(self, prompt: Optional[str]) -> None:
        """设置 System Prompt（通过消息前缀注入）。"""
        self.config.system_prompt = prompt
        self._session.set_system_prompt(prompt)

    def get_system_prompt(self) -> Optional[str]:
        return self._session.system_prompt

    def set_model(self, model: str) -> None:
        """切换模型需新建会话（Agent 创建时绑定 model）。"""
        self.config.model = model
        self.new_session()

    def new_session(self) -> str:
        return self._session.new_session()

    def clear_history(self) -> str:
        return self._session.clear_history()

    @property
    def agent_id(self) -> Optional[str]:
        return self._session.agent_id

    def chat(
        self,
        message: str,
        *,
        stream: bool = False,
        use_knowledge: bool = False,
    ) -> ChatResponse | Iterator[str]:
        result = self._session.chat(message, stream=stream, use_knowledge=use_knowledge)
        if stream:
            return result
        assert isinstance(result, str)
        return ChatResponse(text=result, agent_id=self._session.agent_id or "")

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "CursorChatService":
        self._session.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
