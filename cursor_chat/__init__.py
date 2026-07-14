"""cursor_chat — 基于 Cursor 官方 Python SDK 的聊天模块。"""

from cursor_chat.chat.service import ChatResponse, CursorChatService
from cursor_chat.config import ChatConfig, load_chat_config
from cursor_chat.exceptions import (
    ChatConfigError,
    ChatError,
    ChatRunError,
    ChatStartupError,
)
from cursor_chat.knowledge.base import KnowledgeRetriever, RetrievedChunk
from cursor_chat.knowledge.local_retriever import LocalDirectoryRetriever

__all__ = [
    "ChatConfig",
    "ChatConfigError",
    "ChatError",
    "ChatResponse",
    "ChatRunError",
    "ChatStartupError",
    "CursorChatService",
    "KnowledgeRetriever",
    "LocalDirectoryRetriever",
    "RetrievedChunk",
    "load_chat_config",
]

__version__ = "0.1.0"
