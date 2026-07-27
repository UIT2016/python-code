"""聊天模块异常定义。"""

from __future__ import annotations

from typing import Optional


class ChatError(Exception):
    """聊天模块基础异常。"""


class ChatConfigError(ChatError):
    """配置缺失或无效。"""


class ChatStartupError(ChatError):
    """Agent / Run 启动失败（对应官方 CursorAgentError）。"""

    def __init__(self, message: str, *, is_retryable: bool = False, cause: Optional[BaseException] = None):
        super().__init__(message)
        self.is_retryable = is_retryable
        self.cause = cause


class ChatRunError(ChatError):
    """Run 已启动但执行失败（result.status == error）。"""

    def __init__(self, message: str, *, run_id: str = "", agent_id: str = ""):
        super().__init__(message)
        self.run_id = run_id
        self.agent_id = agent_id
