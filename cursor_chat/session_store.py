"""进程内会话注册表：session_id -> CursorChatService。"""

from __future__ import annotations

import atexit
import threading
import uuid
from typing import Dict, Optional

from cursor_chat.chat.service import CursorChatService
from cursor_chat.config import ChatConfig, load_chat_config
from cursor_chat.exceptions import ChatConfigError


class SessionStore:
    """线程安全的内存会话管理。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: Dict[str, CursorChatService] = {}
        self._base_config: Optional[ChatConfig] = None
        atexit.register(self.close_all)

    def _config(self) -> ChatConfig:
        if self._base_config is None:
            self._base_config = load_chat_config()
        # 每会话独立拷贝 model/system_prompt 等可变字段
        from cursor_chat.config import KnowledgeConfig

        cfg = self._base_config
        return ChatConfig(
            api_key=cfg.api_key,
            model=cfg.model,
            cwd=cfg.cwd,
            mode=cfg.mode,
            runtime=cfg.runtime,
            base_url=cfg.base_url,
            auth_token=cfg.auth_token,
            timeout=cfg.timeout,
            max_retries=cfg.max_retries,
            system_prompt=cfg.system_prompt,
            knowledge=KnowledgeConfig(
                directories=list(cfg.knowledge.directories),
                top_k=cfg.knowledge.top_k,
                max_chunk_chars=cfg.knowledge.max_chunk_chars,
            ),
        )

    def create(self, mode: str | None = None) -> tuple[str, CursorChatService]:
        config = self._config()
        service = CursorChatService(config)
        if mode:
            from cursor_chat.prompts.presets import get_preset

            preset = get_preset(mode)
            service.task_mode = preset.id
            service.set_system_prompt(preset.system_prompt)
        service.new_session()
        session_id = uuid.uuid4().hex
        with self._lock:
            self._sessions[session_id] = service
        return session_id, service

    def get(self, session_id: str) -> Optional[CursorChatService]:
        with self._lock:
            return self._sessions.get(session_id)

    def remove(self, session_id: str) -> None:
        with self._lock:
            service = self._sessions.pop(session_id, None)
        if service is not None:
            service.close()

    def close_all(self) -> None:
        with self._lock:
            services = list(self._sessions.values())
            self._sessions.clear()
        for service in services:
            try:
                service.close()
            except Exception:
                pass

    def health(self) -> dict:
        try:
            cfg = self._config()
            has_key = bool(cfg.api_key)
            return {
                "ok": True,
                "model": cfg.model,
                "runtime": cfg.runtime,
                "has_api_key": has_key,
                "session_count": len(self._sessions),
                "cwd": cfg.cwd,
            }
        except ChatConfigError as exc:
            return {"ok": False, "error": str(exc), "session_count": len(self._sessions)}


session_store = SessionStore()
