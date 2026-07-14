"""Cursor SDK 客户端创建与 Agent 生命周期辅助。"""

from __future__ import annotations

import codecs
import os
import time
from contextlib import AbstractContextManager
from typing import Any, Mapping, Optional

from cursor_chat.config import ChatConfig
from cursor_chat.exceptions import ChatStartupError

_WIN_BRIDGE_PATCHED = False


def _patch_windows_bridge_discovery() -> None:
    """Windows 上 SDK 用 select() 监听子进程管道会触发 WinError 10038。

    在调用官方 Bridge.launch 前，将发现逻辑替换为轮询读取 stderr。
    """
    global _WIN_BRIDGE_PATCHED
    if os.name != "nt" or _WIN_BRIDGE_PATCHED:
        return
    try:
        import cursor_sdk._bridge as bridge_mod
    except ImportError:
        return

    parse_discovery_line = bridge_mod.parse_discovery_line
    CursorSDKError = __import__("cursor_sdk.errors", fromlist=["CursorSDKError"]).CursorSDKError

    def _read_discovery_windows(process: Any, timeout: float) -> Mapping[str, Any]:
        if process.stderr is None:
            raise CursorSDKError("Bridge process stderr is unavailable")
        stderr_fd = process.stderr.fileno()
        was_blocking = os.get_blocking(stderr_fd)
        os.set_blocking(stderr_fd, False)
        try:
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            deadline = time.monotonic() + timeout
            stderr_lines: list[str] = []
            pending = ""

            def drain_available() -> Mapping[str, Any] | None:
                nonlocal pending
                while True:
                    try:
                        chunk = os.read(stderr_fd, 8192)
                    except BlockingIOError:
                        return None
                    if not chunk:
                        final_text = decoder.decode(b"", final=True)
                        if final_text:
                            pending += final_text
                        if pending:
                            line = pending
                            pending = ""
                            stderr_lines.append(line)
                            return parse_discovery_line(line)
                        return None
                    pending += decoder.decode(chunk)
                    while "\n" in pending:
                        line, pending = pending.split("\n", 1)
                        line += "\n"
                        stderr_lines.append(line)
                        discovery = parse_discovery_line(line)
                        if discovery is not None:
                            return discovery

            while time.monotonic() < deadline:
                discovery = drain_available()
                if discovery is not None:
                    return discovery
                exit_code = process.poll()
                if exit_code is not None:
                    discovery = drain_available()
                    if discovery is not None:
                        return discovery
                    raise CursorSDKError(
                        "Bridge exited before discovery with status "
                        f"{exit_code}: " + "".join(stderr_lines) + pending
                    )
                time.sleep(0.05)
            raise CursorSDKError("Timed out waiting for bridge discovery")
        finally:
            os.set_blocking(stderr_fd, was_blocking)

    bridge_mod._read_discovery = _read_discovery_windows
    _WIN_BRIDGE_PATCHED = True


def create_client_context(config: ChatConfig) -> Optional[AbstractContextManager[Any]]:
    """若配置了 base_url，返回 CursorClient.connect 上下文；cloud REST 模式无需 client。"""
    if getattr(config, "runtime", "cloud") == "cloud":
        return None
    _patch_windows_bridge_discovery()
    try:
        from cursor_sdk import CursorClient
    except ImportError as exc:
        raise ChatStartupError(
            "未安装 cursor-sdk。请使用 Python 3.10+ 并执行: pip install -r cursor_chat/requirements.txt",
            cause=exc,
        ) from exc

    if config.base_url:
        return CursorClient.connect(
            base_url=config.base_url,
            auth_token=config.auth_token or "",
        )
    return None


def create_agent(config: ChatConfig, client: Any | None = None):
    """创建 Agent：cloud 走官方 REST，local/auto 走官方 Python SDK。"""
    runtime = getattr(config, "runtime", "cloud") or "cloud"

    if runtime == "cloud":
        from cursor_chat.chat.cloud_rest import CloudRestAgent

        return CloudRestAgent(config)

    if runtime == "auto":
        try:
            return _create_sdk_agent(config, client=client)
        except ChatStartupError as exc:
            msg = str(exc)
            if "502" in msg or "10038" in msg or "Bridge" in msg:
                from cursor_chat.chat.cloud_rest import CloudRestAgent

                return CloudRestAgent(config)
            raise

    return _create_sdk_agent(config, client=client)


def _create_sdk_agent(config: ChatConfig, client: Any | None = None):
    """创建官方 SDK Agent 实例（调用方负责 close）。"""
    _patch_windows_bridge_discovery()
    try:
        from cursor_sdk import Agent, AgentOptions, LocalAgentOptions
    except ImportError as exc:
        raise ChatStartupError(
            "未安装 cursor-sdk。请使用 Python 3.10+ 并执行: pip install -r cursor_chat/requirements.txt",
            cause=exc,
        ) from exc

    # mode 需放在 AgentOptions 中，不能作为 Agent.create 的独立关键字参数
    options = AgentOptions(
        model=config.model,
        api_key=config.api_key,
        local=LocalAgentOptions(cwd=config.cwd),
        mode=config.mode,
    )
    try:
        if client is not None:
            return client.agents.create(options)
        return Agent.create(options)
    except Exception as exc:
        _raise_startup_error(exc)


def close_agent(agent: Any | None) -> None:
    if agent is None:
        return
    close_fn = getattr(agent, "close", None)
    if callable(close_fn):
        close_fn()


def _raise_startup_error(exc: BaseException) -> None:
    try:
        from cursor_sdk import CursorAgentError
    except ImportError:
        raise ChatStartupError(str(exc), cause=exc) from exc

    if isinstance(exc, CursorAgentError):
        raise ChatStartupError(
            exc.message if hasattr(exc, "message") else str(exc),
            is_retryable=bool(getattr(exc, "is_retryable", False)),
            cause=exc,
        ) from exc
    raise ChatStartupError(str(exc), cause=exc) from exc
