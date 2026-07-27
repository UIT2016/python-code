#!/usr/bin/env python3
"""示例：基础单次聊天。"""

from __future__ import annotations

from _common import setup_path

setup_path()

from cursor_chat import ChatStartupError, CursorChatService, load_chat_config


def main() -> None:
    config = load_chat_config()
    config.system_prompt = "你是一个简洁助手，用中文回答。"

    try:
        with CursorChatService(config) as service:
            response = service.chat("用一句话介绍 Python 的 asyncio。")
            print("Agent ID:", service.agent_id)
            print("回复:\n", response.text)
    except ChatStartupError as exc:
        print(f"[启动失败] {exc}")
        if exc.is_retryable:
            print("该错误可重试。")
    except Exception as exc:
        print(f"[错误] {exc}")


if __name__ == "__main__":
    main()
