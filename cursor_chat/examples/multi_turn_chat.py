#!/usr/bin/env python3
"""示例：多轮连续聊天（同一 Agent 自动保留上下文）。"""

from __future__ import annotations

from _common import setup_path

setup_path()

from cursor_chat import ChatRunError, ChatStartupError, CursorChatService, load_chat_config
from cursor_chat.exceptions import ChatError


def main() -> None:
    config = load_chat_config()
    config.system_prompt = "你是编程助手，回答要简短。"

    questions = [
        "什么是 REST API？",
        "它和 GraphQL 的主要区别是什么？",
        "刚才你提到的 REST，给一个最简单的 HTTP 示例。",
    ]

    try:
        with CursorChatService(config) as service:
            print(f"会话 Agent ID: {service.agent_id}\n")
            for idx, question in enumerate(questions, 1):
                print(f"--- 第 {idx} 轮 ---")
                print(f"用户: {question}")
                response = service.chat(question)
                print(f"助手: {response.text}\n")

            print("--- 清空历史，开始新会话 ---")
            new_id = service.clear_history()
            print(f"新 Agent ID: {new_id}")
            response = service.chat("我们之前聊过什么？")
            print(f"助手: {response.text}")
    except ChatStartupError as exc:
        print(f"[启动失败] {exc}")
    except ChatRunError as exc:
        print(f"[运行失败] run_id={exc.run_id}: {exc}")
    except ChatError as exc:
        print(f"[聊天错误] {exc}")


if __name__ == "__main__":
    main()
