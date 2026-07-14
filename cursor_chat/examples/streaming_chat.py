#!/usr/bin/env python3
"""示例：流式输出（官方 run.iter_text）。"""

from __future__ import annotations

import sys

from _common import setup_path

setup_path()

from cursor_chat import ChatRunError, ChatStartupError, CursorChatService, load_chat_config


def main() -> None:
    config = load_chat_config()
    config.system_prompt = "用中文流式回答，条理清晰。"

    prompt = "简要说明 Cursor Python SDK 中 Agent.create 与 agent.send 的关系。"

    try:
        with CursorChatService(config) as service:
            print(f"用户: {prompt}\n助手: ", end="", flush=True)
            stream = service.chat(prompt, stream=True)
            assert hasattr(stream, "__iter__")
            for chunk in stream:
                sys.stdout.write(chunk)
                sys.stdout.flush()
            print("\n")
    except ChatStartupError as exc:
        print(f"\n[启动失败] {exc}")
    except ChatRunError as exc:
        print(f"\n[运行失败] run_id={exc.run_id}: {exc}")


if __name__ == "__main__":
    main()
