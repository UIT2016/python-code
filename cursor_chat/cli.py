#!/usr/bin/env python3
"""交互式 CLI 聊天：默认流式输出，支持斜杠命令。

用法（在仓库根目录）::

    python -m cursor_chat.cli
"""

from __future__ import annotations

import sys
from pathlib import Path

# 允许直接 `python cursor_chat/cli.py` 运行
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cursor_chat import (  # noqa: E402
    ChatConfigError,
    ChatError,
    ChatRunError,
    ChatStartupError,
    CursorChatService,
    load_chat_config,
)

HELP_TEXT = """\
命令:
  /new, /clear     清空历史，新建会话
  /system <文本>   设置 System Prompt（无参数则查看当前值）
  /kb on|off       开关知识库检索（默认 off）
  /model <id>      切换模型（会新建会话）
  /help            显示帮助
  /quit, /exit     退出
"""


def _print_banner(service: CursorChatService, use_kb: bool) -> None:
    print("cursor_chat CLI")
    print(f"  model:    {service.config.model}")
    print(f"  agent:    {service.agent_id or '(未创建)'}")
    print(f"  knowledge:{' on' if use_kb else ' off'}")
    print(HELP_TEXT)


def _handle_command(service: CursorChatService, line: str, use_kb: bool) -> tuple[bool, bool]:
    """处理斜杠命令。返回 (should_continue, use_kb)。should_continue=False 表示退出。"""
    parts = line.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd in ("/quit", "/exit", "/q"):
        return False, use_kb

    if cmd == "/help":
        print(HELP_TEXT)
        return True, use_kb

    if cmd in ("/new", "/clear"):
        try:
            agent_id = service.clear_history()
            print(f"已清空历史，新会话: {agent_id}")
        except ChatError as exc:
            print(f"[错误] {exc}")
        return True, use_kb

    if cmd == "/system":
        if not arg:
            current = service.get_system_prompt() or "(空)"
            print(f"当前 System Prompt:\n{current}")
        else:
            service.set_system_prompt(arg)
            print("已更新 System Prompt。")
        return True, use_kb

    if cmd == "/kb":
        flag = arg.lower()
        if flag in ("on", "1", "true", "yes"):
            use_kb = True
            print("知识库检索: on")
        elif flag in ("off", "0", "false", "no"):
            use_kb = False
            print("知识库检索: off")
        else:
            print("用法: /kb on|off")
        return True, use_kb

    if cmd == "/model":
        if not arg:
            print(f"当前模型: {service.config.model}")
            print("用法: /model <model_id>")
        else:
            try:
                service.set_model(arg)
                print(f"已切换模型: {arg}，新会话: {service.agent_id}")
            except ChatError as exc:
                print(f"[错误] {exc}")
        return True, use_kb

    print(f"未知命令: {cmd}，输入 /help 查看帮助。")
    return True, use_kb


def _stream_chat(service: CursorChatService, message: str, use_kb: bool) -> None:
    print("助手: ", end="", flush=True)
    try:
        stream = service.chat(message, stream=True, use_knowledge=use_kb)
        for chunk in stream:
            sys.stdout.write(chunk)
            sys.stdout.flush()
        print()
    except ChatStartupError as exc:
        print(f"\n[启动失败] {exc}")
        if exc.is_retryable:
            print("（该错误可重试）")
    except ChatRunError as exc:
        print(f"\n[运行失败] run_id={exc.run_id}: {exc}")
    except ChatError as exc:
        print(f"\n[错误] {exc}")


def main() -> int:
    try:
        config = load_chat_config()
    except ChatConfigError as exc:
        print(f"[配置错误] {exc}", file=sys.stderr)
        return 1

    use_kb = False
    try:
        with CursorChatService(config) as service:
            # 进入会话时预创建 Agent，便于展示 agent_id
            try:
                service.new_session()
            except ChatStartupError as exc:
                print(f"[启动失败] {exc}", file=sys.stderr)
                return 1

            _print_banner(service, use_kb)

            while True:
                try:
                    line = input("你: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n再见。")
                    break

                if not line:
                    continue

                if line.startswith("/"):
                    should_continue, use_kb = _handle_command(service, line, use_kb)
                    if not should_continue:
                        print("再见。")
                        break
                    continue

                _stream_chat(service, line, use_kb)

    except ChatError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
