"""python -m cursor_chat 入口。

默认启动 CLI；可选子命令::

    python -m cursor_chat          # CLI
    python -m cursor_chat cli
    python -m cursor_chat web      # Flask Web (5055)
    python -m cursor_chat app      # 同 web
"""

from __future__ import annotations

import sys


def main() -> int:
    cmd = (sys.argv[1] if len(sys.argv) > 1 else "cli").lower()

    if cmd in ("cli", "chat", "-i", "--cli"):
        from cursor_chat.cli import main as cli_main

        return cli_main()

    if cmd in ("web", "app", "server", "--web"):
        # 去掉子命令后再把剩余参数留给 Flask（通常无）
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        from cursor_chat.app import main as app_main

        app_main()
        return 0

    if cmd in ("-h", "--help", "help"):
        print(__doc__)
        return 0

    print(f"未知子命令: {cmd}\n", file=sys.stderr)
    print(__doc__, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
