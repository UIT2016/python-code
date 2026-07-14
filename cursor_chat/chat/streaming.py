"""流式输出与 Run 结果处理。"""

from __future__ import annotations

from typing import Any, Iterator

from cursor_chat.exceptions import ChatRunError


def iter_assistant_text(run: Any) -> Iterator[str]:
    """使用官方 run.iter_text() 流式输出助手文本。"""
    if hasattr(run, "iter_text"):
        yield from run.iter_text()
        return

    for message in run.messages():
        msg_type = getattr(message, "type", None)
        if msg_type != "assistant":
            continue
        message_obj = getattr(message, "message", None)
        content = getattr(message_obj, "content", None) if message_obj else None
        if not content:
            continue
        for block in content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text = getattr(block, "text", "")
                if text:
                    yield text


def wait_run_result(run: Any) -> Any:
    """等待 Run 完成并校验状态。"""
    try:
        result = run.wait()
    except Exception as exc:
        from cursor_chat.chat.client_helper import _raise_startup_error

        _raise_startup_error(exc)

    status = getattr(result, "status", None) or getattr(run, "status", "unknown")
    if status == "error":
        run_id = getattr(run, "id", "")
        agent_id = getattr(run, "agent_id", "")
        detail = getattr(result, "result", "") or "Run 执行失败"
        raise ChatRunError(str(detail), run_id=run_id, agent_id=agent_id)
    return result


def collect_assistant_text(run: Any) -> str:
    """非流式收集完整回复（优先 run.text()）。"""
    text_fn = getattr(run, "text", None)
    if callable(text_fn):
        try:
            return text_fn()
        except Exception:
            pass
    parts = list(iter_assistant_text(run))
    if parts:
        return "".join(parts)
    result = wait_run_result(run)
    return getattr(result, "result", "") or ""
