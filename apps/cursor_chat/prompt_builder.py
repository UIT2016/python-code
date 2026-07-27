"""将 System Prompt 与知识库上下文拼接到用户消息中。

Cursor SDK 的 Agent 没有独立的 system_prompt 参数，官方 Agent 使用内置系统提示。
业务层通过消息前缀注入自定义指令，与 Cursor IDE 中 <custom_instructions> 的用法一致。
"""

from __future__ import annotations


def build_user_message(
    user_message: str,
    *,
    system_prompt: str | None = None,
    knowledge_context: str | None = None,
) -> str:
    parts: list[str] = []
    if system_prompt and system_prompt.strip():
        parts.append(f"<custom_instructions>\n{system_prompt.strip()}\n</custom_instructions>")
    if knowledge_context and knowledge_context.strip():
        parts.append(f"<knowledge_base>\n{knowledge_context.strip()}\n</knowledge_base>")
    parts.append(f"<user_query>\n{user_message.strip()}\n</user_query>")
    return "\n\n".join(parts)


def format_knowledge_context(chunks: list) -> str:
    """将检索片段格式化为可读上下文。"""
    if not chunks:
        return ""
    lines: list[str] = []
    for idx, chunk in enumerate(chunks, 1):
        source = getattr(chunk, "source", "unknown")
        score = getattr(chunk, "score", 0.0)
        content = getattr(chunk, "content", str(chunk))
        lines.append(f"### 片段 {idx}（来源: {source}，相关度: {score:.3f}）\n{content}")
    return "\n\n".join(lines)
