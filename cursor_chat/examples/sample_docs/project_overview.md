# 项目简介

cursor_chat 是基于 **Cursor 官方 Python SDK**（cursor-sdk）的聊天模块。

## 能力

- 普通聊天与多轮对话（Agent.send 自动维护上下文）
- 新建会话 / 清空历史（重建 Agent）
- System Prompt（消息前缀注入）
- 流式输出（run.iter_text）
- 本地目录知识库检索

## 环境要求

- Python 3.10+
- Cursor API Key（环境变量 CURSOR_API_KEY 或 config.local.json）
