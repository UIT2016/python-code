import os

os.environ["DASHSCOPE_API_KEY"] = "sk-9f303fc5a6924aed8b99d9f77a750c72"

from langchain.messages import HumanMessage
from langchain_community.chat_models.tongyi import ChatTongyi

chatLLM = ChatTongyi(
    name="qwen3.5-397b-a17b",  # qwen3.5
    streaming=True,
)
text = "你是谁，我调用的是哪个版本"
res = chatLLM.stream([HumanMessage(content=text)], streaming=True)
for r in res:
    # r 是 AIMessageChunk 对象
    # 提取 content 字段
    chunk_content = r.content

    # 实时打印片段 (end="" 防止自动换行，实现打字机效果)
    print(chunk_content, end="", flush=True)
