#!/usr/bin/env python3
"""示例：带本地知识库的聊天。"""

from __future__ import annotations

from pathlib import Path

from _common import setup_path

setup_path()

from cursor_chat import ChatStartupError, CursorChatService, LocalDirectoryRetriever, load_chat_config


def main() -> None:
    config = load_chat_config()
    sample_docs = Path(__file__).resolve().parent / "sample_docs"

    retriever = LocalDirectoryRetriever([sample_docs], max_chunk_chars=1200)
    print(f"知识库已加载 {retriever.chunk_count} 个片段，目录: {sample_docs}\n")

    config.system_prompt = "优先依据 <knowledge_base> 中的内容回答；若无相关信息再说明。"
    config.knowledge.directories = [str(sample_docs)]

    try:
        with CursorChatService(config) as service:
            service.attach_knowledge(retriever)

            question = "cursor_chat 模块默认使用什么模型？配置文件有哪些？"
            print(f"用户: {question}\n")

            # 演示检索结果
            hits = retriever.retrieve(question, top_k=3)
            print("--- 检索命中 ---")
            for hit in hits:
                print(f"[{hit.score:.3f}] {hit.source}")
            print()

            response = service.chat(question, use_knowledge=True)
            print(f"助手: {response.text}")
    except ChatStartupError as exc:
        print(f"[启动失败] {exc}")


if __name__ == "__main__":
    main()
