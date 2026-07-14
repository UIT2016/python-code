# 投资逻辑知识库

本目录用于 Agent3 归纳审计规则，以及后续 RAG 逻辑匹配的参考文档。

## 建议放入的文档

- 各类投资框架笔记（超景气、困境反转、均值回归、流动性等）
- 术语表与 logic_id 命名约定
- 好/坏 logic_card 样例（`examples/good_card.json`）
- 书籍或文章摘录（Markdown 格式）

## 逻辑大类参考

见 [`../rules/logic_families.yaml`](../rules/logic_families.yaml)，提炼时会作为 hint 注入，**仅提取原文实际出现的框架**。

## 提炼输出（RAG 用）

精炼后会生成：

| 文件 | 用途 |
|------|------|
| `{标题}_logic_cards.json` | **RAG 主数据源**，每张卡含 triggers、veto、keywords、rag_text |
| `{标题}_structured.json` | 完整结构化结果 |
| `{标题}_essence.md` | 人类可读预览 |

## logic_card 字段说明

- `logic_id` / `logic_name`：逻辑标识
- `logic_family`：大类（超景气 / 困境反转 / …）
- `triggers` / `veto_conditions`：匹配与否决信号
- `analysis_steps` / `answer_sections`：命中后如何分析、如何回答
- `rag_text`：预拼接的检索文本，可直接 embedding
