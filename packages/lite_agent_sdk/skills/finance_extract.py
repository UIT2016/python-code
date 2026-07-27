import json
import re
from typing import List

from lite_agent.client import OpenAIClient

from lite_agent_sdk.base import BaseSkill, ExtractedItem, MessageBatch, SkillContext, llm_chat

SYSTEM_PROMPT = """你是一位专业的 A 股投资聊天消息分析助手。
从一批聊天室消息中提取与投资相关的热点信息，输出严格 JSON 数组，不要输出其它文字。

每条记录格式：
{"term":"热点词或个股/板块名","category":"stock|sector|keyword|policy|strategy","sentiment":"乐观|谨慎|中性|恐慌","note":"一句话说明"}

要求：
- term 要具体，优先保留股票名称/简称/代码、板块、政策关键词
- 忽略「收到一条语音消息」等无效内容
- 同一批次内相同 term 只保留一条，count 不必输出
- 若本批无有效信息，返回 []"""


class FinanceExtractSkill(BaseSkill):
    name = "FinanceExtractSkill"

    def __init__(self, llm: OpenAIClient):
        self.llm = llm

    async def execute(self, ctx: SkillContext) -> None:
        raise RuntimeError("FinanceExtractSkill 请通过 execute_batch 逐批调用")

    async def execute_batch(self, batch: MessageBatch, ctx: SkillContext) -> List[ExtractedItem]:
        if not batch.text.strip():
            return []
        user_prompt = f"""数据日期: {ctx.date_str}
批次: {batch.batch_id + 1}
涉及聊天室: {', '.join(batch.room_titles)}
消息条数: {batch.message_count}

消息内容：
{batch.text}
"""
        raw = await llm_chat(self.llm, SYSTEM_PROMPT, user_prompt)
        return _parse_extract_json(raw, batch.room_titles)


def _parse_extract_json(raw: str, rooms: List[str]) -> List[ExtractedItem]:
    text = raw.strip()
    match = re.search(r"\[[\s\S]*\]", text)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []

    items: List[ExtractedItem] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        term = str(row.get("term") or "").strip()
        if not term:
            continue
        items.append(
            ExtractedItem(
                term=term,
                category=str(row.get("category") or "keyword"),
                sentiment=str(row.get("sentiment") or ""),
                note=str(row.get("note") or ""),
                rooms=list(rooms),
            )
        )
    return items
