from __future__ import annotations

import json
import re
from typing import Any, Optional

from lite_agent.client import OpenAIClient

from transcript_agent.base import parse_json_object
from transcript_agent.llm_config import llm_chat
from transcript_agent.query.models import QueryContext

PARSER_SYSTEM = """你是 A 股投资查询解析助手。将用户输入解析为结构化检索意图。
输出严格 JSON 对象：
{
  "subject": "标的名称或板块名",
  "subject_type": "stock|sector|auto",
  "sector_hint": "所属行业/板块",
  "event_keywords": ["事件关键词"],
  "wind_intents": ["stock_profile", "recent_news", "financial_summary", "sector_context"],
  "search_queries": ["用于检索的查询词"]
}

规则：
- subject_type=stock 表示个股，sector 表示板块/行业
- wind_intents 从以下选用：stock_profile, recent_news, financial_summary, sector_context, macro_policy
- 若输入含6位代码或 .SH/.SZ，subject_type 应为 stock
- search_queries 2~4 条，覆盖公司与事件"""


class QueryParserAgent:
    name = "QueryParserAgent"

    def __init__(self, llm: OpenAIClient):
        self.llm = llm

    async def parse(self, raw_query: str, subject_type: str = "auto") -> QueryContext:
        user = f"用户输入：{raw_query}\n期望 subject_type：{subject_type}"
        raw = await llm_chat(self.llm, PARSER_SYSTEM, user)
        data = parse_json_object(raw) or {}
        ctx = QueryContext(
            raw_query=raw_query,
            subject=(data.get("subject") or raw_query).strip(),
            subject_type=(data.get("subject_type") or subject_type or "auto").strip(),
            sector_hint=(data.get("sector_hint") or "").strip(),
            event_keywords=[str(x).strip() for x in (data.get("event_keywords") or []) if str(x).strip()],
            wind_intents=[str(x).strip() for x in (data.get("wind_intents") or []) if str(x).strip()],
            search_queries=[str(x).strip() for x in (data.get("search_queries") or []) if str(x).strip()],
        )
        ctx = self._apply_rules(ctx, raw_query)
        if not ctx.wind_intents:
            ctx.wind_intents = ["stock_profile", "recent_news", "financial_summary"]
        if not ctx.search_queries:
            ctx.search_queries = [raw_query]
        return ctx

    @staticmethod
    def _apply_rules(ctx: QueryContext, raw_query: str) -> QueryContext:
        code_match = re.search(r"\b(\d{6})\b", raw_query)
        if code_match or re.search(r"\.\s*(SH|SZ)\b", raw_query, re.I):
            ctx.subject_type = "stock"
        if "板块" in raw_query or "行业" in raw_query:
            if ctx.subject_type == "auto":
                ctx.subject_type = "sector"
        if not ctx.subject:
            ctx.subject = raw_query.strip()
        return ctx
