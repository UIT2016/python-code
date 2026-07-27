from __future__ import annotations

import json
from typing import List

from lite_agent.client import OpenAIClient

from transcript_agent.base import parse_json_object
from transcript_agent.llm_config import llm_chat
from transcript_agent.query.models import BusinessSegment, FactBundle, QueryContext

EXTRACT_SYSTEM = """你是财务分析助手。从「主营业务/业务构成」检索文本中抽取 2-6 条业务线。
输出严格 JSON：
{
  "segments": [
    {"name": "业务名称", "revenue_share": "占比或未知", "summary": "一句话描述"}
  ]
}
只输出 JSON，不要编造文本中未出现的业务。"""


class BusinessSegmentExtractor:
    def __init__(self, llm: OpenAIClient):
        self.llm = llm

    async def extract(self, query: QueryContext, facts: FactBundle) -> List[BusinessSegment]:
        text = facts.research_raw_sections.get("business_mix") or ""
        if not text.strip():
            for item in facts.facts:
                if item.fact_type == "business_mix":
                    text = item.text
                    break
        if not text.strip():
            return [BusinessSegment(name=query.subject, summary=query.raw_query)]
        user = f"标的:{query.subject}\n\n检索文本:\n{text[:6000]}"
        raw = await llm_chat(self.llm, EXTRACT_SYSTEM, user)
        data = parse_json_object(raw) or {}
        rows = data.get("segments") or []
        segments: List[BusinessSegment] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name") or "").strip()
            if not name:
                continue
            segments.append(
                BusinessSegment(
                    name=name,
                    revenue_share=str(row.get("revenue_share") or ""),
                    summary=str(row.get("summary") or "")[:500],
                )
            )
        return segments[:6] if segments else [BusinessSegment(name=query.subject, summary=text[:300])]
