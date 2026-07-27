from __future__ import annotations

import json
from typing import List

from lite_agent.client import OpenAIClient

from transcript_agent.base import parse_json_object
from transcript_agent.llm_config import llm_chat
from transcript_agent.query.models import BalanceSheetRisk, FactBundle

RISK_SYSTEM = """你是财务风险分析助手。根据资产负债表检索文本，提取可能影响投资的风险要素（不计入框架匹配分）。
输出严格 JSON：
{
  "risks": [
    {"factor": "风险描述", "evidence": "数据依据", "severity": "low|medium|high"}
  ]
}
关注：固定资产占比、流动比率、有息负债、商誉、应收账款、存货等。无依据则不输出。"""


class BalanceSheetRiskAnalyzer:
    def __init__(self, llm: OpenAIClient):
        self.llm = llm

    async def analyze(self, facts: FactBundle) -> List[BalanceSheetRisk]:
        text = facts.research_raw_sections.get("balance_sheet") or ""
        if not text.strip():
            for item in facts.facts:
                if item.fact_type == "balance_sheet":
                    text = item.text
                    break
        if not text.strip():
            return []
        user = f"资产负债表检索文本:\n{text[:8000]}"
        raw = await llm_chat(self.llm, RISK_SYSTEM, user)
        data = parse_json_object(raw) or {}
        rows = data.get("risks") or []
        risks: List[BalanceSheetRisk] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            factor = str(row.get("factor") or "").strip()
            if not factor:
                continue
            severity = str(row.get("severity") or "medium").lower()
            if severity not in ("low", "medium", "high"):
                severity = "medium"
            risks.append(
                BalanceSheetRisk(
                    factor=factor,
                    evidence=str(row.get("evidence") or "")[:500],
                    severity=severity,
                )
            )
        return risks[:8]
