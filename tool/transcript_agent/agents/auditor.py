from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from lite_agent.client import OpenAIClient

from transcript_agent.base import AuditIssue, AuditResult, TranscriptContext, parse_json_object
from transcript_agent.llm_config import llm_chat

AUDIT_SYSTEM = """你是一位投资内容质量审计员，按给定 Audit Rubric 评估提取结果。
输出严格 JSON 对象，不要输出其它文字。

格式：
{
  "passed": false,
  "score": 72,
  "dimension_scores": {
    "fidelity": 18,
    "investment_value": 16,
    "structure": 14,
    "actionability": 12,
    "noise_removal": 12
  },
  "issues": [
    {"severity": "major|minor", "field": "insights[2]", "problem": "问题描述", "fix": "修复建议"}
  ],
  "revision_prompt": "给提取 Agent 的修订指令，≤500字"
}

判定规则：
- passed = score >= pass_threshold 且无 major issue
- fidelity 单项低于 min_fidelity 则 passed 必须为 false
- 不得编造原文没有的标的、数据、结论"""


class AuditorAgent:
    name = "AuditorAgent"

    def __init__(self, llm: OpenAIClient):
        self.llm = llm

    async def audit(
        self,
        ctx: TranscriptContext,
        draft: Dict[str, Any],
        *,
        on_progress: Optional[Any] = None,
    ) -> AuditResult:
        if on_progress:
            on_progress(70, f"审计中（第 {ctx.retry_count + 1} 轮）...")
        rubric = ctx.rubric
        rubric_version = str(rubric.get("version") or "unknown")
        user = f"""Audit Rubric（版本 {rubric_version}）：
{json.dumps(rubric, ensure_ascii=False, indent=2)}

源转写稿（前 8000 字，供核对忠实度）：
{ctx.source_text[:8000]}

待审计提取结果：
{json.dumps(draft, ensure_ascii=False, indent=2)}
"""
        raw = await llm_chat(self.llm, AUDIT_SYSTEM, user)
        data = parse_json_object(raw) or {}
        issues = []
        for row in data.get("issues") or []:
            if not isinstance(row, dict):
                continue
            issues.append(
                AuditIssue(
                    severity=str(row.get("severity") or "minor"),
                    field=str(row.get("field") or ""),
                    problem=str(row.get("problem") or ""),
                    fix=str(row.get("fix") or ""),
                )
            )
        score = int(data.get("score") or 0)
        dim_scores = {k: int(v) for k, v in (data.get("dimension_scores") or {}).items()}
        passed = bool(data.get("passed"))
        pass_threshold = int(rubric.get("pass_threshold") or 85)
        min_fidelity = int(rubric.get("min_fidelity") or 16)
        has_major = any(i.severity == "major" for i in issues)
        fidelity = dim_scores.get("fidelity", 0)
        if score < pass_threshold or has_major or fidelity < min_fidelity:
            passed = False
        result = AuditResult(
            passed=passed,
            score=score,
            rubric_version=rubric_version,
            dimension_scores=dim_scores,
            issues=issues,
            revision_prompt=str(data.get("revision_prompt") or ""),
        )
        ctx.audit_history.append(result)
        return result
