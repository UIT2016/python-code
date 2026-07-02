from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from lite_agent.client import OpenAIClient

from transcript_agent.base import TextBatch, TranscriptContext, parse_json_object
from transcript_agent.llm_config import llm_chat

EXTRACT_SYSTEM = """你是一位 A 股投资定性分析助手，从视频 ASR 转写稿中提取投资精华。
输出严格 JSON 对象，不要输出其它文字。

格式：
{
  "core_thesis": "本段核心论点（一句话）",
  "framework_tags": ["框架标签"],
  "insights": [
    {
      "category": "strategy|sector|stock|risk|mindset|case",
      "claim": "观点陈述",
      "reasoning": "因果逻辑",
      "evidence_quotes": ["原文短引句，≤80字，必须来自输入"],
      "actionable": "可操作建议（可为空）"
    }
  ],
  "removed_summary": "本段已剔除的寒暄/重复/无关内容简述"
}

要求：
- 保留：投资框架、定性判断、因果链、板块/个股案例、风险提示
- 剔除：B站口播套话、求关注、重复口癖、与投资无关闲聊
- 每条 insight 必须有 evidence_quotes，且不得编造原文没有的标的或数据
- 若本段无有效投资内容，insights 返回 []"""

REVISE_SYSTEM = """你是一位 A 股投资定性分析助手，根据审计反馈修订已有提取结果。
输出与提取任务相同的 JSON 对象结构。
只修正审计 issues 指出的问题，保留其它正确内容，不得删除未提及问题的 insights。"""


class ExtractorAgent:
    name = "ExtractorAgent"

    def __init__(self, llm: OpenAIClient):
        self.llm = llm

    async def extract_batches(
        self,
        ctx: TranscriptContext,
        *,
        on_progress: Optional[Any] = None,
    ) -> None:
        ctx.batch_extracts = []
        total = len(ctx.batches)
        for idx, batch in enumerate(ctx.batches):
            if on_progress:
                pct = 10 + int(50 * idx / max(total, 1))
                on_progress(pct, f"提取批次 {idx + 1}/{total}...")
            data = await self._extract_batch(ctx, batch, idx, total)
            if data:
                ctx.batch_extracts.append(data)

    async def _extract_batch(
        self,
        ctx: TranscriptContext,
        batch: TextBatch,
        idx: int,
        total: int,
    ) -> Dict[str, Any]:
        user = f"""源文件: {ctx.source_path.name}
批次: {idx + 1}/{total}（{batch.char_count} 字）

转写内容：
{batch.text}
"""
        raw = await llm_chat(self.llm, EXTRACT_SYSTEM, user)
        return parse_json_object(raw) or {}

    async def revise_draft(
        self,
        ctx: TranscriptContext,
        previous_draft: Dict[str, Any],
        revision_prompt: str,
        *,
        on_progress: Optional[Any] = None,
    ) -> Dict[str, Any]:
        if on_progress:
            on_progress(30, "根据审计反馈修订...")
        user = f"""源文件: {ctx.source_path.name}
审计修订指令：
{revision_prompt}

当前提取结果：
{json.dumps(previous_draft, ensure_ascii=False, indent=2)}

完整转写稿（供核对原文）：
{ctx.source_text[:16000]}

请输出修订后的完整 JSON（含 core_thesis、framework_tags、insights、removed_summary）。"""
        raw = await llm_chat(self.llm, REVISE_SYSTEM, user)
        data = parse_json_object(raw)
        return data if data else previous_draft
