from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from lite_agent.client import OpenAIClient

from transcript_agent.base import RULES_DIR, AuditIssue, TextBatch, TranscriptContext, parse_json_object
from transcript_agent.llm_config import llm_chat
from transcript_agent.revision import apply_revision_patch, build_issues_block, collect_revision_scope
from transcript_agent.timing import TimingCollector

LOGIC_FAMILIES_PATH = RULES_DIR / "logic_families.yaml"

EXTRACT_SYSTEM = """你是一位 A 股投资定性分析助手，从视频 ASR 转写稿中提取「可 RAG 匹配的逻辑框架卡」。
输出严格 JSON 对象，不要输出其它文字。

目标：每种独立的投资逻辑框架输出一张 logic_card，供后续 Agent 对个股/板块做逻辑命中匹配。

格式：
{
  "core_thesis": "本段核心论点（一句话）",
  "logic_cards": [
    {
      "logic_id": "snake_case 唯一标识，如 phenomenon_event",
      "logic_name": "中文逻辑名，如 现象级高景气赛道",
      "logic_family": "逻辑大类，如 超景气价值投机 / 困境反转 / 均值回归 / 流动性宏观 / 个股评分选股",
      "summary": "一句话定义，用于向量检索",
      "triggers": ["命中该逻辑的信号，如 技术突破、政策带明确资金规模、产能出清"],
      "veto_conditions": ["否决条件，如 真实性未验证、规模过小、重复催化"],
      "preconditions": ["适用前置，如 板块已确认高景气"],
      "analysis_steps": ["分析步骤 1", "分析步骤 2"],
      "answer_sections": ["命中后回答应包含：事件定性", "适用边界", "风险提示"],
      "related_logics": ["相关 logic_id"],
      "exclude_logics": ["互斥 logic_id"],
      "keywords": ["检索关键词"],
      "evidence_quotes": ["原文短引句，≤80字"],
      "examples": [{"subject": "标的或行业", "scenario": "情境", "note": "说明"}]
    }
  ],
  "removed_summary": "已剔除的寒暄/重复/无关内容"
}

要求：
- 原文出现几种独立逻辑就提取几张卡，勿合并；未出现的逻辑框架不要编造
- 逻辑大类不限于超景气，困境反转、均值回归、流动性、选股评分、交易纪律等须分别成卡
- triggers / veto_conditions / keywords 要具体，便于与网上检索事实匹配
- 每条 logic_card 至少 1 条 evidence_quotes
- logic_id 使用英文 snake_case，全篇保持一致
- 若本段无有效投资逻辑，logic_cards 返回 []"""

REVISE_INCREMENTAL_SYSTEM = """你是一位 A 股投资定性分析助手，根据审计反馈对 logic_cards 提取结果做增量修订。
输出严格 JSON patch 对象，不要输出其它文字。只输出需要修改的字段，未修改的字段不要出现。

格式：
{
  "core_thesis": "仅当需要修改时填写",
  "logic_cards": [
    {
      "logic_id": "必须填写，用于定位要修改的卡",
      "logic_name": "仅当需要修改时填写",
      "triggers": ["仅当需要修改时填写"],
      "veto_conditions": [],
      "evidence_quotes": [],
      "analysis_steps": [],
      "answer_sections": [],
      "keywords": []
    }
  ],
  "removed_summary": "仅当需要修改时填写"
}

要求：
- 只修正审计 issues 指出的问题，其它 logic_card 与字段保持不动（不要出现在输出中）
- evidence_quotes 必须能在提供的源转写稿中找到依据，不得编造
- logic_cards 数组中每张卡必须带 logic_id"""


def _load_logic_families_hint() -> str:
    if not LOGIC_FAMILIES_PATH.exists():
        return ""
    with LOGIC_FAMILIES_PATH.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    families = data.get("families") or []
    if not families:
        return ""
    return json.dumps(families, ensure_ascii=False, indent=2)


class ExtractorAgent:
    name = "ExtractorAgent"

    def __init__(self, llm: OpenAIClient):
        self.llm = llm

    async def extract_batches(
        self,
        ctx: TranscriptContext,
        *,
        on_progress: Optional[Any] = None,
        timing: Optional[TimingCollector] = None,
    ) -> None:
        ctx.batch_extracts = []
        total = len(ctx.batches)
        if total == 0:
            return
        families_hint = _load_logic_families_hint()
        completed = 0
        progress_lock = asyncio.Lock()

        async def run_one(idx: int, batch: TextBatch) -> tuple[int, Dict[str, Any]]:
            nonlocal completed
            step_name = f"extract_batch_{idx + 1}/{total}"
            if timing:
                with timing.step(step_name):
                    data = await self._extract_batch(ctx, batch, idx, total, families_hint)
            else:
                data = await self._extract_batch(ctx, batch, idx, total, families_hint)
            async with progress_lock:
                completed += 1
                if on_progress:
                    pct = 10 + int(50 * completed / total)
                    on_progress(pct, f"提取逻辑卡 {completed}/{total}...")
            return idx, data

        results = await asyncio.gather(*(run_one(i, b) for i, b in enumerate(ctx.batches)))
        results.sort(key=lambda item: item[0])
        ctx.batch_extracts = [data for _, data in results if data]

    async def _extract_batch(
        self,
        ctx: TranscriptContext,
        batch: TextBatch,
        idx: int,
        total: int,
        families_hint: str,
    ) -> Dict[str, Any]:
        hint_block = f"\n逻辑大类参考（按原文选用，勿强行套用）：\n{families_hint}\n" if families_hint else ""
        user = f"""源文件: {ctx.source_path.name}
批次: {idx + 1}/{total}（{batch.char_count} 字）
{hint_block}
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
        issues: Optional[List[AuditIssue]] = None,
        on_progress: Optional[Any] = None,
        timing: Optional[TimingCollector] = None,
    ) -> Dict[str, Any]:
        if on_progress:
            on_progress(30, "根据审计反馈增量修订逻辑卡...")
        issue_list = issues or []
        scope = collect_revision_scope(previous_draft, issue_list)
        issues_block = build_issues_block(issue_list, revision_prompt)
        user = f"""源文件: {ctx.source_path.name}

审计修订指令：
{issues_block}

待修订片段（仅这些字段需要修改，其余保持不变）：
{json.dumps(scope, ensure_ascii=False, indent=2)}

源转写稿（前 8000 字，供核对引用）：
{ctx.source_text[:8000]}

请输出增量 patch JSON。"""
        async def _call() -> Dict[str, Any]:
            raw = await llm_chat(self.llm, REVISE_INCREMENTAL_SYSTEM, user)
            patch = parse_json_object(raw)
            if not patch:
                return previous_draft
            return apply_revision_patch(previous_draft, patch)

        if timing:
            with timing.step(f"revise_incremental_r{ctx.retry_count + 1}"):
                return await _call()
        return await _call()
