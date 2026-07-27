from __future__ import annotations

import json
from typing import List

from lite_agent.client import OpenAIClient

from transcript_agent.base import LogicCard, parse_json_object
from transcript_agent.llm_config import llm_chat
from transcript_agent.query.card_registry import card_by_id
from transcript_agent.query.models import (
    FactBundle,
    LogicRanking,
    MatchResult,
    QueryContext,
    SegmentLogicScore,
    StockCompositeMatch,
)

SYNTH_SYSTEM = """你是投资逻辑综合评审员。根据各业务线对 logic_card 的评分，综合选出个股主框架。
输出严格 JSON：
{
  "composite_logic_id": "xxx",
  "composite_score": 85,
  "runner_up": ["id2", "id3"],
  "synthesis_reason": "综合理由（说明如何权衡各业务线）"
}
composite_logic_id 必须来自输入的业务线评分或候选 logic_id；composite_score 0-100。"""


class StockScoreSynthesizer:
    def __init__(self, llm: OpenAIClient):
        self.llm = llm

    async def synthesize(
        self,
        segment_scores: List[SegmentLogicScore],
        facts: FactBundle,
        cards: List[LogicCard],
        query: QueryContext,
        *,
        threshold: int = 70,
    ) -> StockCompositeMatch:
        if not segment_scores:
            return StockCompositeMatch(
                segment_scores=[],
                composite_logic_id="",
                composite_score=0,
                synthesis_reason="无业务线评分",
            )
        user = f"""标的: {query.subject}
代码: {query.stock_code}

业务线评分:
{json.dumps([s.to_dict() for s in segment_scores], ensure_ascii=False, indent=2)}

事实摘要:
{facts.summary_for_match}
"""
        raw = await llm_chat(self.llm, SYNTH_SYSTEM, user)
        data = parse_json_object(raw) or {}
        composite_id = str(data.get("composite_logic_id") or "").strip()
        composite_score = int(data.get("composite_score") or 0)
        runner_up = [str(x) for x in (data.get("runner_up") or []) if str(x).strip()]
        reason = str(data.get("synthesis_reason") or "")
        if not composite_id:
            best = max(segment_scores, key=lambda s: s.score)
            composite_id = best.logic_id
            composite_score = best.score
        valid_ids = {s.logic_id for s in segment_scores if s.logic_id}
        if composite_id not in valid_ids and segment_scores:
            composite_id = segment_scores[0].logic_id
        return StockCompositeMatch(
            segment_scores=segment_scores,
            composite_logic_id=composite_id,
            composite_score=composite_score,
            synthesis_reason=reason,
            runner_up=runner_up,
        )

    @staticmethod
    def to_match_result(
        composite: StockCompositeMatch,
        cards: List[LogicCard],
        facts: FactBundle,
        query: QueryContext,
        *,
        threshold: int = 70,
    ) -> MatchResult:
        match_text = f"{query.raw_query}\n{facts.summary_for_match}"
        rankings: List[LogicRanking] = []
        for seg in composite.segment_scores:
            if not seg.logic_id:
                continue
            rankings.append(
                LogicRanking(
                    logic_id=seg.logic_id,
                    score=seg.score,
                    trigger_hits=seg.trigger_hits,
                    reason=f"[{seg.segment}] {seg.reason}",
                )
            )
        rankings.sort(key=lambda r: r.score, reverse=True)
        selected = composite.composite_logic_id
        selected_score = composite.composite_score
        card = card_by_id(cards, selected) if selected else None
        veto_hits: List[str] = []
        if card:
            veto_hits = [v for v in card.veto_conditions if v and v in facts.summary_for_match]
        sufficient = selected_score >= threshold and not veto_hits
        if selected_score < 50:
            selected = ""
        for ranking in rankings:
            if ranking.logic_id == composite.composite_logic_id and veto_hits:
                ranking.veto_hits = veto_hits
                ranking.score = min(ranking.score, 60)
        return MatchResult(
            selected_logic_id=selected if sufficient else (selected if selected_score >= 50 else ""),
            selected_score=selected_score,
            runner_up=composite.runner_up or [r.logic_id for r in rankings[1:4] if r.logic_id != selected],
            rankings=rankings,
            match_text=match_text,
            sufficient=sufficient,
        )
