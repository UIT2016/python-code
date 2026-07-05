from __future__ import annotations

import json
from typing import List

from lite_agent.client import OpenAIClient

from transcript_agent.base import LogicCard, parse_json_object
from transcript_agent.llm_config import llm_chat
from transcript_agent.query.card_registry import card_by_id
from transcript_agent.query.embed_index import EmbedIndex
from transcript_agent.query.models import BusinessSegment, FactBundle, SegmentLogicScore

SEGMENT_MATCH_SYSTEM = """你是投资逻辑框架匹配审计员。针对单一业务线，评估候选 logic_card 的匹配度。
输出严格 JSON：
{
  "logic_id": "xxx",
  "score": 88,
  "trigger_hits": ["..."],
  "reason": "简要理由"
}
score 0-100；有 veto 触发时 score 不得高于 60；logic_id 必须来自候选列表。"""


class BusinessSegmentMatcher:
    def __init__(self, llm: OpenAIClient, *, top_k: int = 6):
        self.llm = llm
        self.top_k = top_k
        self.index = EmbedIndex()

    async def score_all(
        self,
        segments: List[BusinessSegment],
        cards: List[LogicCard],
        facts: FactBundle,
    ) -> List[SegmentLogicScore]:
        self.index.ensure_loaded(cards)
        results: List[SegmentLogicScore] = []
        for segment in segments:
            score = await self._score_one(segment, cards, facts)
            results.append(score)
        return results

    async def _score_one(
        self,
        segment: BusinessSegment,
        cards: List[LogicCard],
        facts: FactBundle,
    ) -> SegmentLogicScore:
        match_text = f"{segment.name} {segment.summary}\n{facts.summary_for_match[:600]}"
        vector_hits = self.index.search(match_text, top_k=self.top_k)
        candidate_ids = [lid for lid, _ in vector_hits]
        candidate_cards = [card_by_id(cards, lid) for lid in candidate_ids]
        candidate_cards = [c for c in candidate_cards if c]
        if not candidate_cards:
            return SegmentLogicScore(segment=segment.name, logic_id="", score=0, reason="无候选 logic_card")

        payload = [
            {
                "logic_id": c.logic_id,
                "logic_name": c.logic_name,
                "triggers": c.triggers,
                "veto_conditions": c.veto_conditions,
                "summary": c.summary,
            }
            for c in candidate_cards
        ]
        user = f"""业务线: {segment.name}
占比: {segment.revenue_share}
摘要: {segment.summary}

事实摘要:
{facts.summary_for_match[:800]}

候选 logic_cards:
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""
        raw = await llm_chat(self.llm, SEGMENT_MATCH_SYSTEM, user)
        data = parse_json_object(raw) or {}
        logic_id = str(data.get("logic_id") or "").strip()
        score = int(data.get("score") or 0)
        triggers = [str(x) for x in (data.get("trigger_hits") or []) if str(x).strip()]
        reason = str(data.get("reason") or "")
        if logic_id not in {c.logic_id for c in candidate_cards}:
            logic_id = candidate_cards[0].logic_id
            score = min(score, 50)
        card = card_by_id(cards, logic_id)
        if card:
            veto_hits = [v for v in card.veto_conditions if v and v in facts.summary_for_match + segment.summary]
            if veto_hits:
                score = min(score, 60)
        return SegmentLogicScore(
            segment=segment.name,
            logic_id=logic_id,
            score=score,
            trigger_hits=triggers,
            reason=reason,
        )
