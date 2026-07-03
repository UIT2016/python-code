from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from lite_agent.client import OpenAIClient

from transcript_agent.base import LogicCard, parse_json_object
from transcript_agent.llm_config import llm_chat
from transcript_agent.query.card_registry import card_by_id
from transcript_agent.query.embed_index import EmbedIndex
from transcript_agent.query.models import FactBundle, LogicRanking, MatchResult, QueryContext

MATCH_SYSTEM = """你是投资逻辑框架匹配审计员。根据调研事实（万行/Wind Alice/用户输入）与 logic_card 定义，评估每张卡的匹配度。
输出严格 JSON：
{
  "rankings": [
    {
      "logic_id": "xxx",
      "score": 88,
      "trigger_hits": ["命中的触发信号"],
      "veto_hits": ["命中的否决条件，无则[]"],
      "confidence": "high|medium|low",
      "reason": "简要理由"
    }
  ],
  "selected_logic_id": "得分最高且 veto 为空的 logic_id",
  "runner_up": ["次优 logic_id"]
}

规则：
- score 0-100；有 veto_hits 时 score 不得高于 60
- 只从候选 logic_id 中选择，不得编造新 id
- 事实不足时降低 confidence 并降低 score"""


class LogicMatcher:
    def __init__(self, llm: OpenAIClient, *, threshold: int = 70, top_k: int = 10):
        self.llm = llm
        self.threshold = threshold
        self.top_k = top_k
        self.index = EmbedIndex()

    async def match(
        self,
        query: QueryContext,
        facts: FactBundle,
        cards: List[LogicCard],
    ) -> MatchResult:
        self.index.ensure_loaded(cards)
        match_text = f"{query.raw_query}\n{facts.summary_for_match}"
        vector_hits = self.index.search(match_text, top_k=self.top_k)
        candidate_ids = [lid for lid, _ in vector_hits]
        candidate_cards = [card_by_id(cards, lid) for lid in candidate_ids]
        candidate_cards = [c for c in candidate_cards if c]
        if not candidate_cards:
            return MatchResult(
                selected_logic_id="",
                selected_score=0,
                sufficient=False,
                match_text=match_text,
            )

        card_payload = [
            {
                "logic_id": c.logic_id,
                "logic_name": c.logic_name,
                "logic_family": c.logic_family,
                "summary": c.summary,
                "triggers": c.triggers,
                "veto_conditions": c.veto_conditions,
                "preconditions": c.preconditions,
            }
            for c in candidate_cards
        ]
        user = f"""用户查询：
{json.dumps(query.to_dict(), ensure_ascii=False, indent=2)}

事实摘要：
{facts.summary_for_match}

候选 logic_cards：
{json.dumps(card_payload, ensure_ascii=False, indent=2)}
"""
        raw = await llm_chat(self.llm, MATCH_SYSTEM, user)
        data = parse_json_object(raw) or {}
        vector_map = {lid: score for lid, score in vector_hits}
        rankings = self._parse_rankings(data, vector_map)
        rankings = self._apply_hard_rules(rankings, cards)
        rankings.sort(key=lambda r: r.score, reverse=True)

        selected = (data.get("selected_logic_id") or "").strip()
        if not selected and rankings:
            selected = rankings[0].logic_id
        selected_score = next((r.score for r in rankings if r.logic_id == selected), 0)
        runner_up = [r.logic_id for r in rankings[1:4] if r.logic_id != selected]
        sufficient = selected_score >= self.threshold and not any(
            r.logic_id == selected and r.veto_hits for r in rankings
        )
        return MatchResult(
            selected_logic_id=selected if sufficient else (selected if selected_score >= 50 else ""),
            selected_score=selected_score,
            runner_up=runner_up,
            rankings=rankings,
            match_text=match_text,
            sufficient=sufficient,
        )

    @staticmethod
    def _parse_rankings(data: Dict[str, Any], vector_map: Dict[str, float]) -> List[LogicRanking]:
        rows = data.get("rankings") or []
        out: List[LogicRanking] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            lid = str(row.get("logic_id") or "").strip()
            if not lid:
                continue
            out.append(
                LogicRanking(
                    logic_id=lid,
                    score=int(row.get("score") or 0),
                    trigger_hits=[str(x) for x in (row.get("trigger_hits") or [])],
                    veto_hits=[str(x) for x in (row.get("veto_hits") or [])],
                    confidence=str(row.get("confidence") or "medium"),
                    reason=str(row.get("reason") or ""),
                    vector_score=vector_map.get(lid, 0.0),
                )
            )
        return out

    @staticmethod
    def _apply_hard_rules(rankings: List[LogicRanking], cards: List[LogicCard]) -> List[LogicRanking]:
        card_map = {c.logic_id: c for c in cards}
        adjusted: List[LogicRanking] = []
        for row in rankings:
            score = row.score
            if row.veto_hits:
                score = min(score, 60)
            card = card_map.get(row.logic_id)
            if card:
                for ex in card.exclude_logics:
                    for other in rankings:
                        if other.logic_id == ex and other.score > score:
                            other.score = max(other.score - 15, 0)
            adjusted.append(
                LogicRanking(
                    logic_id=row.logic_id,
                    score=score,
                    trigger_hits=row.trigger_hits,
                    veto_hits=row.veto_hits,
                    confidence=row.confidence,
                    reason=row.reason,
                    vector_score=row.vector_score,
                )
            )
        return adjusted
