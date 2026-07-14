from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class QueryContext:
    raw_query: str
    subject: str = ""
    subject_type: str = "auto"
    sector_hint: str = ""
    event_keywords: List[str] = field(default_factory=list)
    wind_intents: List[str] = field(default_factory=list)
    search_queries: List[str] = field(default_factory=list)
    stock_code: str = ""
    stock_code_resolve: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_query": self.raw_query,
            "subject": self.subject,
            "subject_type": self.subject_type,
            "sector_hint": self.sector_hint,
            "event_keywords": self.event_keywords,
            "wind_intents": self.wind_intents,
            "search_queries": self.search_queries,
            "stock_code": self.stock_code,
            "stock_code_resolve": self.stock_code_resolve,
        }


@dataclass
class BusinessSegment:
    name: str
    revenue_share: str = ""
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "revenue_share": self.revenue_share, "summary": self.summary}


@dataclass
class SegmentLogicScore:
    segment: str
    logic_id: str
    score: int
    trigger_hits: List[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment": self.segment,
            "logic_id": self.logic_id,
            "score": self.score,
            "trigger_hits": self.trigger_hits,
            "reason": self.reason,
        }


@dataclass
class BalanceSheetRisk:
    factor: str
    evidence: str
    severity: str = "medium"

    def to_dict(self) -> Dict[str, Any]:
        return {"factor": self.factor, "evidence": self.evidence, "severity": self.severity}


@dataclass
class StockCompositeMatch:
    segment_scores: List[SegmentLogicScore]
    composite_logic_id: str
    composite_score: int
    synthesis_reason: str = ""
    runner_up: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_scores": [s.to_dict() for s in self.segment_scores],
            "composite_logic_id": self.composite_logic_id,
            "composite_score": self.composite_score,
            "synthesis_reason": self.synthesis_reason,
            "runner_up": self.runner_up,
        }


@dataclass
class FactItem:
    fact_type: str
    text: str
    date: str = ""
    source: str = "wind"

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.fact_type, "text": self.text, "date": self.date, "source": self.source}


@dataclass
class FactBundle:
    subject: str
    facts: List[FactItem] = field(default_factory=list)
    summary_for_match: str = ""
    wind_status: str = "ok"
    research_mode: str = "wanxing"
    research_raw_md: str = ""
    wind_calls: List[Dict[str, Any]] = field(default_factory=list)
    business_segments: List[BusinessSegment] = field(default_factory=list)
    segment_match: Optional[StockCompositeMatch] = None
    balance_sheet_risks: List[BalanceSheetRisk] = field(default_factory=list)
    research_raw_sections: Dict[str, str] = field(default_factory=dict)

    @property
    def research_status(self) -> str:
        return self.wind_status

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject,
            "facts": [f.to_dict() for f in self.facts],
            "summary_for_match": self.summary_for_match,
            "wind_status": self.wind_status,
            "research_mode": self.research_mode,
            "has_research_raw": bool(self.research_raw_md),
            "business_segments": [s.to_dict() for s in self.business_segments],
            "segment_match": self.segment_match.to_dict() if self.segment_match else None,
            "balance_sheet_risks": [r.to_dict() for r in self.balance_sheet_risks],
            "research_raw_sections": self.research_raw_sections,
        }


@dataclass
class LogicRanking:
    logic_id: str
    score: int
    trigger_hits: List[str] = field(default_factory=list)
    veto_hits: List[str] = field(default_factory=list)
    confidence: str = "medium"
    reason: str = ""
    vector_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "logic_id": self.logic_id,
            "score": self.score,
            "trigger_hits": self.trigger_hits,
            "veto_hits": self.veto_hits,
            "confidence": self.confidence,
            "reason": self.reason,
            "vector_score": round(self.vector_score, 4),
        }


@dataclass
class MatchResult:
    selected_logic_id: str
    selected_score: int
    runner_up: List[str] = field(default_factory=list)
    rankings: List[LogicRanking] = field(default_factory=list)
    match_text: str = ""
    sufficient: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_logic_id": self.selected_logic_id,
            "selected_score": self.selected_score,
            "runner_up": self.runner_up,
            "rankings": [r.to_dict() for r in self.rankings],
            "match_text": self.match_text,
            "sufficient": self.sufficient,
        }


@dataclass
class AnalysisResult:
    subject: str
    query: QueryContext
    facts: FactBundle
    match: MatchResult
    analysis_md: str
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject,
            "query": self.query.to_dict(),
            "facts": self.facts.to_dict(),
            "match": self.match.to_dict(),
            "meta": self.meta,
        }
