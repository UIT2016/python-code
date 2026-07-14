from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

AGENT_DIR = Path(__file__).resolve().parent
TOOL_DIR = AGENT_DIR.parent
TRANSCRIPT_DIR = TOOL_DIR / "transcripts"
PROCESSED_DIR = AGENT_DIR / "processed"
OLD_PROCESSED_DIR = AGENT_DIR / "old_processed"
ANALYSIS_RESULTS_DIR = AGENT_DIR / "analysis_results"
RULES_DIR = AGENT_DIR / "rules"
KNOWLEDGE_DIR = AGENT_DIR / "knowledge"
ACTIVE_RUBRIC_PATH = RULES_DIR / "active_rubric.json"
MANUAL_OVERRIDES_PATH = RULES_DIR / "manual_overrides.yaml"
BASE_RUBRIC_PATH = RULES_DIR / "base_rubric.yaml"

ProgressCallback = Callable[[int, str], None]


@dataclass
class TextBatch:
    batch_id: int
    text: str
    char_count: int


@dataclass
class LogicCard:
    logic_id: str
    logic_name: str
    logic_family: str = ""
    summary: str = ""
    triggers: List[str] = field(default_factory=list)
    veto_conditions: List[str] = field(default_factory=list)
    preconditions: List[str] = field(default_factory=list)
    analysis_steps: List[str] = field(default_factory=list)
    answer_sections: List[str] = field(default_factory=list)
    related_logics: List[str] = field(default_factory=list)
    exclude_logics: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    evidence_quotes: List[str] = field(default_factory=list)
    examples: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "logic_id": self.logic_id,
            "logic_name": self.logic_name,
            "logic_family": self.logic_family,
            "summary": self.summary,
            "triggers": self.triggers,
            "veto_conditions": self.veto_conditions,
            "preconditions": self.preconditions,
            "analysis_steps": self.analysis_steps,
            "answer_sections": self.answer_sections,
            "related_logics": self.related_logics,
            "exclude_logics": self.exclude_logics,
            "keywords": self.keywords,
            "evidence_quotes": self.evidence_quotes,
            "examples": self.examples,
            "rag_text": build_rag_text(self),
        }


@dataclass
class ExtractResult:
    source_file: str
    video_meta: Dict[str, str]
    core_thesis: str = ""
    logic_cards: List[LogicCard] = field(default_factory=list)
    removed_summary: str = ""
    revision_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_file": self.source_file,
            "video_meta": self.video_meta,
            "core_thesis": self.core_thesis,
            "logic_cards": [c.to_dict() for c in self.logic_cards],
            "removed_summary": self.removed_summary,
            "revision_notes": self.revision_notes,
        }


def normalize_logic_id(raw: str) -> str:
    text = (raw or "").strip().lower()
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"[\s-]+", "_", text).strip("_")
    return text or "unnamed_logic"


def build_rag_text(card: LogicCard) -> str:
    parts = [
        f"逻辑: {card.logic_name} ({card.logic_id})",
        f"大类: {card.logic_family}",
        f"定义: {card.summary}",
        f"触发信号: {'; '.join(card.triggers)}",
        f"否决条件: {'; '.join(card.veto_conditions)}",
        f"前置条件: {'; '.join(card.preconditions)}",
        f"分析步骤: {'; '.join(card.analysis_steps)}",
        f"回答结构: {'; '.join(card.answer_sections)}",
        f"关键词: {', '.join(card.keywords)}",
    ]
    if card.related_logics:
        parts.append(f"相关逻辑: {', '.join(card.related_logics)}")
    if card.exclude_logics:
        parts.append(f"互斥逻辑: {', '.join(card.exclude_logics)}")
    if card.examples:
        ex = "; ".join(
            f"{e.get('subject', '')}:{e.get('note', e.get('scenario', ''))}"
            for e in card.examples
            if isinstance(e, dict)
        )
        if ex:
            parts.append(f"案例: {ex}")
    return "\n".join(p for p in parts if p.split(": ", 1)[-1])


def _merge_str_lists(*groups: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for group in groups:
        for item in group:
            s = str(item).strip()
            if s and s not in seen:
                seen.add(s)
                out.append(s)
    return out


def parse_logic_card(row: Dict[str, Any]) -> Optional[LogicCard]:
    if not isinstance(row, dict):
        return None
    logic_id = normalize_logic_id(str(row.get("logic_id") or row.get("logic_name") or ""))
    logic_name = str(row.get("logic_name") or logic_id).strip()
    if not logic_name:
        return None
    examples: List[Dict[str, str]] = []
    for ex in row.get("examples") or []:
        if isinstance(ex, dict):
            examples.append(
                {
                    "subject": str(ex.get("subject") or "").strip(),
                    "scenario": str(ex.get("scenario") or "").strip(),
                    "note": str(ex.get("note") or "").strip(),
                }
            )
        elif isinstance(ex, str) and ex.strip():
            examples.append({"subject": ex.strip(), "scenario": "", "note": ""})
    return LogicCard(
        logic_id=logic_id,
        logic_name=logic_name,
        logic_family=str(row.get("logic_family") or "").strip(),
        summary=str(row.get("summary") or "").strip(),
        triggers=[str(x).strip() for x in (row.get("triggers") or []) if str(x).strip()],
        veto_conditions=[str(x).strip() for x in (row.get("veto_conditions") or []) if str(x).strip()],
        preconditions=[str(x).strip() for x in (row.get("preconditions") or []) if str(x).strip()],
        analysis_steps=[str(x).strip() for x in (row.get("analysis_steps") or []) if str(x).strip()],
        answer_sections=[str(x).strip() for x in (row.get("answer_sections") or []) if str(x).strip()],
        related_logics=[normalize_logic_id(str(x)) for x in (row.get("related_logics") or []) if str(x).strip()],
        exclude_logics=[normalize_logic_id(str(x)) for x in (row.get("exclude_logics") or []) if str(x).strip()],
        keywords=[str(x).strip() for x in (row.get("keywords") or []) if str(x).strip()],
        evidence_quotes=[str(x).strip() for x in (row.get("evidence_quotes") or []) if str(x).strip()],
        examples=examples,
    )


def merge_logic_cards(cards: List[LogicCard]) -> List[LogicCard]:
    merged: Dict[str, LogicCard] = {}
    for card in cards:
        key = card.logic_id
        if key not in merged:
            merged[key] = card
            continue
        existing = merged[key]
        merged[key] = LogicCard(
            logic_id=key,
            logic_name=existing.logic_name or card.logic_name,
            logic_family=existing.logic_family or card.logic_family,
            summary=existing.summary if len(existing.summary) >= len(card.summary) else card.summary,
            triggers=_merge_str_lists(existing.triggers, card.triggers),
            veto_conditions=_merge_str_lists(existing.veto_conditions, card.veto_conditions),
            preconditions=_merge_str_lists(existing.preconditions, card.preconditions),
            analysis_steps=_merge_str_lists(existing.analysis_steps, card.analysis_steps),
            answer_sections=_merge_str_lists(existing.answer_sections, card.answer_sections),
            related_logics=_merge_str_lists(existing.related_logics, card.related_logics),
            exclude_logics=_merge_str_lists(existing.exclude_logics, card.exclude_logics),
            keywords=_merge_str_lists(existing.keywords, card.keywords),
            evidence_quotes=_merge_str_lists(existing.evidence_quotes, card.evidence_quotes)[:5],
            examples=existing.examples + [e for e in card.examples if e not in existing.examples],
        )
    return list(merged.values())


@dataclass
class AuditIssue:
    severity: str
    field: str
    problem: str
    fix: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity,
            "field": self.field,
            "problem": self.problem,
            "fix": self.fix,
        }


@dataclass
class AuditResult:
    passed: bool
    score: int
    rubric_version: str
    dimension_scores: Dict[str, int] = field(default_factory=dict)
    issues: List[AuditIssue] = field(default_factory=list)
    revision_prompt: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "score": self.score,
            "rubric_version": self.rubric_version,
            "dimension_scores": self.dimension_scores,
            "issues": [i.to_dict() for i in self.issues],
            "revision_prompt": self.revision_prompt,
        }


@dataclass
class TranscriptContext:
    source_path: Path
    source_text: str = ""
    batch_size: int = 7000
    max_retry: int = 3
    provider: str = "deepseek"
    batches: List[TextBatch] = field(default_factory=list)
    batch_extracts: List[Dict[str, Any]] = field(default_factory=list)
    draft: Optional[ExtractResult] = None
    audit_history: List[AuditResult] = field(default_factory=list)
    rubric: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    essence_md: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


class BaseSkill(ABC):
    name: str = "BaseSkill"

    @abstractmethod
    async def execute(self, ctx: TranscriptContext) -> None:
        raise NotImplementedError


def parse_video_meta(filename: str) -> Dict[str, str]:
    stem = Path(filename).stem
    match = re.search(r"\[(BV[\w]+)\]\s*$", stem)
    bvid = match.group(1) if match else ""
    title = stem[: match.start()].strip() if match else stem
    return {"title": title, "bvid": bvid}


def parse_json_object(raw: str) -> Optional[Dict[str, Any]]:
    text = raw.strip()
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def now_version() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
