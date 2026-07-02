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
class InsightItem:
    category: str
    claim: str
    reasoning: str = ""
    evidence_quotes: List[str] = field(default_factory=list)
    actionable: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "claim": self.claim,
            "reasoning": self.reasoning,
            "evidence_quotes": self.evidence_quotes,
            "actionable": self.actionable,
        }


@dataclass
class ExtractResult:
    source_file: str
    video_meta: Dict[str, str]
    core_thesis: str = ""
    framework_tags: List[str] = field(default_factory=list)
    insights: List[InsightItem] = field(default_factory=list)
    removed_summary: str = ""
    revision_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_file": self.source_file,
            "video_meta": self.video_meta,
            "core_thesis": self.core_thesis,
            "framework_tags": self.framework_tags,
            "insights": [i.to_dict() for i in self.insights],
            "removed_summary": self.removed_summary,
            "revision_notes": self.revision_notes,
        }


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
