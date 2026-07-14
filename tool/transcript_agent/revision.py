from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Set, Tuple

from transcript_agent.base import AuditIssue, normalize_logic_id

_CARD_FIELD_RE = re.compile(r"^logic_cards\[(\d+)\](?:\.(.+))?$")


def strip_rag_text(draft: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = copy.deepcopy(draft)
    cards = cleaned.get("logic_cards") or []
    for card in cards:
        if isinstance(card, dict):
            card.pop("rag_text", None)
    return cleaned


def _parse_issue_field(field: str) -> Tuple[str, int | None, str | None]:
    text = (field or "").strip()
    match = _CARD_FIELD_RE.match(text)
    if match:
        return "logic_card", int(match.group(1)), match.group(2)
    if text.startswith("logic_cards"):
        return "logic_cards", None, None
    return text.split(".", 1)[0] if text else "unknown", None, None


def collect_revision_scope(draft: Dict[str, Any], issues: List[AuditIssue]) -> Dict[str, Any]:
    draft = strip_rag_text(draft)
    card_indices: Set[int] = set()
    top_fields: Set[str] = set()
    cards = draft.get("logic_cards") or []

    for issue in issues:
        kind, idx, _sub = _parse_issue_field(issue.field)
        if kind == "logic_card" and idx is not None:
            card_indices.add(idx)
        elif kind in {"logic_cards", "logic_card"}:
            card_indices.update(range(len(cards)))
        elif kind in {"core_thesis", "removed_summary"}:
            top_fields.add(kind)

    scope: Dict[str, Any] = {}
    if "core_thesis" in top_fields or (not issues and draft.get("core_thesis")):
        if draft.get("core_thesis"):
            scope["core_thesis"] = draft.get("core_thesis")
    if "removed_summary" in top_fields:
        scope["removed_summary"] = draft.get("removed_summary")

    scoped_cards: List[Dict[str, Any]] = []
    if card_indices:
        for idx in sorted(card_indices):
            if 0 <= idx < len(cards) and isinstance(cards[idx], dict):
                scoped_cards.append(cards[idx])
    elif issues and cards:
        mentions_cards = any("logic_card" in issue.field for issue in issues)
        only_core = top_fields == {"core_thesis"} and not mentions_cards
        if mentions_cards or not only_core:
            scoped_cards = list(cards)

    if scoped_cards:
        scope["logic_cards"] = scoped_cards

    return scope


def build_issues_block(issues: List[AuditIssue], revision_prompt: str) -> str:
    lines: List[str] = []
    if revision_prompt.strip():
        lines.append(revision_prompt.strip())
    if issues:
        if lines:
            lines.append("")
        lines.append("结构化问题列表：")
        for issue in issues:
            lines.append(f"- [{issue.severity}] {issue.field}: {issue.problem} → {issue.fix}")
    return "\n".join(lines)


def apply_revision_patch(draft: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    if not patch:
        return draft
    result = copy.deepcopy(draft)
    if patch.get("core_thesis"):
        result["core_thesis"] = patch["core_thesis"]
    if patch.get("removed_summary"):
        result["removed_summary"] = patch["removed_summary"]

    patch_cards = patch.get("logic_cards") or []
    if not patch_cards:
        return result

    cards = result.get("logic_cards") or []
    patch_by_id = {
        normalize_logic_id(str(c.get("logic_id") or "")): c
        for c in patch_cards
        if isinstance(c, dict) and c.get("logic_id")
    }
    merged_cards: List[Dict[str, Any]] = []
    for card in cards:
        if not isinstance(card, dict):
            merged_cards.append(card)
            continue
        lid = normalize_logic_id(str(card.get("logic_id") or ""))
        delta = patch_by_id.pop(lid, None)
        if delta:
            merged_cards.append(_merge_card_patch(card, delta))
        else:
            merged_cards.append(card)
    for delta in patch_by_id.values():
        merged_cards.append(delta)
    result["logic_cards"] = merged_cards
    return result


def _merge_card_patch(card: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(card)
    for key, value in patch.items():
        if key == "logic_id":
            continue
        if value is None:
            continue
        if isinstance(value, list) and not value:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        merged[key] = value
    return merged
