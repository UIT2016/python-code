from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from transcript_agent.base import (
    ANALYSIS_RESULTS_DIR,
    OLD_PROCESSED_DIR,
    PROCESSED_DIR,
    LogicCard,
    merge_logic_cards,
    now_version,
    parse_logic_card,
)

REGISTRY_PATH = ANALYSIS_RESULTS_DIR / "logic_registry.json"
REGISTRY_META_PATH = ANALYSIS_RESULTS_DIR / "logic_registry_meta.json"


def _scan_logic_card_files() -> List[Path]:
    paths: List[Path] = []
    for root in (PROCESSED_DIR, OLD_PROCESSED_DIR):
        if not root.exists():
            continue
        paths.extend(sorted(root.glob("*_logic_cards.json")))
    return paths


def load_registry_cards(force_rescan: bool = False) -> List[LogicCard]:
    if not force_rescan and REGISTRY_PATH.exists():
        with REGISTRY_PATH.open(encoding="utf-8") as f:
            rows = json.load(f)
        cards = [parse_logic_card(row) for row in rows if isinstance(row, dict)]
        return [c for c in cards if c]

    all_cards: List[LogicCard] = []
    for path in _scan_logic_card_files():
        with path.open(encoding="utf-8") as f:
            rows = json.load(f)
        if not isinstance(rows, list):
            continue
        for row in rows:
            card = parse_logic_card(row)
            if card:
                all_cards.append(card)
    return merge_logic_cards(all_cards)


def save_registry_snapshot(cards: List[LogicCard]) -> Dict[str, Any]:
    ANALYSIS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = [c.to_dict() for c in cards]
    meta = {
        "version": now_version(),
        "card_count": len(cards),
        "logic_ids": sorted({c.logic_id for c in cards}),
        "source_files": len(_scan_logic_card_files()),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    with REGISTRY_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with REGISTRY_META_PATH.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


def rescan_registry_meta() -> Dict[str, Any]:
    cards = load_registry_cards(force_rescan=True)
    return save_registry_snapshot(cards)


def get_registry_meta(*, refresh: bool = False) -> Dict[str, Any]:
    if refresh or not REGISTRY_META_PATH.exists():
        return rescan_registry_meta()
    with REGISTRY_META_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def card_by_id(cards: List[LogicCard], logic_id: str) -> Optional[LogicCard]:
    for card in cards:
        if card.logic_id == logic_id:
            return card
    return None
