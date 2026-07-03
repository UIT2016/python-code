from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

from transcript_agent.base import ANALYSIS_RESULTS_DIR, LogicCard, build_rag_text, now_version
from transcript_agent.query.card_registry import load_registry_cards, save_registry_snapshot

EMBED_MANIFEST = ANALYSIS_RESULTS_DIR / "embeddings_manifest.json"
EMBED_VECTORS = ANALYSIS_RESULTS_DIR / "embeddings.npy"


def _load_wind_embed_config() -> Dict[str, Any]:
    from wind_config import load_wind_config

    return load_wind_config()


def _resolve_embedding_key(cfg: Dict[str, Any]) -> str:
    key = (cfg.get("embedding_api_key") or "").strip()
    if key:
        return key
    from transcript_agent.llm_config import load_llm_config

    llm_cfg = load_llm_config()
    return (llm_cfg.get("llm") or {}).get("qwen", {}).get("api_key") or llm_cfg.get("api_key") or ""


def embed_texts(texts: List[str], cfg: Optional[Dict[str, Any]] = None) -> List[List[float]]:
    if not texts:
        return []
    cfg = cfg or _load_wind_embed_config()
    api_key = _resolve_embedding_key(cfg)
    if not api_key:
        raise ValueError("缺少 embedding API Key，请在 tool/wind.local.json 或 message/config.local.json 配置")
    url = (cfg.get("embedding_api_url") or "").rstrip("/") + "/embeddings"
    model = cfg.get("embedding_model") or "text-embedding-v3"
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "input": texts},
        timeout=120,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Embedding API 失败 ({resp.status_code}): {resp.text[:300]}")
    data = resp.json()
    items = sorted(data.get("data") or [], key=lambda x: x.get("index", 0))
    return [item["embedding"] for item in items if "embedding" in item]


def _normalize_vectors(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return matrix / norms


class EmbedIndex:
    def __init__(self) -> None:
        self.logic_ids: List[str] = []
        self.vectors: Optional[np.ndarray] = None
        self.version: str = ""

    def build(self, cards: Optional[List[LogicCard]] = None) -> Dict[str, Any]:
        cards = cards if cards is not None else load_registry_cards(force_rescan=True)
        if not cards:
            raise ValueError("logic_cards 为空，请先完成转录精炼")
        save_registry_snapshot(cards)
        texts = [build_rag_text(c) for c in cards]
        vectors = embed_texts(texts)
        matrix = _normalize_vectors(np.array(vectors, dtype=np.float32))
        self.logic_ids = [c.logic_id for c in cards]
        self.vectors = matrix
        self.version = now_version()
        ANALYSIS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        np.save(EMBED_VECTORS, matrix)
        manifest = {
            "version": self.version,
            "logic_ids": self.logic_ids,
            "card_count": len(cards),
            "embedding_model": _load_wind_embed_config().get("embedding_model"),
        }
        with EMBED_MANIFEST.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        return manifest

    def load(self) -> bool:
        if not EMBED_MANIFEST.exists() or not EMBED_VECTORS.exists():
            return False
        with EMBED_MANIFEST.open(encoding="utf-8") as f:
            manifest = json.load(f)
        self.logic_ids = manifest.get("logic_ids") or []
        self.vectors = np.load(EMBED_VECTORS)
        self.version = manifest.get("version") or ""
        return bool(len(self.logic_ids))

    def ensure_loaded(self, cards: Optional[List[LogicCard]] = None) -> None:
        if self.load():
            return
        self.build(cards)

    def search(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        if self.vectors is None or not self.logic_ids:
            raise RuntimeError("embedding 索引未构建")
        q_vec = embed_texts([query_text])[0]
        q = np.array(q_vec, dtype=np.float32)
        q = q / (np.linalg.norm(q) or 1.0)
        scores = self.vectors @ q
        k = min(top_k, len(scores))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(self.logic_ids[i], float(scores[i])) for i in idx]
