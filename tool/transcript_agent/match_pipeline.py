from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from transcript_agent.base import ANALYSIS_RESULTS_DIR
from transcript_agent.llm_config import create_llm_client_from_cfg, load_llm_config
from transcript_agent.query.answerer import AnswerAgent
from transcript_agent.query.card_registry import card_by_id, get_registry_meta, load_registry_cards, save_registry_snapshot
from transcript_agent.query.embed_index import EmbedIndex
from transcript_agent.query.matcher import LogicMatcher
from transcript_agent.query.models import AnalysisResult
from transcript_agent.query.parser import QueryParserAgent
from transcript_agent.query.wind_fetcher import WindAIFinClient
from wind_config import load_wind_config

ProgressCallback = Callable[[int, str], None]


def _safe_stem(subject: str) -> str:
    text = re.sub(r'[<>:"/\\|?*]', "_", subject.strip())[:80]
    return text or "analysis"


class MatchOrchestrator:
    def __init__(self, *, provider: Optional[str] = None):
        self.provider = provider
        self.wind_cfg = load_wind_config()

    async def rebuild_index(self) -> Dict[str, Any]:
        cards = load_registry_cards(force_rescan=True)
        meta = save_registry_snapshot(cards)
        manifest = EmbedIndex().build(cards)
        return {**meta, **manifest}

    async def run(
        self,
        raw_query: str,
        *,
        subject_type: str = "auto",
        context: str = "",
        on_progress: Optional[ProgressCallback] = None,
    ) -> Dict[str, Any]:
        started = time.monotonic()

        def prog(pct: int, msg: str) -> None:
            if on_progress:
                on_progress(pct, msg)

        cfg = load_llm_config(self.provider)
        active = cfg["active_provider"]
        parser_llm = create_llm_client_from_cfg(cfg, provider=active, temperature=0.1)
        match_llm = create_llm_client_from_cfg(cfg, provider=active, temperature=0.1)
        answer_llm = create_llm_client_from_cfg(cfg, provider=active, temperature=0.3)

        full_query = raw_query.strip()
        if context.strip():
            full_query = f"{full_query} {context.strip()}"

        prog(5, "解析查询...")
        query_ctx = await QueryParserAgent(parser_llm).parse(full_query, subject_type=subject_type)

        prog(20, "Wind 取数...")
        facts = WindAIFinClient(self.wind_cfg).fetch(query_ctx)

        prog(35, "加载 logic_cards 索引...")
        cards = load_registry_cards(force_rescan=True)
        if not cards:
            raise ValueError("logic_cards 为空，请先完成转录精炼并确保 processed/*_logic_cards.json 存在")

        threshold = int(self.wind_cfg.get("match_threshold") or 70)
        top_k = int(self.wind_cfg.get("match_top_k") or 10)
        matcher = LogicMatcher(match_llm, threshold=threshold, top_k=top_k)

        prog(55, "向量召回 + LLM 重排...")
        match_result = await matcher.match(query_ctx, facts, cards)

        selected_card = card_by_id(cards, match_result.selected_logic_id) if match_result.selected_logic_id else None

        prog(75, "生成分析报告...")
        analysis_md = await AnswerAgent(answer_llm).generate(query_ctx, facts, selected_card, match_result)

        elapsed = round(time.monotonic() - started, 1)
        meta = {
            "llm_provider": active,
            "llm_model": cfg.get("model"),
            "elapsed_sec": elapsed,
            "wind_status": facts.wind_status,
            "registry_card_count": len(cards),
            "embedding_version": matcher.index.version,
        }
        result = AnalysisResult(
            subject=query_ctx.subject,
            query=query_ctx,
            facts=facts,
            match=match_result,
            analysis_md=analysis_md,
            meta=meta,
        )
        return self._save_outputs(result)

    def _save_outputs(self, result: AnalysisResult) -> Dict[str, Any]:
        ANALYSIS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        stem = _safe_stem(result.subject)
        analysis_path = ANALYSIS_RESULTS_DIR / f"{stem}_analysis.md"
        match_path = ANALYSIS_RESULTS_DIR / f"{stem}_match.json"
        pipeline_path = ANALYSIS_RESULTS_DIR / f"{stem}_pipeline.json"

        with analysis_path.open("w", encoding="utf-8") as f:
            f.write(result.analysis_md)
        with match_path.open("w", encoding="utf-8") as f:
            json.dump(result.match.to_dict(), f, ensure_ascii=False, indent=2)
        with pipeline_path.open("w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

        return {
            "subject": result.subject,
            "analysis": analysis_path.name,
            "match": match_path.name,
            "pipeline": pipeline_path.name,
            "selected_logic_id": result.match.selected_logic_id,
            "selected_score": result.match.selected_score,
            "sufficient": result.match.sufficient,
            "wind_status": result.facts.wind_status,
            "elapsed_sec": result.meta.get("elapsed_sec"),
        }


def get_registry_stats(*, refresh: bool = False) -> Dict[str, Any]:
    meta = get_registry_meta(refresh=refresh)
    index = EmbedIndex()
    index_loaded = index.load()
    return {
        **meta,
        "embedding_loaded": index_loaded,
        "embedding_version": index.version if index_loaded else None,
    }


def list_analysis_results() -> List[Dict[str, Any]]:
    if not ANALYSIS_RESULTS_DIR.exists():
        return []
    items = []
    for path in sorted(ANALYSIS_RESULTS_DIR.glob("*_analysis.md")):
        stem = path.name.replace("_analysis.md", "")
        match_path = ANALYSIS_RESULTS_DIR / f"{stem}_match.json"
        items.append(
            {
                "stem": stem,
                "analysis": path.name,
                "match": f"{stem}_match.json",
                "has_match": match_path.is_file(),
            }
        )
    return items


if __name__ == "__main__":
    import asyncio
    import sys

    q = " ".join(sys.argv[1:]) or "京东方A 玻璃基板"
    orch = MatchOrchestrator()

    async def _main() -> None:
        if "--rebuild-index" in sys.argv:
            print(await orch.rebuild_index())
            return
        out = await orch.run(q)
        print(json.dumps(out, ensure_ascii=False, indent=2))

    asyncio.run(_main())
