from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from transcript_agent.base import ANALYSIS_RESULTS_DIR
from transcript_agent.llm_config import create_llm_client_from_cfg, load_llm_config
from transcript_agent.query.answerer import AnswerAgent
from transcript_agent.query.card_registry import card_by_id, get_registry_meta, load_registry_cards, save_registry_snapshot
from transcript_agent.query.embed_index import EmbedIndex
from transcript_agent.query.fact_provider import fetch_research_facts
from transcript_agent.query.matcher import LogicMatcher
from transcript_agent.query.models import AnalysisResult
from transcript_agent.query.parser import QueryParserAgent
from transcript_agent.query.wanxing_fetcher import (
    apply_stock_code_from_query,
    resolve_stock_code_by_name,
)
from wind_config import load_wind_config, save_wind_response_enabled
from wanxing_config import save_research_response_enabled

ProgressCallback = Callable[[int, str], None]


_STEM_TS_RE = re.compile(r"^(.+)_(\d{8}_\d{6})$")


def _safe_stem(subject: str) -> str:
    text = re.sub(r'[<>:"/\\|?*]', "_", subject.strip())[:80]
    return text or "analysis"


def _timestamp_suffix() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _output_stem(subject: str) -> str:
    return f"{_safe_stem(subject)}_{_timestamp_suffix()}"


def _parse_stem_timestamp(stem: str) -> Tuple[str, Optional[str]]:
    m = _STEM_TS_RE.match(stem)
    if not m:
        return stem, None
    subject_part, ts = m.group(1), m.group(2)
    try:
        dt = datetime.strptime(ts, "%Y%m%d_%H%M%S")
        return subject_part, dt.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return stem, None


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
        deep_research: bool = False,
        on_progress: Optional[ProgressCallback] = None,
    ) -> Dict[str, Any]:
        started = time.monotonic()
        research_mode = "wind_alice" if deep_research else "wanxing"

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
        apply_stock_code_from_query(query_ctx)

        if not deep_research and query_ctx.subject_type in ("stock", "auto"):
            if not query_ctx.stock_code:
                prog(10, f"万行名称转代码: {query_ctx.subject}...")
                code, record = resolve_stock_code_by_name(query_ctx.subject)
                query_ctx.stock_code = code
                query_ctx.stock_code_resolve = record
                if code:
                    prog(11, f"{query_ctx.subject} -> {code}")
                else:
                    prog(11, "名称转代码失败，后续取数可能不完整")

        research_prog_state = {"pct": 20}

        def research_progress(msg: str) -> None:
            research_prog_state["pct"] = min(research_prog_state["pct"] + 3, 50)
            prog(research_prog_state["pct"], msg)

        if deep_research:
            prog(15, "Wind Alice 深度调研...")
        else:
            prog(15, "万行取数...")

        facts = fetch_research_facts(
            query_ctx,
            deep_research=deep_research,
            context=context,
            on_progress=research_progress,
        )

        prog(55, "加载 logic_cards 索引...")
        cards = load_registry_cards(force_rescan=True)
        if not cards:
            raise ValueError("logic_cards 为空，请先完成转录精炼并确保 processed/*_logic_cards.json 存在")

        threshold = int(self.wind_cfg.get("match_threshold") or 70)
        top_k = int(self.wind_cfg.get("match_top_k") or 10)
        matcher = LogicMatcher(match_llm, threshold=threshold, top_k=top_k)

        prog(65, "向量召回 + LLM 重排...")
        match_result = await matcher.match(query_ctx, facts, cards)

        selected_card = card_by_id(cards, match_result.selected_logic_id) if match_result.selected_logic_id else None

        prog(80, "生成分析报告...")
        analysis_md = await AnswerAgent(answer_llm).generate(query_ctx, facts, selected_card, match_result)

        elapsed = round(time.monotonic() - started, 1)
        generated_at = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        meta = {
            "llm_provider": active,
            "llm_model": cfg.get("model"),
            "elapsed_sec": elapsed,
            "generated_at": generated_at,
            "research_mode": research_mode,
            "stock_code": query_ctx.stock_code,
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
        stem = _output_stem(result.subject)
        analysis_path = ANALYSIS_RESULTS_DIR / f"{stem}_analysis.md"
        match_path = ANALYSIS_RESULTS_DIR / f"{stem}_match.json"
        pipeline_path = ANALYSIS_RESULTS_DIR / f"{stem}_pipeline.json"
        wanxing_path = ANALYSIS_RESULTS_DIR / f"{stem}_wanxing.json"
        alice_md_path = ANALYSIS_RESULTS_DIR / f"{stem}_alice.md"
        alice_json_path = ANALYSIS_RESULTS_DIR / f"{stem}_alice.json"

        with analysis_path.open("w", encoding="utf-8") as f:
            f.write(result.analysis_md)
        with match_path.open("w", encoding="utf-8") as f:
            json.dump(result.match.to_dict(), f, ensure_ascii=False, indent=2)
        with pipeline_path.open("w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

        research_mode = result.meta.get("research_mode") or result.facts.research_mode
        provider_saved = False

        if research_mode == "wanxing" and save_research_response_enabled():
            payload = {
                "subject": result.subject,
                "research_mode": research_mode,
                "wind_status": result.facts.wind_status,
                "query": result.query.to_dict(),
                "calls": result.facts.wind_calls,
            }
            with wanxing_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            provider_saved = True
        elif research_mode == "wind_alice" and save_wind_response_enabled(self.wind_cfg):
            if result.facts.research_raw_md:
                with alice_md_path.open("w", encoding="utf-8") as f:
                    f.write(result.facts.research_raw_md)
            alice_payload = {
                "subject": result.subject,
                "research_mode": research_mode,
                "wind_status": result.facts.wind_status,
                "query": result.query.to_dict(),
                "calls": result.facts.wind_calls,
            }
            with alice_json_path.open("w", encoding="utf-8") as f:
                json.dump(alice_payload, f, ensure_ascii=False, indent=2)
            provider_saved = True

        output: Dict[str, Any] = {
            "subject": result.subject,
            "stem": stem,
            "generated_at": result.meta.get("generated_at"),
            "stock_code": result.query.stock_code,
            "analysis": analysis_path.name,
            "analysis_path": str(analysis_path),
            "match": match_path.name,
            "pipeline": pipeline_path.name,
            "research_mode": research_mode,
            "selected_logic_id": result.match.selected_logic_id,
            "selected_score": result.match.selected_score,
            "sufficient": result.match.sufficient,
            "wind_status": result.facts.wind_status,
            "provider_saved": provider_saved,
            "elapsed_sec": result.meta.get("elapsed_sec"),
        }
        if research_mode == "wanxing" and provider_saved:
            output["wanxing"] = wanxing_path.name
        if research_mode == "wind_alice" and provider_saved:
            output["alice"] = alice_md_path.name if result.facts.research_raw_md else None
            output["alice_meta"] = alice_json_path.name
        return output


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
    paths = sorted(
        ANALYSIS_RESULTS_DIR.glob("*_analysis.md"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in paths:
        stem = path.name.replace("_analysis.md", "")
        subject_part, generated_at = _parse_stem_timestamp(stem)
        label = f"{subject_part} ({generated_at})" if generated_at else stem
        match_path = ANALYSIS_RESULTS_DIR / f"{stem}_match.json"
        wanxing_path = ANALYSIS_RESULTS_DIR / f"{stem}_wanxing.json"
        alice_md_path = ANALYSIS_RESULTS_DIR / f"{stem}_alice.md"
        items.append(
            {
                "stem": stem,
                "subject": subject_part,
                "label": label,
                "generated_at": generated_at,
                "analysis": path.name,
                "match": f"{stem}_match.json",
                "has_match": match_path.is_file(),
                "wanxing": wanxing_path.name if wanxing_path.is_file() else None,
                "has_wanxing": wanxing_path.is_file(),
                "alice": alice_md_path.name if alice_md_path.is_file() else None,
                "has_alice": alice_md_path.is_file(),
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
        deep = "--deep" in sys.argv
        out = await orch.run(q, deep_research=deep)
        print(json.dumps(out, ensure_ascii=False, indent=2))

    asyncio.run(_main())
