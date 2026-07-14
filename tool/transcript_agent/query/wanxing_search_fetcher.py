from __future__ import annotations

from typing import Callable, List, Optional

from transcript_agent.query.models import FactBundle, FactItem, QueryContext
from transcript_agent.query.research_query_plan import build_search_plan, spec_to_payload
from transcript_agent.query.wanxing_search_client import WanxingSearchClient

ProgressCallback = Callable[[str], None]


class WanxingSearchFetcher:
    def __init__(self, config: Optional[dict] = None) -> None:
        from wanxing_config import load_wanxing_config

        self.config = config or load_wanxing_config()
        self.api_key = (self.config.get("wanxing_api_key") or "").strip()
        self.api_url = (self.config.get("wanxing_search_api_url") or WanxingSearchClient.DEFAULT_URL).strip()
        self.timeout = int(self.config.get("wanxing_search_timeout_sec") or 120)

    def fetch(
        self,
        query: QueryContext,
        *,
        on_progress: Optional[ProgressCallback] = None,
    ) -> FactBundle:
        from transcript_agent.query.wanxing_fetcher import (
            WanxingMcpFetcher,
            _degraded_bundle,
            _resolve_code_for_query,
        )

        def prog(msg: str) -> None:
            if on_progress:
                on_progress(msg)

        if not self.api_key:
            bundle = _degraded_bundle(query, "未配置 WANXING_API_KEY", mode="wanxing_search")
            return bundle

        client = WanxingSearchClient(self.api_key, api_url=self.api_url, timeout=self.timeout)
        research_calls: List[dict] = []
        raw_sections: dict = {}
        facts = []

        if query.subject_type != "sector":
            prog("万行名称转代码...")
            stock_code, resolve_record = _resolve_code_for_query(query, WanxingMcpFetcher(self.config))
            research_calls.append(resolve_record)

        specs = build_search_plan(query)
        total = len(specs)
        errors: List[str] = []

        for idx, spec in enumerate(specs, start=1):
            prog(f"万行金融搜索 ({idx}/{total}): {spec.intent}...")
            payload = spec_to_payload(spec)
            call_record = {
                "intent": spec.intent,
                "tool_name": "web_search",
                "arguments": payload,
            }
            try:
                resp = client.search(
                    spec.query,
                    count=spec.count,
                    freshness=spec.freshness,
                    domains=spec.domains,
                )
                call_record["response"] = resp.raw or {"results_count": len(resp.results)}
                call_record["error"] = resp.error
                if resp.error:
                    errors.append(f"{spec.intent}: {resp.error}")
                else:
                    text = resp.combined_text()
                    raw_sections[spec.intent] = text
                    if text.strip():
                        facts.append(
                            FactItem(
                                fact_type=spec.intent,
                                text=text[:4000],
                                source="wanxing_search",
                            )
                        )
            except Exception as exc:
                call_record["response"] = None
                call_record["error"] = str(exc)
                errors.append(f"{spec.intent}: {exc}")
            research_calls.append(call_record)

        if not facts:
            bundle = _degraded_bundle(
                query,
                "; ".join(errors) if errors else "金融搜索无返回数据",
                mode="wanxing_search",
            )
            bundle.wind_calls = research_calls
            bundle.research_raw_sections = raw_sections
            return bundle

        summary = _build_summary(query, facts, raw_sections)
        return FactBundle(
            subject=query.subject,
            facts=facts,
            summary_for_match=summary,
            wind_status="ok",
            research_mode="wanxing_search",
            wind_calls=research_calls,
            research_raw_sections=raw_sections,
        )


def _build_summary(query: QueryContext, facts, raw_sections: dict) -> str:
    parts = [f"标的:{query.subject}", f"类型:{query.subject_type}"]
    if query.stock_code:
        parts.append(f"代码:{query.stock_code}")
    if query.sector_hint:
        parts.append(f"板块:{query.sector_hint}")
    for fact in facts[:10]:
        parts.append(f"[{fact.fact_type}] {fact.text[:120]}")
    return "；".join(parts)[:1500]
