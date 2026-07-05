"""Wind MCP 取数，复用与万行金融搜索相同的检索要素模板。"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from transcript_agent.query.models import FactBundle, FactItem, QueryContext
from transcript_agent.query.research_query_plan import build_search_plan
from transcript_agent.query.wind_fetcher import WindAIFinClient, json_dumps_safe

ProgressCallback = Callable[[str], None]

# research intent -> (wind_intent for WindAIFinClient)
_WIND_INTENT_MAP = {
    "business_mix": "financial_summary",
    "balance_sheet": "financial_summary",
    "announcements": "recent_news",
    "broker_research": "recent_news",
    "sector_overview": "sector_context",
    "sector_catalysts": "macro_policy",
}


class WindSearchFetcher:
    def __init__(self, config: Optional[dict] = None) -> None:
        self._client = WindAIFinClient(config)

    def fetch(
        self,
        query: QueryContext,
        *,
        on_progress: Optional[ProgressCallback] = None,
    ) -> FactBundle:
        def prog(msg: str) -> None:
            if on_progress:
                on_progress(msg)

        if not self._client.api_key:
            return self._degraded(query, "未配置 WIND_API_KEY")

        specs = build_search_plan(query)
        research_calls: List[Dict[str, Any]] = []
        raw_sections: Dict[str, str] = {}
        facts: List[FactItem] = []
        errors: List[str] = []
        total = len(specs)

        for idx, spec in enumerate(specs, start=1):
            wind_intent = _WIND_INTENT_MAP.get(spec.intent, "stock_profile")
            prog(f"Wind 检索 ({idx}/{total}): {spec.intent}...")
            call_record: Dict[str, Any] = {
                "intent": spec.intent,
                "wind_intent": wind_intent,
                "tool_name": "wind_mcp",
                "arguments": {"search_query": spec.query},
            }
            try:
                payload = self._call_by_search_query(query, spec.intent, spec.query, wind_intent)
                text = self._payload_to_text(payload)
                call_record["response"] = payload
                call_record["error"] = None
                if text.strip():
                    raw_sections[spec.intent] = text
                    facts.append(FactItem(fact_type=spec.intent, text=text[:4000], source="wind_search"))
            except Exception as exc:
                call_record["response"] = None
                call_record["error"] = str(exc)
                errors.append(f"{spec.intent}: {exc}")
            research_calls.append(call_record)

        if not facts:
            bundle = self._degraded(query, "; ".join(errors) if errors else "Wind 检索无数据")
            bundle.wind_calls = research_calls
            bundle.research_raw_sections = raw_sections
            return bundle

        summary_parts = [f"标的:{query.subject}", f"类型:{query.subject_type}"]
        if query.stock_code:
            summary_parts.append(f"代码:{query.stock_code}")
        for fact in facts[:8]:
            summary_parts.append(f"[{fact.fact_type}] {fact.text[:120]}")
        return FactBundle(
            subject=query.subject,
            facts=facts,
            summary_for_match="；".join(summary_parts)[:1200],
            wind_status="ok",
            research_mode="wind_search",
            wind_calls=research_calls,
            research_raw_sections=raw_sections,
        )

    def _call_by_search_query(
        self,
        query: QueryContext,
        intent: str,
        search_query: str,
        wind_intent: str,
    ) -> object:
        from transcript_agent.query.wind_fetcher import INTENT_TOOLS, _format_tool_args

        spec = INTENT_TOOLS.get(wind_intent)
        if not spec:
            raise RuntimeError(f"未知 wind_intent: {wind_intent}")
        server_type, tool_name, arg_tpl = spec
        args = _format_tool_args(arg_tpl, query)
        for key in ("question", "query"):
            if key in args:
                args[key] = search_query
        if wind_intent == "recent_news" and "query" in args:
            args["query"] = search_query.replace(" ", "")
        endpoint = self._client.servers.get(server_type)
        self._client._mcp_request(
            endpoint,
            "initialize",
            {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "python-code-transcript-agent", "version": "1.0"},
            },
            timeout=30,
        )
        result = self._client._mcp_request(
            endpoint,
            "tools/call",
            {"name": tool_name, "arguments": args, "_meta": {"clientVersion": "1.0"}},
            timeout=120,
        )
        return self._client._unwrap_tool_result(result)

    @staticmethod
    def _payload_to_text(payload: object) -> str:
        if isinstance(payload, str):
            return payload.strip()
        if isinstance(payload, (dict, list)):
            return json_dumps_safe(payload)
        return str(payload)

    @staticmethod
    def _degraded(query: QueryContext, reason: str) -> FactBundle:
        return FactBundle(
            subject=query.subject,
            facts=[FactItem(fact_type="query", text=query.raw_query, source="user")],
            summary_for_match=f"标的:{query.subject}；Wind降级:{reason}",
            wind_status="degraded",
            research_mode="wind_search",
        )
