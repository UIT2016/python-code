from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests

from transcript_agent.query.models import FactBundle, FactItem, QueryContext

INTENT_TOOLS = {
    "stock_profile": ("stock_data", "get_stock_basicinfo", {"keyword": "{subject}"}),
    "recent_news": ("financial_docs", "search_financial_news", {"keyword": "{subject}", "limit": 5}),
    "financial_summary": ("stock_data", "get_stock_financial_indicators", {"keyword": "{subject}"}),
    "sector_context": ("index_data", "get_index_basicinfo", {"keyword": "{sector}"}),
    "macro_policy": ("economic_data", "search_edb_indicators", {"keyword": "{subject}"}),
}


class WindAIFinClient:
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        from wind_config import load_wind_config

        self.config = config or load_wind_config()
        self.api_key = (self.config.get("wind_api_key") or "").strip()
        base = (self.config.get("wind_api_base") or "https://aifinmarket.wind.com.cn").rstrip("/")
        path = self.config.get("wind_mcp_path") or "/mcp/v1/tools/call"
        self.endpoint = base + path

    def fetch(self, query: QueryContext) -> FactBundle:
        if not self.api_key:
            return self._degraded(query, "未配置 WIND_API_KEY")

        facts: List[FactItem] = []
        errors: List[str] = []
        for intent in query.wind_intents:
            spec = INTENT_TOOLS.get(intent)
            if not spec:
                continue
            server_type, tool_name, arg_tpl = spec
            args = {
                k: v.format(subject=query.subject, sector=query.sector_hint or query.subject)
                for k, v in arg_tpl.items()
            }
            try:
                payload = self._call_tool(server_type, tool_name, args)
                facts.extend(self._parse_tool_response(intent, payload))
            except Exception as exc:
                errors.append(f"{intent}: {exc}")

        if not facts:
            return self._degraded(query, "; ".join(errors) if errors else "Wind 无返回数据")

        summary = self._build_summary(query, facts)
        return FactBundle(subject=query.subject, facts=facts, summary_for_match=summary, wind_status="ok")

    def _call_tool(self, server_type: str, tool_name: str, arguments: Dict[str, Any]) -> Any:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {"server_type": server_type, "tool_name": tool_name, "arguments": arguments}
        resp = requests.post(self.endpoint, headers=headers, json=body, timeout=90)
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        if isinstance(data, dict) and data.get("error"):
            raise RuntimeError(str(data.get("error")))
        return data

    @staticmethod
    def _parse_tool_response(intent: str, payload: Any) -> List[FactItem]:
        items: List[FactItem] = []
        if isinstance(payload, str):
            if payload.strip():
                items.append(FactItem(fact_type=intent, text=payload.strip()[:2000]))
            return items
        if isinstance(payload, dict):
            text = payload.get("text") or payload.get("content") or payload.get("result")
            if isinstance(text, str) and text.strip():
                items.append(FactItem(fact_type=intent, text=text.strip()[:2000]))
                return items
            rows = payload.get("data") or payload.get("items") or payload.get("records")
            if isinstance(rows, list):
                for row in rows[:8]:
                    if isinstance(row, dict):
                        line = " | ".join(f"{k}:{v}" for k, v in row.items() if v not in (None, ""))
                        if line:
                            items.append(FactItem(fact_type=intent, text=line[:500]))
                    elif isinstance(row, str):
                        items.append(FactItem(fact_type=intent, text=row[:500]))
                return items
            compact = json_dumps_safe(payload)
            if compact:
                items.append(FactItem(fact_type=intent, text=compact[:2000]))
        elif isinstance(payload, list):
            for row in payload[:8]:
                items.append(FactItem(fact_type=intent, text=str(row)[:500]))
        return items

    @staticmethod
    def _build_summary(query: QueryContext, facts: List[FactItem]) -> str:
        parts = [f"标的:{query.subject}", f"类型:{query.subject_type}"]
        if query.sector_hint:
            parts.append(f"板块:{query.sector_hint}")
        if query.event_keywords:
            parts.append("事件:" + ",".join(query.event_keywords))
        for fact in facts[:6]:
            parts.append(f"[{fact.fact_type}] {fact.text[:120]}")
        text = "；".join(parts)
        return text[:800]

    @staticmethod
    def _degraded(query: QueryContext, reason: str) -> FactBundle:
        summary_parts = [f"标的:{query.subject}", f"查询:{query.raw_query}"]
        if query.event_keywords:
            summary_parts.append("事件:" + ",".join(query.event_keywords))
        summary_parts.append(f"Wind降级:{reason}")
        return FactBundle(
            subject=query.subject,
            facts=[FactItem(fact_type="query", text=query.raw_query, source="user")],
            summary_for_match="；".join(summary_parts)[:800],
            wind_status="degraded",
        )


def json_dumps_safe(obj: Any) -> str:
    import json

    try:
        return json.dumps(obj, ensure_ascii=False)
    except TypeError:
        return str(obj)
