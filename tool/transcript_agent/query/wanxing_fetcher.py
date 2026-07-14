from __future__ import annotations

import json
import re
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

from transcript_agent.query.models import FactBundle, FactItem, QueryContext
from transcript_agent.query.wanxing_mcp_client import WanxingMcpClient

ProgressCallback = Callable[[str], None]

_STOCK_CODE_RE = re.compile(r"^\d{6}\.[A-Z]{2,8}$", re.I)

INTENT_TOOLS: Dict[str, tuple[str, Dict[str, Any]]] = {
    "stock_profile": ("get_listed_company_info", {"stock_code": "{stock_code}"}),
    "recent_news": (
        "query_announcement",
        {"codes": "{stock_code}", "outputpara": "title,pubdate", "report_type": "903"},
    ),
    "financial_summary": ("get_stock_financial_data", {"stock_code": "{stock_code}"}),
    "sector_context": ("smart_stock_picking", {"searchstring": "{sector}板块龙头"}),
    "macro_policy": ("smart_stock_picking", {"searchstring": "{subject}宏观政策影响"}),
}


def _is_valid_stock_code(code: str) -> bool:
    return bool(_STOCK_CODE_RE.match((code or "").strip()))


def _market_code(stock_code: str) -> str:
    return "212001" if stock_code.upper().endswith(".SH") else "212100"


def _extract_code_from_search(payload: Any) -> str:
    if not isinstance(payload, list):
        return ""
    for row in payload:
        if not isinstance(row, dict):
            continue
        table = row.get("table") or {}
        thscode = table.get("thscode")
        if isinstance(thscode, list):
            for code in thscode:
                if code:
                    candidate = str(code).split(",")[0].strip()
                    if _is_valid_stock_code(candidate):
                        return candidate
        elif thscode:
            candidate = str(thscode).split(",")[0].strip()
            if _is_valid_stock_code(candidate):
                return candidate
        direct = row.get("thscode")
        if direct:
            candidate = str(direct).split(",")[0].strip()
            if _is_valid_stock_code(candidate):
                return candidate
    return ""


def _code_from_get_stock_payload(payload: Any) -> str:
    if isinstance(payload, dict):
        code = str(payload.get("stock_code") or payload.get("thscode") or "")
        if _is_valid_stock_code(code):
            return code
    return ""


def resolve_stock_code_by_name(
    subject: str,
    *,
    client: Optional[WanxingMcpClient] = None,
) -> Tuple[str, Dict[str, Any]]:
    """公司名 -> 标准股票代码。优先 get_stock_code，失败时用 search_securities_code 兜底。"""
    from wanxing_config import load_wanxing_config

    name = subject.strip()
    get_stock_args = {"stock_name": name}
    if client is None:
        cfg = load_wanxing_config()
        api_key = (cfg.get("wanxing_api_key") or "").strip()
        mcp_url = (cfg.get("wanxing_mcp_url") or "https://mcp.wanxingai.com/sse").strip()
        if not api_key:
            return "", {
                "intent": "resolve_code",
                "tool_name": "get_stock_code",
                "arguments": get_stock_args,
                "response": None,
                "stock_code": None,
                "error": "未配置 wanxing_api_key",
            }
        client = WanxingMcpClient(mcp_url, api_key)

    resolve_trace: Dict[str, Any] = {}
    try:
        payload = client.call_tool("get_stock_code", get_stock_args)
        resolve_trace["get_stock_code"] = payload
        code = _code_from_get_stock_payload(payload)
        if code:
            return code, {
                "intent": "resolve_code",
                "tool_name": "get_stock_code",
                "arguments": get_stock_args,
                "response": payload,
                "stock_code": code,
                "error": None,
            }
    except Exception as exc:
        resolve_trace["get_stock_code"] = {"error": str(exc)}

    search_variants = [
        {"query": name, "mode": "secname", "isexact": "0"},
        {"query": re.sub(r"\s+", "", name), "mode": "secname", "isexact": "0"},
    ]
    code_match = re.search(r"\b(\d{6})\b", name)
    if code_match:
        search_variants.insert(0, {"query": code_match.group(1), "mode": "seccode", "sectype": "001"})

    for attempt, search_args in enumerate(search_variants):
        try:
            payload = client.call_tool("search_securities_code", search_args)
            resolve_trace["search_securities_code"] = {"search_args": search_args, "result": payload}
            code = _extract_code_from_search(payload)
            if code:
                return code, {
                    "intent": "resolve_code",
                    "tool_name": "get_stock_code",
                    "arguments": get_stock_args,
                    "response": resolve_trace,
                    "fallback_tool": "search_securities_code",
                    "stock_code": code,
                    "error": None,
                }
        except Exception as exc:
            resolve_trace.setdefault("search_attempts", []).append(
                {"search_args": search_args, "error": str(exc)}
            )
        if attempt == 0:
            time.sleep(0.3)

    return "", {
        "intent": "resolve_code",
        "tool_name": "get_stock_code",
        "arguments": get_stock_args,
        "response": resolve_trace,
        "stock_code": None,
        "error": "未能将公司名称解析为标准股票代码",
    }


def apply_stock_code_from_query(query: QueryContext) -> QueryContext:
    """若用户输入已含标准代码，直接写入 query.stock_code。"""
    subject = query.subject.strip()
    if _is_valid_stock_code(subject):
        query.stock_code = subject
        return query
    m = re.search(r"\b(\d{6})\.(SH|SZ|BJ)\b", subject, re.I)
    if m:
        query.stock_code = f"{m.group(1)}.{m.group(2).upper()}"
    return query


def _build_stock_tool_calls(query: QueryContext, stock_code: str) -> List[Tuple[str, str, Dict[str, Any]]]:
    now = datetime.now()
    end = now.strftime("%Y-%m-%d")
    start_30 = (now - timedelta(days=30)).strftime("%Y-%m-%d")
    start_90 = (now - timedelta(days=90)).strftime("%Y-%m-%d")
    start_365 = (now - timedelta(days=365)).strftime("%Y-%m-%d")
    today_open = now.strftime("%Y-%m-%d 09:30:00")
    today_now = now.strftime("%Y-%m-%d %H:%M:%S")
    market = _market_code(stock_code)
    quote_indicators = "preClose,open,high,low,latest,amount,volume,changeRatio,turnoverRatio"

    calls: List[Tuple[str, str, Dict[str, Any]]] = [
        ("basic_info", "get_stock_basic_info", {"stock_code": stock_code}),
        ("company_profile", "get_listed_company_info", {"stock_code": stock_code}),
        ("shareholder", "get_stock_equity_shareholder", {"stock_code": stock_code}),
        ("financial_data", "get_stock_financial_data", {"stock_code": stock_code}),
        ("daily_tech", "get_stock_daily_quotes_tech", {"stock_code": stock_code}),
        (
            "realtime_quote",
            "get_stock_real_time_quotation",
            {"stock_code": stock_code, "indicators": quote_indicators},
        ),
        (
            "history_quote_30d",
            "get_stock_history_quotation",
            {
                "stock_code": stock_code,
                "indicators": "open,high,low,latest,volume,changeRatio",
                "startdate": start_30,
                "enddate": end,
            },
        ),
        (
            "history_quote_1y",
            "get_stock_history_quotation",
            {
                "stock_code": stock_code,
                "indicators": "open,high,low,latest,volume",
                "startdate": start_365,
                "enddate": end,
            },
        ),
        (
            "announcements",
            "query_announcement",
            {
                "codes": stock_code,
                "outputpara": "title,pubdate,abstract",
                "report_type": "903",
                "begin_date": start_90,
                "end_date": end,
            },
        ),
        (
            "announcements_reports",
            "query_announcement",
            {
                "codes": stock_code,
                "outputpara": "title,pubdate",
                "report_type": "901",
                "begin_date": start_365,
                "end_date": end,
            },
        ),
        (
            "intraday_snapshot",
            "get_intraday_snapshot",
            {
                "stock_code": stock_code,
                "indicators": "latest,volume,amount",
                "starttime": today_open,
                "endtime": today_now,
            },
        ),
        (
            "high_frequency",
            "get_high_frequency_quotes",
            {
                "stock_code": stock_code,
                "indicators": "close,volume",
                "starttime": today_open,
                "endtime": today_now,
                "interval": "5",
            },
        ),
        (
            "date_sequence",
            "get_date_sequence",
            {
                "stock_code": stock_code,
                "indicators": [{"indicator": "close"}, {"indicator": "volume"}],
                "startdate": start_30,
                "enddate": end,
            },
        ),
        (
            "trading_dates",
            "query_trading_dates",
            {"marketcode": market, "startdate": start_30, "enddate": end},
        ),
        (
            "trading_date_offset",
            "offset_trading_date",
            {"marketcode": market, "startdate": end, "offset": "-1"},
        ),
    ]
    sector = query.sector_hint or query.subject
    calls.append(("sector_peers", "smart_stock_picking", {"searchstring": f"{sector}板块龙头"}))
    for kw in query.event_keywords[:2]:
        calls.append(("event_screen", "smart_stock_picking", {"searchstring": f"{query.subject} {kw}"}))
    return calls


def json_dumps_safe(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except TypeError:
        return str(obj)


def _flatten_wanxing_record(row: Any) -> str:
    if not isinstance(row, dict):
        return str(row)[:500]
    nested = row.get("data")
    if isinstance(nested, list):
        parts: List[str] = []
        thscode = row.get("thscode")
        if thscode:
            parts.append(f"代码:{thscode}")
        for item in nested[:30]:
            if not isinstance(item, dict):
                continue
            label = item.get("description") or item.get("desc") or item.get("indicator") or ""
            value = item.get("value")
            if value in (None, ""):
                continue
            parts.append(f"{label}:{value}" if label else str(value))
        if parts:
            return " | ".join(parts)[:2000]
    table = row.get("table")
    if isinstance(table, dict):
        flat: List[str] = []
        for key, values in table.items():
            if isinstance(values, list):
                flat.append(f"{key}:{','.join(str(v) for v in values if v)}")
            elif values not in (None, ""):
                flat.append(f"{key}:{values}")
        if flat:
            return " | ".join(flat)[:2000]
    line = " | ".join(f"{k}:{v}" for k, v in row.items() if v not in (None, "", [], {}))
    return line[:2000] if line else json_dumps_safe(row)[:2000]


def _parse_tool_response(intent: str, payload: Any) -> List[FactItem]:
    items: List[FactItem] = []
    source = "wanxing"
    if isinstance(payload, str):
        if payload.strip():
            items.append(FactItem(fact_type=intent, text=payload.strip()[:2000], source=source))
        return items
    if isinstance(payload, dict):
        text = payload.get("text") or payload.get("content") or payload.get("result")
        if isinstance(text, str) and text.strip():
            items.append(FactItem(fact_type=intent, text=text.strip()[:2000], source=source))
            return items
        flat = _flatten_wanxing_record(payload)
        if flat:
            items.append(FactItem(fact_type=intent, text=flat, source=source))
            return items
        compact = json_dumps_safe(payload)
        if compact and compact not in ("{}", "[]"):
            items.append(FactItem(fact_type=intent, text=compact[:2000], source=source))
    elif isinstance(payload, list):
        for row in payload[:12]:
            flat = _flatten_wanxing_record(row)
            if flat:
                items.append(FactItem(fact_type=intent, text=flat[:2000], source=source))
        if not items and payload:
            items.append(FactItem(fact_type=intent, text=json_dumps_safe(payload)[:2000], source=source))
    return items


class WanxingMcpFetcher:
    """万行 MCP 结构化取数（后备模式）。"""
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        from wanxing_config import load_wanxing_config

        self.config = config or load_wanxing_config()
        self.api_key = (self.config.get("wanxing_api_key") or "").strip()
        self.mcp_url = (self.config.get("wanxing_mcp_url") or "https://mcp.wanxingai.com/sse").strip()
        self._client: Optional[WanxingMcpClient] = None

    def _client_or_raise(self) -> WanxingMcpClient:
        if not self.api_key:
            raise ValueError("未配置 wanxing_api_key，请在 tool/wanxing.local.json 填写")
        if self._client is None:
            self._client = WanxingMcpClient(self.mcp_url, self.api_key)
        return self._client

    def fetch(
        self,
        query: QueryContext,
        *,
        on_progress: Optional[ProgressCallback] = None,
    ) -> FactBundle:
        research_calls: List[Dict[str, Any]] = []

        def prog(msg: str) -> None:
            if on_progress:
                on_progress(msg)

        if not self.api_key:
            bundle = _degraded_bundle(query, "未配置 WANXING_API_KEY", mode="wanxing_mcp")
            bundle.research_mode = "wanxing_mcp"
            return bundle

        client = self._client_or_raise()
        facts: List[FactItem] = []
        errors: List[str] = []
        research_calls: List[Dict[str, Any]] = []

        if query.subject_type == "sector":
            sector = query.sector_hint or query.subject
            tool_calls: List[Tuple[str, str, Dict[str, Any]]] = [
                ("sector_overview", "smart_stock_picking", {"searchstring": f"{sector}板块龙头 景气度"}),
                ("sector_catalysts", "smart_stock_picking", {"searchstring": f"{sector}近期催化 风险"}),
            ]
        else:
            apply_stock_code_from_query(query)
            if not _is_valid_stock_code(query.stock_code):
                prog("万行名称转代码...")
            stock_code, research_calls = _resolve_code_for_query(query, self)
            if not _is_valid_stock_code(stock_code):
                bundle = _degraded_bundle(
                    query,
                    f"未能将「{query.subject}」解析为股票代码（如 600170.SH），公司简介/财务/公告等接口无法查询",
                    mode="wanxing_mcp",
                )
                bundle.wind_calls = research_calls
                return bundle
            tool_calls = _build_stock_tool_calls(query, stock_code)

        total = len(tool_calls)

        for idx, (intent, tool_name, args) in enumerate(tool_calls, start=1):
            call_record: Dict[str, Any] = {
                "intent": intent,
                "tool_name": tool_name,
                "arguments": args,
            }
            prog(f"万行取数 ({idx}/{total}): {intent}...")
            try:
                payload = client.call_tool(tool_name, args)
                call_record["response"] = payload
                call_record["error"] = None
                facts.extend(_parse_tool_response(intent, payload))
            except Exception as exc:
                call_record["response"] = None
                call_record["error"] = str(exc)
                errors.append(f"{intent}: {exc}")
            research_calls.append(call_record)

        if not facts:
            bundle = _degraded_bundle(query, "; ".join(errors) if errors else "万行无返回数据", mode="wanxing_mcp")
            bundle.wind_calls = research_calls
            return bundle

        summary = self._build_summary(
            query,
            facts,
            stock_code=query.stock_code if query.subject_type != "sector" else "",
        )
        return FactBundle(
            subject=query.subject,
            facts=facts,
            summary_for_match=summary,
            wind_status="ok",
            research_mode="wanxing_mcp",
            wind_calls=research_calls,
        )

    @staticmethod
    def _build_summary(query: QueryContext, facts: List[FactItem], *, stock_code: str = "") -> str:
        parts = [f"标的:{query.subject}", f"类型:{query.subject_type}"]
        if stock_code:
            parts.append(f"代码:{stock_code}")
        if query.sector_hint:
            parts.append(f"板块:{query.sector_hint}")
        if query.event_keywords:
            parts.append("事件:" + ",".join(query.event_keywords))
        for fact in facts[:12]:
            parts.append(f"[{fact.fact_type}] {fact.text[:120]}")
        return "；".join(parts)[:1200]


def _resolve_code_for_query(
    query: QueryContext,
    fetcher: "WanxingMcpFetcher",
) -> Tuple[str, List[Dict[str, Any]]]:
    research_calls: List[Dict[str, Any]] = []
    if _is_valid_stock_code(query.stock_code):
        stock_code = query.stock_code.strip()
        if query.stock_code_resolve:
            research_calls.append(query.stock_code_resolve)
        else:
            research_calls.append(
                {
                    "intent": "resolve_code",
                    "tool_name": "get_stock_code",
                    "arguments": {"stock_name": query.subject},
                    "response": {"stock_code": stock_code},
                    "stock_code": stock_code,
                    "error": None,
                }
            )
        return stock_code, research_calls
    client = fetcher._client_or_raise()
    stock_code, resolve_record = resolve_stock_code_by_name(query.subject, client=client)
    query.stock_code = stock_code
    query.stock_code_resolve = resolve_record
    research_calls.append(resolve_record)
    return stock_code, research_calls


def _degraded_bundle(query: QueryContext, reason: str, *, mode: str = "wanxing_mcp") -> FactBundle:
    summary_parts = [f"标的:{query.subject}", f"查询:{query.raw_query}"]
    if query.event_keywords:
        summary_parts.append("事件:" + ",".join(query.event_keywords))
    summary_parts.append(f"万行降级:{reason}")
    return FactBundle(
        subject=query.subject,
        facts=[FactItem(fact_type="query", text=query.raw_query, source="user")],
        summary_for_match="；".join(summary_parts)[:800],
        wind_status="degraded",
        research_mode=mode,
    )


WanxingFetcher = WanxingMcpFetcher
