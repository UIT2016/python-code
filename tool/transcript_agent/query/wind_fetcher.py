from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional

import requests

from transcript_agent.query.models import FactBundle, FactItem, QueryContext

_SSE_BLOCK_RE = re.compile(r"\r?\n\r?\n")

# 与 AIFin Market wind-mcp-skill 文档一致：各 server_type 直连 mcp.wind.com.cn
DEFAULT_MCP_SERVERS: Dict[str, str] = {
    "stock_data": "https://mcp.wind.com.cn/vserver_stock_data/mcp/",
    "fund_data": "https://mcp.wind.com.cn/vserver_fund_data/mcp/",
    "index_data": "https://mcp.wind.com.cn/vserver_index_data/mcp/",
    "bond_data": "https://mcp.wind.com.cn/vserver_bond_data/mcp/",
    "financial_docs": "https://mcp.wind.com.cn/vserver_financial_docs/mcp/",
    "economic_data": "https://mcp.wind.com.cn/vserver_economic_data/mcp/",
    "analytics_data": "https://mcp.wind.com.cn/vserver_analytics_data/mcp/",
}

# tool_name / 参数 key 见 Wind references/tool-contracts.md
INTENT_TOOLS = {
    "stock_profile": (
        "stock_data",
        "get_stock_basicinfo",
        {"question": "{subject}公司基本档案"},
    ),
    "recent_news": (
        "financial_docs",
        "get_financial_news",
        {"query": "{subject_nospace}", "top_k": 5},
    ),
    "financial_summary": (
        "stock_data",
        "get_stock_fundamentals",
        {"question": "{subject}最近一期财报ROE和净利润"},
    ),
    "sector_context": (
        "index_data",
        "get_index_basicinfo",
        {"question": "{sector}指数档案"},
    ),
    "macro_policy": (
        "economic_data",
        "natural_language_get_edb_data",
        {
            "executionMode": "searchFetch",
            "question": "{subject_nospace}宏观政策",
            "observation": "10",
        },
    ),
}


def _format_tool_args(arg_tpl: Dict[str, Any], query: QueryContext) -> Dict[str, Any]:
    sector = query.sector_hint or query.subject
    subject_nospace = re.sub(r"\s+", "", query.subject)
    fmt_vars = {
        "subject": query.subject,
        "sector": sector,
        "subject_nospace": subject_nospace,
    }
    args: Dict[str, Any] = {}
    for key, value in arg_tpl.items():
        if isinstance(value, str):
            args[key] = value.format(**fmt_vars)
        else:
            args[key] = value
    return args


def json_dumps_safe(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except TypeError:
        return str(obj)


def _collect_sse_data_payloads(text: str) -> List[str]:
    payloads: List[str] = []
    for block in _SSE_BLOCK_RE.split(text):
        block = block.strip()
        if not block:
            continue
        lines = block.splitlines()
        idx = 0
        while idx < len(lines):
            line = lines[idx]
            if not line.startswith("data:"):
                idx += 1
                continue
            chunk = line[5:]
            if chunk.startswith(" "):
                chunk = chunk[1:]
            idx += 1
            while idx < len(lines):
                nxt = lines[idx]
                if nxt.startswith("data:") or nxt.startswith("event:") or nxt.startswith(":"):
                    break
                chunk += nxt
                idx += 1
            chunk = chunk.strip()
            if chunk and chunk != "[DONE]":
                payloads.append(chunk)
    return payloads


def _parse_sse_or_json(text: str) -> Dict[str, Any]:
    for payload in reversed(_collect_sse_data_payloads(text)):
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and ("result" in parsed or "error" in parsed):
            return parsed
    text = text.strip()
    if text:
        return json.loads(text)
    raise ValueError("Wind MCP 响应为空")


class WindAIFinClient:
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        from wind_config import load_wind_config

        self.config = config or load_wind_config()
        self.api_key = (self.config.get("wind_api_key") or "").strip()
        self.servers = dict(self.config.get("wind_mcp_servers") or DEFAULT_MCP_SERVERS)

    def fetch(self, query: QueryContext) -> FactBundle:
        wind_calls: List[Dict[str, Any]] = []
        if not self.api_key:
            bundle = self._degraded(query, "未配置 WIND_API_KEY")
            bundle.wind_calls = wind_calls
            return bundle

        facts: List[FactItem] = []
        errors: List[str] = []
        for intent in query.wind_intents:
            spec = INTENT_TOOLS.get(intent)
            if not spec:
                continue
            server_type, tool_name, arg_tpl = spec
            args = _format_tool_args(arg_tpl, query)
            endpoint = self.servers.get(server_type)
            call_record: Dict[str, Any] = {
                "intent": intent,
                "server_type": server_type,
                "tool_name": tool_name,
                "endpoint": endpoint,
                "arguments": args,
            }
            try:
                payload = self._call_tool(server_type, tool_name, args)
                call_record["response"] = payload
                call_record["error"] = None
                facts.extend(self._parse_tool_response(intent, payload))
            except Exception as exc:
                call_record["response"] = None
                call_record["error"] = str(exc)
                errors.append(f"{intent}: {exc}")
            wind_calls.append(call_record)

        if not facts:
            bundle = self._degraded(query, "; ".join(errors) if errors else "Wind 无返回数据")
            bundle.wind_calls = wind_calls
            return bundle

        summary = self._build_summary(query, facts)
        return FactBundle(
            subject=query.subject,
            facts=facts,
            summary_for_match=summary,
            wind_status="ok",
            wind_calls=wind_calls,
        )

    def _call_tool(self, server_type: str, tool_name: str, arguments: Dict[str, Any]) -> Any:
        endpoint = self.servers.get(server_type)
        if not endpoint:
            raise RuntimeError(f"未知 server_type: {server_type}")

        self._mcp_request(
            endpoint,
            "initialize",
            {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "python-code-transcript-agent", "version": "1.0"},
            },
            timeout=30,
        )
        result = self._mcp_request(
            endpoint,
            "tools/call",
            {
                "name": tool_name,
                "arguments": arguments,
                "_meta": {"clientVersion": "1.0"},
            },
            timeout=120,
        )
        return self._unwrap_tool_result(result)

    def _mcp_request(self, endpoint: str, method: str, params: Dict[str, Any], *, timeout: int) -> Any:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        }
        body = {"jsonrpc": "2.0", "id": int(time.time() * 1000), "method": method, "params": params}
        resp = requests.post(endpoint, headers=headers, json=body, timeout=timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        payload = _parse_sse_or_json(resp.text)
        if payload.get("error"):
            err = payload["error"]
            msg = err.get("message") if isinstance(err, dict) else str(err)
            raise RuntimeError(msg or json_dumps_safe(err))
        result = payload.get("result")
        if isinstance(result, dict) and result.get("isError"):
            content = result.get("content") or []
            msg = content[0].get("text") if content and isinstance(content[0], dict) else json_dumps_safe(result)
            raise RuntimeError(str(msg)[:500])
        return result

    @staticmethod
    def _unwrap_tool_result(result: Any) -> Any:
        if not isinstance(result, dict):
            return result
        content = result.get("content")
        if not isinstance(content, list) or not content:
            return result
        first = content[0]
        if not isinstance(first, dict):
            return result
        text = first.get("text")
        if not isinstance(text, str) or not text.strip():
            return result
        try:
            inner = json.loads(text)
        except json.JSONDecodeError:
            return text
        if isinstance(inner, dict):
            if inner.get("mcp_tool_error_code") not in (None, 0):
                raise RuntimeError(inner.get("mcp_tool_error_msg") or json_dumps_safe(inner))
            if inner.get("error"):
                err = inner["error"]
                if isinstance(err, dict):
                    code = err.get("code") or ""
                    message = err.get("message") or ""
                    raise RuntimeError(f"{code}: {message}".strip(": ") or json_dumps_safe(err))
        return inner

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
                if items:
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
