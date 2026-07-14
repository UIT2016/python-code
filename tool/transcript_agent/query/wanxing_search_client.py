"""万行一言金融搜索 API 客户端（web_search）。"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests


@dataclass
class SearchResultItem:
    title: str
    url: str
    context: str
    pub_time: str = ""
    meta_type: str = ""
    score: float = 0.0

    def to_fact_text(self) -> str:
        parts = [self.title]
        if self.pub_time:
            parts.append(f"时间:{self.pub_time}")
        if self.context:
            parts.append(self.context.strip()[:1500])
        if self.url:
            parts.append(f"来源:{self.url}")
        return " | ".join(parts)


@dataclass
class SearchResponse:
    query: str
    results: List[SearchResultItem] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return not self.error and bool(self.results)

    def combined_text(self, *, max_chars: int = 6000) -> str:
        chunks: List[str] = []
        for item in self.results:
            text = item.to_fact_text()
            if text.strip():
                chunks.append(text.strip())
        return "\n\n".join(chunks)[:max_chars]


class WanxingSearchClient:
    DEFAULT_URL = "https://api.wanxingai.com/v1/web-search/"

    def __init__(
        self,
        api_key: str,
        *,
        api_url: Optional[str] = None,
        timeout: int = 120,
    ) -> None:
        self.api_key = api_key.strip()
        self.api_url = (api_url or self.DEFAULT_URL).strip()
        self.timeout = timeout

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def search(
        self,
        query: str,
        *,
        count: int = 8,
        freshness: str = "",
        domains: str = "",
        market: str = "zh-CN",
        set_lang: str = "zh",
    ) -> SearchResponse:
        payload: Dict[str, Any] = {
            "query": query,
            "count": count,
            "market": market,
            "set_lang": set_lang,
        }
        if freshness:
            payload["freshness"] = freshness
        if domains:
            payload["domains"] = domains

        resp = requests.post(
            self.api_url,
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
        )
        if resp.status_code != 200:
            return SearchResponse(
                query=query,
                error=f"HTTP {resp.status_code}: {resp.text[:300]}",
            )
        try:
            data = resp.json()
        except json.JSONDecodeError:
            return SearchResponse(query=query, error="响应非 JSON")

        status = data.get("status")
        if status != 200:
            err = data.get("error") or data.get("message") or str(data)
            return SearchResponse(query=query, raw=data, error=str(err)[:500])

        message = data.get("message") or {}
        rows = message.get("results") if isinstance(message, dict) else message
        items: List[SearchResultItem] = []
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                items.append(
                    SearchResultItem(
                        title=str(row.get("page_title") or row.get("title") or ""),
                        url=str(row.get("page_url") or row.get("url") or ""),
                        context=str(row.get("context") or row.get("snippet") or ""),
                        pub_time=str(row.get("page_pub_time") or row.get("pub_time") or ""),
                        meta_type=str(row.get("meta_type") or ""),
                        score=float(row.get("score") or 0),
                    )
                )
        return SearchResponse(query=query, results=items, raw=data)
