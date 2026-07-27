from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from transcript_agent.query.models import QueryContext


@dataclass
class SearchQuerySpec:
    intent: str
    query: str
    freshness: str = ""
    count: int = 8
    domains: str = ""


STOCK_SEARCH_INTENTS = (
    "business_mix",
    "balance_sheet",
    "announcements",
    "broker_research",
)

SECTOR_SEARCH_INTENTS = (
    "sector_overview",
    "sector_catalysts",
)


def build_stock_search_plan(query: QueryContext) -> List[SearchQuerySpec]:
    subject = query.subject.strip()
    code = (query.stock_code or "").strip()
    label = f"{subject}({code})" if code else subject
    return [
        SearchQuerySpec(
            intent="business_mix",
            query=f"{label} 主营业务构成、营收占比、各业务板块",
            freshness="oneyear",
            count=8,
        ),
        SearchQuerySpec(
            intent="balance_sheet",
            query=f"{label} 最新资产负债表 流动资产 固定资产 流动负债 非流动负债 有息负债 商誉",
            freshness="oneyear",
            count=8,
        ),
        SearchQuerySpec(
            intent="announcements",
            query=f"{label} 近期重大事项公告 业绩预告 监管问询",
            freshness="onemonth",
            count=8,
        ),
        SearchQuerySpec(
            intent="broker_research",
            query=f"{label} 近6个月券商研报 核心观点 评级变化",
            freshness="onemonth",
            count=8,
        ),
    ]


def build_sector_search_plan(query: QueryContext) -> List[SearchQuerySpec]:
    sector = (query.sector_hint or query.subject).strip()
    return [
        SearchQuerySpec(
            intent="sector_overview",
            query=f"{sector} 行业景气度 政策 龙头 估值水平",
            freshness="onemonth",
            count=10,
        ),
        SearchQuerySpec(
            intent="sector_catalysts",
            query=f"{sector} 近期催化 风险 资金流向",
            freshness="onemonth",
            count=8,
        ),
    ]


def build_search_plan(query: QueryContext) -> List[SearchQuerySpec]:
    if query.subject_type == "sector":
        return build_sector_search_plan(query)
    return build_stock_search_plan(query)


def spec_to_payload(spec: SearchQuerySpec) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "query": spec.query,
        "count": spec.count,
        "market": "zh-CN",
        "set_lang": "zh",
    }
    if spec.freshness:
        payload["freshness"] = spec.freshness
    if spec.domains:
        payload["domains"] = spec.domains
    return payload
