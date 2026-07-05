from __future__ import annotations

from typing import Callable, Optional

from transcript_agent.query.models import FactBundle, QueryContext
from transcript_agent.query.wanxing_fetcher import WanxingMcpFetcher
from transcript_agent.query.wanxing_search_fetcher import WanxingSearchFetcher
from transcript_agent.query.wind_alice_client import WindAliceClient
from transcript_agent.query.wind_search_fetcher import WindSearchFetcher
from wanxing_config import wanxing_research_mode

ProgressCallback = Callable[[str], None]


def fetch_research_facts(
    query: QueryContext,
    *,
    deep_research: bool = False,
    context: str = "",
    on_progress: Optional[ProgressCallback] = None,
) -> FactBundle:
    if deep_research:
        structured = WindSearchFetcher().fetch(query, on_progress=on_progress)
        return WindAliceClient().research(
            query,
            context=context,
            on_progress=on_progress,
            structured_facts=structured,
        )
    mode = wanxing_research_mode()
    if mode == "mcp":
        return WanxingMcpFetcher().fetch(query, on_progress=on_progress)
    return WanxingSearchFetcher().fetch(query, on_progress=on_progress)
