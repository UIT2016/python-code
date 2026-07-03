from __future__ import annotations

from typing import Callable, Optional

from transcript_agent.query.models import FactBundle, QueryContext
from transcript_agent.query.wanxing_fetcher import WanxingFetcher
from transcript_agent.query.wind_alice_client import WindAliceClient

ProgressCallback = Callable[[str], None]


def fetch_research_facts(
    query: QueryContext,
    *,
    deep_research: bool = False,
    context: str = "",
    on_progress: Optional[ProgressCallback] = None,
) -> FactBundle:
    if deep_research:
        return WindAliceClient().research(query, context=context, on_progress=on_progress)
    return WanxingFetcher().fetch(query, on_progress=on_progress)
