from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, Generator, List

logger = logging.getLogger(__name__)


class TimingCollector:
    def __init__(self) -> None:
        self.steps: List[Dict[str, Any]] = []

    @contextmanager
    def step(self, name: str) -> Generator[None, None, None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            entry = {"name": name, "seconds": round(elapsed, 2)}
            self.steps.append(entry)
            logger.info("[transcript_agent] %s %.2fs", name, elapsed)

    def to_dict(self) -> Dict[str, Any]:
        total = round(sum(s["seconds"] for s in self.steps), 2)
        return {"total_seconds": total, "steps": self.steps}
