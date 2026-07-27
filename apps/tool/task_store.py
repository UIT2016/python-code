"""内存任务状态存储，供 Web 端轮询进度。"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional


class TaskStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tasks: Dict[str, Dict[str, Any]] = {}

    def create(self, task_type: str, *, meta: Optional[Dict[str, Any]] = None) -> str:
        task_id = uuid.uuid4().hex
        now = time.time()
        with self._lock:
            self._tasks[task_id] = {
                "id": task_id,
                "type": task_type,
                "label": (meta or {}).get("label", ""),
                "status": "running",
                "progress": 0,
                "message": "准备中...",
                "elapsed_sec": 0.0,
                "started_at": now,
                "finished_at": None,
                "result": None,
                "error": None,
                "items": meta.get("items", []) if meta else [],
            }
        return task_id

    def update(
        self,
        task_id: str,
        *,
        progress: Optional[int] = None,
        message: Optional[str] = None,
        status: Optional[str] = None,
        result: Any = None,
        error: Optional[str] = None,
        items: Optional[List[Any]] = None,
    ) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return
            if progress is not None:
                task["progress"] = max(0, min(100, int(progress)))
            if message is not None:
                task["message"] = message
            if status is not None:
                task["status"] = status
                if status in ("done", "error"):
                    task["finished_at"] = time.time()
            if result is not None:
                task["result"] = result
            if error is not None:
                task["error"] = error
            if items is not None:
                task["items"] = items
            task["elapsed_sec"] = round(time.time() - task["started_at"], 1)

    def get(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return None
            out = dict(task)
            out["elapsed_sec"] = round(time.time() - task["started_at"], 1)
            return out

    def make_updater(self, task_id: str) -> Callable[[int, str], None]:
        def _update(progress: int, message: str) -> None:
            self.update(task_id, progress=progress, message=message)

        return _update


task_store = TaskStore()
