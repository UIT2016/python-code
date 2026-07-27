"""官方 Cloud Agents REST API（/v1/agents）客户端。

本机 Python SDK bridge 在部分 Windows 环境 CreateAgent 会返回 HTTP 502。
此时改用官方 Cloud Agents HTTP API，能力对齐：多轮 / 流式 / 新建会话。
"""

from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

from cursor_chat.exceptions import ChatRunError, ChatStartupError

API_BASE = "https://api.cursor.com"


def _basic_auth_header(api_key: str) -> str:
    token = base64.b64encode(f"{api_key}:".encode("utf-8")).decode("ascii")
    return f"Basic {token}"


class CloudRestClient:
    def __init__(self, api_key: str, *, timeout: float = 300.0):
        self.api_key = api_key
        self.timeout = timeout

    def request(
        self,
        method: str,
        path: str,
        *,
        body: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        accept: Optional[str] = None,
    ) -> Any:
        url = f"{API_BASE}{path}"
        data = None if body is None else json.dumps(body).encode("utf-8")
        headers = {
            "Authorization": _basic_auth_header(self.api_key),
            "User-Agent": "cursor_chat/0.1",
        }
        if body is not None:
            headers["Content-Type"] = "application/json"
        if accept:
            headers["Accept"] = accept
        elif stream:
            headers["Accept"] = "text/event-stream"

        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            err_body = ""
            try:
                err_body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            raise ChatStartupError(
                f"Cloud API {method} {path} 失败 HTTP {exc.code}: {err_body or exc.reason}",
                is_retryable=exc.code in (408, 429, 500, 502, 503, 504),
                cause=exc,
            ) from exc
        except Exception as exc:
            raise ChatStartupError(f"Cloud API 请求失败: {exc}", cause=exc) from exc

        if stream:
            return resp
        raw = resp.read().decode("utf-8", errors="replace")
        if not raw.strip():
            return {}
        return json.loads(raw)

    def create_agent(
        self,
        prompt: str,
        *,
        model: str,
        mode: str = "plan",
        name: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "prompt": {"text": prompt},
            "model": {"id": model},
            "mode": mode,
        }
        if name:
            payload["name"] = name
        return self.request("POST", "/v1/agents", body=payload)

    def create_run(
        self,
        agent_id: str,
        prompt: str,
        *,
        model: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"prompt": {"text": prompt}}
        if model:
            payload["model"] = {"id": model}
        if mode:
            payload["mode"] = mode
        return self.request("POST", f"/v1/agents/{agent_id}/runs", body=payload)

    def get_run(self, agent_id: str, run_id: str) -> Dict[str, Any]:
        return self.request("GET", f"/v1/agents/{agent_id}/runs/{run_id}")

    def stream_run(self, agent_id: str, run_id: str):
        return self.request(
            "GET",
            f"/v1/agents/{agent_id}/runs/{run_id}/stream",
            stream=True,
            accept="text/event-stream",
        )

    def delete_agent(self, agent_id: str) -> None:
        try:
            self.request("DELETE", f"/v1/agents/{agent_id}")
        except ChatStartupError:
            pass

    def list_models(self) -> List[str]:
        """拉取账号可用模型 ID。

        优先 ``GET /v1/models``（Basic），兼容 ``items`` / ``models`` 数组为字符串或带 ``id`` 的对象。
        若解析为空则回退 ``GET /v0/models``（常返回可直接用于 create agent 的完整 ID）。
        """
        data = self.request("GET", "/v1/models")
        ids = self._parse_model_ids(data)
        if ids:
            return ids
        # v1 结构偶有变更；v0 多为扁平字符串列表
        try:
            data_v0 = self.request("GET", "/v0/models")
            ids = self._parse_model_ids(data_v0)
        except ChatStartupError:
            ids = []
        if not ids:
            raise ChatStartupError(f"Cloud API /v1/models 未返回可用模型: {data!r}")
        return ids

    @staticmethod
    def _parse_model_ids(data: Any) -> List[str]:
        raw = None
        if isinstance(data, dict):
            raw = data.get("items")
            if raw is None:
                raw = data.get("models")
        elif isinstance(data, list):
            raw = data
        if not isinstance(raw, list):
            return []

        ids: List[str] = []
        seen: set[str] = set()
        for item in raw:
            mid: Optional[str] = None
            if isinstance(item, str) and item.strip():
                mid = item.strip()
            elif isinstance(item, dict):
                cand = item.get("id") or item.get("model") or item.get("name")
                if cand and str(cand).strip():
                    mid = str(cand).strip()
            if mid and mid not in seen:
                seen.add(mid)
                ids.append(mid)
        return ids


@dataclass
class CloudRestRunResult:
    status: str
    result: str = ""
    id: str = ""


@dataclass
class CloudRestRun:
    id: str
    agent_id: str
    _client: CloudRestClient
    status: str = "running"
    result: str = ""
    _collected: str = ""
    _finished: bool = False

    def iter_text(self) -> Iterator[str]:
        if self._finished:
            if self._collected:
                yield self._collected
            return

        stream = self._client.stream_run(self.agent_id, self.id)
        try:
            event_name = "message"
            data_lines: List[str] = []
            while True:
                line_b = stream.readline()
                if not line_b:
                    break
                line = line_b.decode("utf-8", errors="replace").rstrip("\r\n")
                if line.startswith("event:"):
                    event_name = line[6:].strip()
                    continue
                if line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
                    continue
                if line == "":
                    if not data_lines:
                        event_name = "message"
                        continue
                    payload_text = "\n".join(data_lines)
                    data_lines = []
                    try:
                        payload = json.loads(payload_text) if payload_text else {}
                    except json.JSONDecodeError:
                        payload = {}
                    if event_name == "assistant":
                        text = payload.get("text") or ""
                        if text:
                            self._collected += text
                            yield text
                    elif event_name == "result":
                        self.status = str(payload.get("status") or self.status).lower()
                        final = payload.get("text")
                        if final and not self._collected:
                            self._collected = str(final)
                            yield str(final)
                        self.result = self._collected
                    elif event_name == "error":
                        msg = payload.get("message") or payload_text or "stream error"
                        raise ChatRunError(str(msg), run_id=self.id, agent_id=self.agent_id)
                    elif event_name == "done":
                        break
                    event_name = "message"
        finally:
            try:
                stream.close()
            except Exception:
                pass
            self._finished = True
            self.result = self._collected

    def text(self) -> str:
        if not self._finished:
            # 优先走 SSE；若无文本再轮询
            try:
                list(self.iter_text())
            except ChatRunError:
                raise
            except Exception:
                self._finished = False
        if self._collected:
            return self._collected
        deadline = time.monotonic() + self._client.timeout
        while time.monotonic() < deadline:
            info = self._client.get_run(self.agent_id, self.id)
            status = str(info.get("status") or "").upper()
            self.status = status.lower()
            if status in ("FINISHED", "COMPLETED", "DONE", "ERROR", "CANCELLED", "EXPIRED"):
                self.result = str(info.get("result") or info.get("text") or "")
                self._collected = self.result
                self._finished = True
                if status in ("ERROR", "CANCELLED", "EXPIRED"):
                    raise ChatRunError(
                        self.result or f"Run {status}",
                        run_id=self.id,
                        agent_id=self.agent_id,
                    )
                return self._collected
            time.sleep(1.5)
        raise ChatRunError("等待 Cloud Run 超时", run_id=self.id, agent_id=self.agent_id)

    def wait(self) -> CloudRestRunResult:
        text = self.text()
        status = (self.status or "finished").lower()
        if status == "error":
            raise ChatRunError(text or "Run 失败", run_id=self.id, agent_id=self.agent_id)
        return CloudRestRunResult(status="finished", result=text, id=self.id)

    def messages(self) -> Iterator[Any]:
        return iter(())

    def supports(self, operation: str) -> bool:
        return operation in ("stream", "wait", "conversation")


class CloudRestAgent:
    """鸭类型 Agent：提供 agent_id / send / close，供 ChatSession 复用。"""

    def __init__(self, config: Any):
        self._config = config
        self._client = CloudRestClient(config.api_key, timeout=float(config.timeout))
        self.agent_id: Optional[str] = None

    def send(self, prompt: str) -> CloudRestRun:
        mode = getattr(self._config, "mode", "plan") or "plan"
        model = getattr(self._config, "model", "composer-2.5")
        if not self.agent_id:
            data = self._client.create_agent(prompt, model=model, mode=mode, name="cursor_chat")
            agent = data.get("agent") or {}
            run = data.get("run") or {}
            self.agent_id = agent.get("id")
            run_id = run.get("id")
            if not self.agent_id or not run_id:
                raise ChatStartupError(f"Cloud 创建 Agent 返回异常: {data}")
            return CloudRestRun(id=run_id, agent_id=self.agent_id, _client=self._client)

        data = self._client.create_run(self.agent_id, prompt, model=model, mode=mode)
        run = data.get("run") if isinstance(data.get("run"), dict) else data
        run_id = (run or {}).get("id")
        if not run_id:
            raise ChatStartupError(f"Cloud 创建 Run 返回异常: {data}")
        return CloudRestRun(id=run_id, agent_id=self.agent_id, _client=self._client)

    def close(self) -> None:
        if self.agent_id:
            self._client.delete_agent(self.agent_id)
            self.agent_id = None
