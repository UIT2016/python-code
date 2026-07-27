"""万行一言 MCP over SSE 客户端（同步版）。

万行使用经典 MCP HTTP+SSE 传输：
1. GET /sse 保持长连接
2. 服务端推送 endpoint（/messages/?session_id=...）
3. POST 到 messages 端点返回 202 Accepted，JSON-RPC 响应经 SSE 流回传
"""
from __future__ import annotations

import json
import queue
import threading
import time
from typing import Any, Dict, Optional
from urllib.parse import urljoin, urlparse

import requests


class WanxingMcpClient:
    def __init__(self, sse_url: str, api_key: str, *, timeout: int = 90) -> None:
        self.sse_url = sse_url.rstrip("/")
        self.api_key = api_key.strip()
        self.timeout = timeout
        self._message_url: Optional[str] = None
        self._initialized = False
        self._req_id = 0
        self._session_lock = threading.Lock()
        self._endpoint_ready = threading.Event()
        self._pending: Dict[str, queue.Queue] = {}
        self._sse_thread: Optional[threading.Thread] = None
        self._sse_resp: Optional[requests.Response] = None

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }

    def _next_id(self) -> str:
        self._req_id += 1
        return str(int(time.time() * 1000) + self._req_id)

    def _sse_base_url(self) -> str:
        parsed = urlparse(self.sse_url)
        return f"{parsed.scheme}://{parsed.netloc}"

    def _resolve_message_url(self, endpoint: str) -> str:
        if endpoint.startswith("http"):
            return endpoint
        return urljoin(self._sse_base_url() + "/", endpoint.lstrip("/"))

    def _dispatch_sse_message(self, payload: str) -> None:
        try:
            msg = json.loads(payload)
        except json.JSONDecodeError:
            return
        req_id = msg.get("id")
        if req_id is None:
            return
        key = str(req_id)
        pending = self._pending.get(key)
        if pending is not None:
            pending.put(msg)

    def _sse_reader_loop(self) -> None:
        resp = requests.get(
            self.sse_url,
            headers=self._headers(),
            stream=True,
            timeout=self.timeout,
        )
        self._sse_resp = resp
        if resp.status_code != 200:
            self._endpoint_ready.set()
            return
        event_type = ""
        for raw in resp.iter_lines(decode_unicode=True):
            if raw is None:
                continue
            line = raw.strip()
            if not line:
                continue
            if line.startswith("event:"):
                event_type = line[6:].strip()
            elif line.startswith("data:"):
                data = line[5:].strip()
                if event_type == "endpoint" and data:
                    self._message_url = self._resolve_message_url(data)
                    self._endpoint_ready.set()
                elif event_type == "message" and data:
                    self._dispatch_sse_message(data)
        resp.close()

    def _ensure_sse_session(self) -> None:
        if self._sse_thread and self._sse_thread.is_alive() and self._message_url:
            return
        self._endpoint_ready.clear()
        self._message_url = None
        self._sse_thread = threading.Thread(target=self._sse_reader_loop, daemon=True)
        self._sse_thread.start()
        if not self._endpoint_ready.wait(timeout=30):
            raise RuntimeError("万行 MCP SSE 连接超时，未收到 endpoint")
        if not self._message_url:
            raise RuntimeError("万行 MCP SSE 连接失败，未获取 messages 端点")

    def _post_raw(self, body: Dict[str, Any]) -> requests.Response:
        self._ensure_sse_session()
        assert self._message_url
        return requests.post(
            self._message_url,
            headers=self._headers(),
            json=body,
            timeout=self.timeout,
        )

    def _try_parse_direct_json(self, resp: requests.Response) -> Optional[Any]:
        if resp.status_code != 200:
            return None
        content_type = (resp.headers.get("content-type") or "").lower()
        text = (resp.text or "").strip()
        if not text or "text/event-stream" in content_type or text.startswith("event:"):
            return None
        try:
            data = resp.json()
        except ValueError:
            return None
        if data.get("error"):
            err = data["error"]
            msg = err.get("message") if isinstance(err, dict) else str(err)
            raise RuntimeError(msg or json.dumps(err, ensure_ascii=False))
        return data.get("result")

    def _post_jsonrpc(self, method: str, params: Optional[Dict[str, Any]] = None) -> Any:
        with self._session_lock:
            req_id = self._next_id()
            body = {
                "jsonrpc": "2.0",
                "id": req_id,
                "method": method,
                "params": params or {},
            }
            pending: queue.Queue = queue.Queue()
            self._pending[req_id] = pending
            try:
                resp = self._post_raw(body)
                if resp.status_code not in (200, 202):
                    raise RuntimeError(f"MCP HTTP {resp.status_code}: {resp.text[:300]}")
                direct = self._try_parse_direct_json(resp)
                if direct is not None:
                    return direct
                try:
                    msg = pending.get(timeout=self.timeout)
                except queue.Empty:
                    raise RuntimeError(f"万行 MCP 等待响应超时: {method}") from None
            finally:
                self._pending.pop(req_id, None)

        if msg.get("error"):
            err = msg["error"]
            err_msg = err.get("message") if isinstance(err, dict) else str(err)
            raise RuntimeError(err_msg or json.dumps(err, ensure_ascii=False))
        return msg.get("result")

    def _post_notification(self, method: str, params: Optional[Dict[str, Any]] = None) -> None:
        with self._session_lock:
            body = {"jsonrpc": "2.0", "method": method, "params": params or {}}
            resp = self._post_raw(body)
            if resp.status_code not in (200, 202):
                raise RuntimeError(f"MCP HTTP {resp.status_code}: {resp.text[:300]}")

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        self._post_jsonrpc(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "python-code-transcript-agent", "version": "1.0"},
            },
        )
        try:
            self._post_notification("notifications/initialized", {})
        except RuntimeError:
            pass
        self._initialized = True

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        self._ensure_initialized()
        result = self._post_jsonrpc(
            "tools/call",
            {"name": tool_name, "arguments": arguments},
        )
        if isinstance(result, dict) and result.get("isError"):
            content = result.get("content") or []
            msg = content[0].get("text") if content and isinstance(content[0], dict) else str(result)
            raise RuntimeError(str(msg)[:500])
        if isinstance(result, dict):
            content = result.get("content")
            if isinstance(content, list) and content:
                first = content[0]
                if isinstance(first, dict) and first.get("text"):
                    text = first["text"]
                    try:
                        inner = json.loads(text)
                    except json.JSONDecodeError:
                        return text
                    if isinstance(inner, dict):
                        if inner.get("message") and inner.get("code"):
                            raise RuntimeError(f"{inner.get('code')}: {inner.get('message')}")
                    return inner
        return result
