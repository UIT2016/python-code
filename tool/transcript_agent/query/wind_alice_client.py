from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

from transcript_agent.query.models import FactBundle, FactItem, QueryContext
from wind_config import DEFAULT_ALICE_URL

ProgressCallback = Callable[[str], None]

_SSE_BLOCK_RE = re.compile(r"\r?\n\r?\n")
_TRIAL_QUOTA_SNIPPET = "很抱歉，今日已超出体验期任务限额"
_REJECT_SNIPPETS = (
    "此对话不支持继续问答",
    "请开启新对话",
)
_ALICE_REJECT_HINT = (
    "（Alice Agent 未产出 agentResult；若同 Key 曾返回「余额不足」，请先充值 AIFin 余额后重试）"
)


class AliceStreamError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        events: Optional[List[Dict[str, Any]]] = None,
        content_type: str = "",
        raw_len: int = 0,
    ) -> None:
        super().__init__(message)
        self.events = events or []
        self.content_type = content_type
        self.raw_len = raw_len

KNOWN_SKILLS: List[Dict[str, str]] = [
    {"nameZh": "公司一页纸", "nameEn": "Company One-Page Investment Memo"},
    {"nameZh": "上市公司调研问题清单", "nameEn": "Stock DD List"},
    {"nameZh": "按主题选股", "nameEn": "Thematic Stock Screening"},
    {"nameZh": "宏观数据解读", "nameEn": "Macro Data Interpretation"},
    {"nameZh": "投资标的创意与筛选", "nameEn": "Investment Idea Generation"},
]


def _normalize_skill_name(name: str) -> str:
    return re.sub(r"[\s\-_]+", "", name or "").lower()


def resolve_skill_name(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return ""
    norm = _normalize_skill_name(text)
    for skill in KNOWN_SKILLS:
        if text == skill["nameEn"] or text == skill["nameZh"]:
            return skill["nameEn"]
        if norm in (_normalize_skill_name(skill["nameEn"]), _normalize_skill_name(skill["nameZh"])):
            return skill["nameEn"]
    return text


def _parse_sse_payload(payload: str) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for block in _SSE_BLOCK_RE.split(payload):
        block = block.strip()
        if not block:
            continue
        data_lines = [
            line[5:].strip()
            for line in block.splitlines()
            if line.startswith("data:") and line[5:].strip() and line[5:].strip() != "[DONE]"
        ]
        if not data_lines:
            continue
        try:
            events.append(json.loads("\n".join(data_lines)))
        except json.JSONDecodeError:
            continue
    return events


def _text_from_content_items(items: Any) -> List[str]:
    chunks: List[str] = []
    if not isinstance(items, list):
        return chunks
    for item in items:
        if isinstance(item, str) and item.strip():
            chunks.append(item.strip())
            continue
        if not isinstance(item, dict):
            continue
        text = item.get("text") or item.get("content") or ""
        if isinstance(text, str) and text.strip():
            chunks.append(text.strip())
    return chunks


def _collect_ui_text(node: Any, out: List[str]) -> None:
    if not isinstance(node, dict):
        return
    props = node.get("properties") or {}
    text = props.get("text")
    if isinstance(text, list):
        out.extend(str(x) for x in text if str(x).strip())
    elif isinstance(text, str) and text.strip():
        out.append(text.strip())
    for child in node.get("children") or []:
        _collect_ui_text(child, out)


def _extract_agent_result_values(events: List[Dict[str, Any]]) -> List[Any]:
    values: List[Any] = []
    for event in events:
        result = event.get("result") if isinstance(event, dict) else None
        if not isinstance(result, dict) or result.get("kind") != "artifact-update":
            continue
        artifact = result.get("artifact") or {}
        if not isinstance(artifact, dict) or artifact.get("name") != "agentResult":
            continue
        for part in artifact.get("parts") or []:
            if not isinstance(part, dict) or part.get("kind") != "data":
                continue
            data = part.get("data") or {}
            if isinstance(data, dict):
                value = data.get("data")
                if value is not None:
                    values.append(value)
    return values


def _extract_streaming_markdown(events: List[Dict[str, Any]]) -> str:
    chunks: List[str] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        result = event.get("result") or {}
        if not isinstance(result, dict):
            continue

        artifact = result.get("artifact") or {}
        if isinstance(artifact, dict) and artifact.get("name") == "UIState":
            for part in artifact.get("parts") or []:
                if not isinstance(part, dict):
                    continue
                ui = (part.get("data") or {}).get("__ui__")
                if isinstance(ui, dict):
                    _collect_ui_text(ui, chunks)

        status = result.get("status") or {}
        message = status.get("message") if isinstance(status, dict) else None
        if isinstance(message, dict):
            for part in message.get("parts") or []:
                if not isinstance(part, dict):
                    continue
                if part.get("kind") == "text":
                    text = part.get("text")
                    if isinstance(text, str) and text.strip():
                        chunks.append(text.strip())
                    continue
                data = part.get("data") or {}
                if not isinstance(data, dict):
                    continue
                for patch in data.get("__ui_patch__") or []:
                    if not isinstance(patch, dict) or patch.get("op") != "add":
                        continue
                    path = str(patch.get("path") or "")
                    value = patch.get("value")
                    if "/text/-" in path or path.endswith("/text"):
                        if isinstance(value, str) and value.strip():
                            chunks.append(value.strip())
                        elif isinstance(value, list):
                            chunks.extend(str(x) for x in value if str(x).strip())

    return "".join(chunks).strip()


def _extract_error_from_events(events: List[Dict[str, Any]]) -> str:
    for event in events:
        if not isinstance(event, dict):
            continue
        if event.get("error"):
            err = event["error"]
            if isinstance(err, dict):
                return str(err.get("message") or json.dumps(err, ensure_ascii=False)).strip()
            return str(err).strip()
        result = event.get("result")
        if not isinstance(result, dict):
            continue
        if result.get("isError"):
            chunks = _text_from_content_items(result.get("content"))
            if chunks:
                return "；".join(chunks)
            return str(result.get("error") or result.get("message") or "Alice 返回错误").strip()
        blob = json.dumps(event, ensure_ascii=False)
        if _TRIAL_QUOTA_SNIPPET in blob:
            return _TRIAL_QUOTA_SNIPPET
    return ""


def _is_reject_only(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    return any(snippet in stripped for snippet in _REJECT_SNIPPETS) and len(stripped) < 80


def extract_content_from_events(events: List[Dict[str, Any]]) -> Tuple[List[str], str]:
    err = _extract_error_from_events(events)
    if err:
        return [], err

    agent_values = _extract_agent_result_values(events)
    if agent_values:
        texts = []
        for value in agent_values:
            if isinstance(value, str) and value.strip():
                texts.append(value.strip())
            else:
                texts.append(json.dumps(value, ensure_ascii=False))
        return texts, ""

    markdown = _extract_streaming_markdown(events)
    if markdown and not _is_reject_only(markdown):
        return [markdown], ""

    if markdown and _is_reject_only(markdown):
        return [], f"{markdown.strip()}{_ALICE_REJECT_HINT}"

    return [], ""


def _parse_json_rpc_body(raw: str) -> Tuple[List[Dict[str, Any]], str]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return [], f"Alice 响应非 JSON: {exc}"
    if not isinstance(data, dict):
        return [], "Alice 响应格式异常"
    if data.get("error"):
        err = data["error"]
        if isinstance(err, dict):
            return [data], err.get("message") or json.dumps(err, ensure_ascii=False)
        return [data], str(err)
    return [data], ""


def _values_to_text(values: List[Any]) -> str:
    chunks: List[str] = []
    for value in values:
        if isinstance(value, str) and value.strip():
            chunks.append(value.strip())
        else:
            chunks.append(json.dumps(value, ensure_ascii=False))
    return "\n\n".join(chunks).strip()


def _facts_from_research_text(text: str, *, max_items: int = 8) -> List[FactItem]:
    if not text.strip():
        return []
    parts = re.split(r"\n{2,}", text.strip())
    facts: List[FactItem] = []
    for part in parts:
        line = re.sub(r"\s+", " ", part).strip()
        if len(line) < 20:
            continue
        facts.append(FactItem(fact_type="alice_research", text=line[:500], source="wind_alice"))
        if len(facts) >= max_items:
            break
    if not facts and text.strip():
        facts.append(FactItem(fact_type="alice_research", text=text.strip()[:2000], source="wind_alice"))
    return facts


def _event_kinds(events: List[Dict[str, Any]]) -> List[str]:
    kinds: List[str] = []
    for event in events[-12:]:
        result = event.get("result") if isinstance(event, dict) else None
        if not isinstance(result, dict):
            continue
        artifact_name = (result.get("artifact") or {}).get("name") if isinstance(result.get("artifact"), dict) else None
        kinds.append(f"{result.get('kind')}/{artifact_name}")
    return kinds


class WindAliceClient:
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        from wind_config import load_wind_config

        self.config = config or load_wind_config()
        self.api_key = (self.config.get("wind_api_key") or "").strip()
        self.api_url = (self.config.get("wind_alice_api_url") or DEFAULT_ALICE_URL).strip()
        self.default_skill = (self.config.get("wind_alice_default_skill") or "").strip()
        self.timeout = int(self.config.get("wind_alice_timeout_sec") or 600)

    def _build_body(self, prompt: str, skill_name: Optional[str]) -> Dict[str, Any]:
        text = prompt
        if skill_name:
            text = f'Using "{skill_name}" skill:{prompt}'
        return {
            "jsonrpc": "2.0",
            "method": "message/stream",
            "params": {
                "message": {
                    "messageId": str(uuid.uuid4()),
                    "role": "user",
                    "kind": "message",
                    "parts": [
                        {"kind": "text", "text": text},
                        {
                            "kind": "data",
                            "data": {
                                "chatMode": "12",
                                "originalChatMode": "4",
                                "switchMode": "auto",
                                "timezone": "Asia/Shanghai",
                            },
                            "metadata": {
                                "key": "Wind.WindSearch.ChatService.A2A",
                                "version": "1.0.0",
                            },
                        },
                    ],
                    "contextId": str(uuid.uuid4()),
                    "taskId": str(uuid.uuid4()),
                }
            },
            "id": str(uuid.uuid4()),
        }

    def _request_alice(
        self,
        body: Dict[str, Any],
        on_progress: Optional[ProgressCallback],
    ) -> Tuple[List[str], List[Dict[str, Any]], str, int]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        }
        resp = requests.post(
            self.api_url,
            headers=headers,
            json=body,
            stream=True,
            timeout=self.timeout,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Alice HTTP {resp.status_code}: {resp.text[:300]}")

        content_type = (resp.headers.get("content-type") or "").lower()
        if "text/event-stream" not in content_type:
            raw = resp.text
            resp.close()
            events, parse_err = _parse_json_rpc_body(raw)
            if parse_err and not events:
                raise RuntimeError(parse_err)
            values, biz_err = extract_content_from_events(events)
            if biz_err:
                raise RuntimeError(biz_err)
            if not values and events:
                raise RuntimeError(parse_err or "Alice 未返回有效内容")
            return values, events, content_type, len(raw)

        raw_events: List[Dict[str, Any]] = []
        buffer = ""
        raw_len = 0
        last_ping = time.monotonic()

        for chunk in resp.iter_content(chunk_size=4096):
            if not chunk:
                continue
            raw_len += len(chunk)
            buffer += chunk.decode("utf-8", errors="replace")
            while True:
                match = _SSE_BLOCK_RE.search(buffer)
                if not match:
                    break
                block_text = buffer[: match.end()]
                buffer = buffer[match.end() :]
                raw_events.extend(_parse_sse_payload(block_text))
            if on_progress and time.monotonic() - last_ping >= 15:
                on_progress("Alice 深度调研进行中...")
                last_ping = time.monotonic()

        if buffer.strip():
            raw_events.extend(_parse_sse_payload(buffer))

        resp.close()

        values, biz_err = extract_content_from_events(raw_events)
        if biz_err:
            raise AliceStreamError(
                biz_err,
                events=raw_events,
                content_type=content_type,
                raw_len=raw_len,
            )

        deduped: List[str] = []
        seen: set[str] = set()
        for chunk in values:
            key = chunk[:200]
            if key in seen:
                continue
            seen.add(key)
            deduped.append(chunk)

        if not deduped:
            failed = any(
                isinstance(event.get("result"), dict)
                and (event["result"].get("status") or {}).get("state") == "failed"
                for event in raw_events
                if isinstance(event, dict)
            )
            reason = "Alice 未返回 agentResult"
            if failed:
                reason += "（任务 state=failed）"
            if raw_events:
                reason += f"；SSE 事件 {len(raw_events)} 个，见 event_kinds"
            raise AliceStreamError(
                reason,
                events=raw_events,
                content_type=content_type,
                raw_len=raw_len,
            )
        return deduped, raw_events, content_type, raw_len

    def research(
        self,
        query: QueryContext,
        *,
        context: str = "",
        on_progress: Optional[ProgressCallback] = None,
        structured_facts: Optional[FactBundle] = None,
    ) -> FactBundle:
        research_calls: List[Dict[str, Any]] = []
        if not self.api_key:
            bundle = self._degraded(query, "未配置 WIND_API_KEY")
            bundle.research_mode = "wind_alice"
            return bundle

        prompt_parts = [query.raw_query.strip() or query.subject]
        if query.stock_code:
            prompt_parts.append(query.stock_code)
        if context.strip():
            prompt_parts.append(context.strip())
        if query.sector_hint:
            prompt_parts.append(f"板块:{query.sector_hint}")
        if structured_facts and structured_facts.facts:
            hint_lines = ["参考检索摘要："]
            for fact in structured_facts.facts[:8]:
                line = re.sub(r"\s+", " ", fact.text).strip()
                if len(line) < 16:
                    continue
                hint_lines.append(f"[{fact.fact_type}] {line[:240]}")
            if len(hint_lines) > 1:
                prompt_parts.append(" ".join(hint_lines))
        prompt = "\n".join(p for p in prompt_parts if p).strip()

        skill_name = resolve_skill_name(self.default_skill) or None
        body = self._build_body(prompt, skill_name)
        call_record: Dict[str, Any] = {
            "provider": "wind_alice",
            "api_url": self.api_url,
            "skill": skill_name,
            "prompt": prompt,
        }
        started = time.monotonic()
        try:
            if on_progress:
                on_progress("Alice 深度调研启动...")
            values, raw_events, content_type, raw_len = self._request_alice(body, on_progress)
            raw_text = _values_to_text(values)
            call_record["response"] = {
                "content_type": content_type,
                "event_count": len(raw_events),
                "value_count": len(values),
                "raw_len": raw_len,
                "event_kinds": _event_kinds(raw_events),
                "preview": raw_text[:1200] if raw_text else "",
            }
            call_record["error"] = None
            call_record["elapsed_sec"] = round(time.monotonic() - started, 1)
            research_calls.append(call_record)

            if not raw_text:
                reason = "Alice 未返回有效内容"
                if raw_events:
                    reason += f"（解析到 {len(raw_events)} 个 SSE 事件，见 event_kinds）"
                bundle = self._degraded(query, reason)
                bundle.wind_calls = research_calls
                bundle.research_mode = "wind_alice"
                return self._merge_structured_facts(bundle, structured_facts, research_calls)

            facts = _facts_from_research_text(raw_text)
            summary_parts = [f"标的:{query.subject}", f"深度调研:Alice"]
            if query.stock_code:
                summary_parts.append(f"代码:{query.stock_code}")
            for fact in facts[:6]:
                summary_parts.append(f"[{fact.fact_type}] {fact.text[:120]}")
            bundle = FactBundle(
                subject=query.subject,
                facts=facts,
                summary_for_match="；".join(summary_parts)[:800],
                wind_status="ok",
                research_mode="wind_alice",
                research_raw_md=raw_text,
                wind_calls=research_calls,
            )
            return self._merge_structured_facts(bundle, structured_facts, research_calls)
        except AliceStreamError as exc:
            call_record["response"] = {
                "content_type": exc.content_type,
                "event_count": len(exc.events),
                "value_count": 0,
                "raw_len": exc.raw_len,
                "event_kinds": _event_kinds(exc.events),
                "preview": "",
            }
            call_record["error"] = str(exc)
            call_record["elapsed_sec"] = round(time.monotonic() - started, 1)
            research_calls.append(call_record)
            bundle = self._degraded(query, str(exc))
            bundle.wind_calls = research_calls
            bundle.research_mode = "wind_alice"
            return self._merge_structured_facts(bundle, structured_facts, research_calls)
        except Exception as exc:
            call_record["response"] = call_record.get("response") or {
                "content_type": "",
                "event_count": 0,
                "value_count": 0,
                "raw_len": 0,
            }
            call_record["error"] = str(exc)
            call_record["elapsed_sec"] = round(time.monotonic() - started, 1)
            research_calls.append(call_record)
            bundle = self._degraded(query, str(exc))
            bundle.wind_calls = research_calls
            bundle.research_mode = "wind_alice"
            return self._merge_structured_facts(bundle, structured_facts, research_calls)

    @staticmethod
    def _merge_structured_facts(
        bundle: FactBundle,
        structured: Optional[FactBundle],
        alice_calls: List[Dict[str, Any]],
    ) -> FactBundle:
        if not structured:
            return bundle
        bundle.research_raw_sections = dict(structured.research_raw_sections)
        wind_calls = list(structured.wind_calls or [])
        for call in alice_calls:
            if call not in wind_calls:
                wind_calls.append(call)
        bundle.wind_calls = wind_calls
        if structured.facts:
            seen = {f.fact_type for f in bundle.facts}
            merged_facts = list(structured.facts) + [f for f in bundle.facts if f.fact_type not in seen]
            bundle.facts = merged_facts[:16]
        if structured.summary_for_match and bundle.wind_status != "ok":
            bundle.summary_for_match = structured.summary_for_match[:1200]
        elif structured.summary_for_match and bundle.summary_for_match:
            bundle.summary_for_match = f"{structured.summary_for_match}；{bundle.summary_for_match}"[:1500]
        if structured.wind_status == "ok" and bundle.wind_status == "degraded":
            bundle.wind_status = "degraded"
        return bundle

    @staticmethod
    def _degraded(query: QueryContext, reason: str) -> FactBundle:
        summary_parts = [f"标的:{query.subject}", f"查询:{query.raw_query}", f"Alice降级:{reason}"]
        if query.stock_code:
            summary_parts.append(f"代码:{query.stock_code}")
        return FactBundle(
            subject=query.subject,
            facts=[FactItem(fact_type="query", text=query.raw_query, source="user")],
            summary_for_match="；".join(summary_parts)[:800],
            wind_status="degraded",
            research_mode="wind_alice",
        )
