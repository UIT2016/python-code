from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

import requests

from transcript_agent.query.models import FactBundle, FactItem, QueryContext

ProgressCallback = Callable[[str], None]

DEFAULT_ALICE_URL = "https://mcp.wind.com.cn/skills/alice"

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


def _parse_sse_payload(text: str) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for block in text.split("\n\n"):
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


def extract_agent_result_values(events: List[Dict[str, Any]]) -> List[Any]:
    values: List[Any] = []
    for event in events:
        artifact = (event.get("result") or {}).get("artifact") if isinstance(event.get("result"), dict) else None
        if not isinstance(artifact, dict):
            continue
        if event.get("result", {}).get("kind") != "artifact-update":
            continue
        if artifact.get("name") != "agentResult":
            continue
        for part in artifact.get("parts") or []:
            if not isinstance(part, dict) or part.get("kind") != "data":
                continue
            data = (part.get("data") or {}).get("data")
            if data is not None:
                values.append(data)
    return values


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

    def _stream_alice(self, body: Dict[str, Any], on_progress: Optional[ProgressCallback]) -> tuple[List[Any], List[Dict[str, Any]]]:
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

        all_values: List[Any] = []
        raw_events: List[Dict[str, Any]] = []
        buffer = ""
        last_ping = time.monotonic()

        for line in resp.iter_lines(decode_unicode=True):
            if line:
                buffer += line + "\n"
            while "\n\n" in buffer:
                block, buffer = buffer.split("\n\n", 1)
                events = _parse_sse_payload(block)
                if events:
                    raw_events.extend(events)
                    all_values.extend(extract_agent_result_values(events))
            if on_progress and time.monotonic() - last_ping >= 15:
                on_progress("Alice 深度调研进行中...")
                last_ping = time.monotonic()
        if buffer.strip():
            events = _parse_sse_payload(buffer)
            raw_events.extend(events)
            all_values.extend(extract_agent_result_values(events))
        resp.close()
        return all_values, raw_events

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
        if context.strip():
            prompt_parts.append(context.strip())
        if query.sector_hint:
            prompt_parts.append(f"板块:{query.sector_hint}")
        if structured_facts and structured_facts.research_raw_sections:
            section_lines = ["结构化检索结果（SearchPlan）："]
            for intent, text in structured_facts.research_raw_sections.items():
                section_lines.append(f"[{intent}] {text[:2000]}")
            prompt_parts.append("\n".join(section_lines))
        prompt = " ".join(p for p in prompt_parts if p)

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
            values, raw_events = self._stream_alice(body, on_progress)
            raw_text = _values_to_text(values)
            call_record["response"] = {"event_count": len(raw_events), "value_count": len(values)}
            call_record["error"] = None
            call_record["elapsed_sec"] = round(time.monotonic() - started, 1)
            research_calls.append(call_record)

            if not raw_text:
                bundle = self._degraded(query, "Alice 未返回有效内容")
                bundle.wind_calls = research_calls
                bundle.research_mode = "wind_alice"
                return bundle

            facts = _facts_from_research_text(raw_text)
            summary_parts = [f"标的:{query.subject}", f"深度调研:Alice"]
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
        except Exception as exc:
            call_record["response"] = None
            call_record["error"] = str(exc)
            call_record["elapsed_sec"] = round(time.monotonic() - started, 1)
            research_calls.append(call_record)
            bundle = self._degraded(query, str(exc))
            bundle.wind_calls = research_calls
            bundle.research_mode = "wind_alice"
            return bundle

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
        wind_calls.extend(alice_calls)
        bundle.wind_calls = wind_calls
        seen = {f.fact_type for f in bundle.facts}
        merged_facts = list(structured.facts) + [f for f in bundle.facts if f.fact_type not in seen]
        bundle.facts = merged_facts[:16]
        if structured.summary_for_match:
            bundle.summary_for_match = f"{structured.summary_for_match}；{bundle.summary_for_match}"[:1500]
        return bundle

    @staticmethod
    def _degraded(query: QueryContext, reason: str) -> FactBundle:
        summary_parts = [f"标的:{query.subject}", f"查询:{query.raw_query}", f"Alice降级:{reason}"]
        return FactBundle(
            subject=query.subject,
            facts=[FactItem(fact_type="query", text=query.raw_query, source="user")],
            summary_for_match="；".join(summary_parts)[:800],
            wind_status="degraded",
            research_mode="wind_alice",
        )
