from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import yaml
from lite_agent.client import OpenAIClient

from transcript_agent.base import (
    ACTIVE_RUBRIC_PATH,
    BASE_RUBRIC_PATH,
    KNOWLEDGE_DIR,
    MANUAL_OVERRIDES_PATH,
    RULES_DIR,
    now_version,
    parse_json_object,
)
from transcript_agent.llm_config import llm_chat

CURATOR_SYSTEM = """你是一位投资内容审计规则专家，根据知识库文档和手工规则生成统一的 Audit Rubric。
输出严格 JSON 对象，不要输出其它文字。

必须包含字段：
{
  "version": "ISO 时间字符串",
  "pass_threshold": 85,
  "min_fidelity": 16,
  "dimensions": [
    {"id": "fidelity", "weight": 20, "label": "忠实原文", "checks": ["..."]},
    {"id": "investment_value", "weight": 20, "label": "投资价值", "checks": ["..."]},
    {"id": "structure", "weight": 20, "label": "结构完整", "checks": ["..."]},
    {"id": "actionability", "weight": 20, "label": "可行动性", "checks": ["..."]},
    {"id": "noise_removal", "weight": 20, "label": "噪音剔除", "checks": ["..."]}
  ],
  "forbidden_patterns": ["必涨", "稳赚"],
  "required_fields": ["core_thesis", "framework_tags", "insights"],
  "manual_must": [],
  "manual_must_not": []
}

manual_overrides 中的 must / must_not 必须原样写入 manual_must / manual_must_not，不得删改。"""


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _read_knowledge_docs() -> List[Dict[str, str]]:
    docs: List[Dict[str, str]] = []
    if not KNOWLEDGE_DIR.exists():
        return docs
    for path in sorted(KNOWLEDGE_DIR.glob("*")):
        if path.suffix.lower() not in (".md", ".txt") or path.name.upper() == "README.MD":
            continue
        text = path.read_text(encoding="utf-8").strip()
        if text:
            docs.append({"name": path.name, "content": text[:8000]})
    return docs


def load_active_rubric() -> Dict[str, Any]:
    if ACTIVE_RUBRIC_PATH.exists():
        with ACTIVE_RUBRIC_PATH.open(encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    base = _read_yaml(BASE_RUBRIC_PATH)
    if base:
        base.setdefault("version", "base-template")
        return base
    return {
        "version": "default",
        "pass_threshold": 85,
        "min_fidelity": 16,
        "dimensions": [
            {"id": "fidelity", "weight": 20, "label": "忠实原文", "checks": ["每条 insight 有 evidence_quotes"]},
            {"id": "investment_value", "weight": 20, "label": "投资价值", "checks": ["保留定性框架与因果链"]},
            {"id": "structure", "weight": 20, "label": "结构完整", "checks": ["core_thesis 与 insights 非空"]},
            {"id": "actionability", "weight": 20, "label": "可行动性", "checks": ["关键观点有可行动建议"]},
            {"id": "noise_removal", "weight": 20, "label": "噪音剔除", "checks": ["剔除寒暄与重复"]},
        ],
        "forbidden_patterns": ["必涨", "稳赚", "神秘代码"],
        "required_fields": ["core_thesis", "framework_tags", "insights"],
        "manual_must": [],
        "manual_must_not": [],
    }


class RuleCuratorAgent:
    name = "RuleCuratorAgent"

    def __init__(self, llm: OpenAIClient):
        self.llm = llm

    async def refresh_rubric(self) -> Dict[str, Any]:
        manual = _read_yaml(MANUAL_OVERRIDES_PATH)
        base = _read_yaml(BASE_RUBRIC_PATH)
        knowledge = _read_knowledge_docs()
        locked = manual.get("locked_rules") or []

        user = f"""基础模板 base_rubric.yaml：
{yaml.dump(base, allow_unicode=True, default_flow_style=False)}

手工规则 manual_overrides.yaml（最高优先级）：
{yaml.dump(manual, allow_unicode=True, default_flow_style=False)}

知识库文档（{len(knowledge)} 篇）：
{json.dumps(knowledge, ensure_ascii=False, indent=2)}

锁定规则（不可覆盖）： {json.dumps(locked, ensure_ascii=False)}
"""
        raw = await llm_chat(self.llm, CURATOR_SYSTEM, user)
        rubric = parse_json_object(raw) or {}
        rubric["version"] = now_version()
        rubric.setdefault("pass_threshold", base.get("pass_threshold", 85))
        rubric.setdefault("min_fidelity", base.get("min_fidelity", 16))
        if manual.get("must"):
            rubric["manual_must"] = manual["must"]
        if manual.get("must_not"):
            rubric["manual_must_not"] = manual["must_not"]
        for key in ("forbidden_patterns", "required_fields"):
            if base.get(key) and key not in rubric:
                rubric[key] = base[key]

        RULES_DIR.mkdir(parents=True, exist_ok=True)
        versions_dir = RULES_DIR / "versions"
        versions_dir.mkdir(exist_ok=True)
        snapshot = versions_dir / f"rubric_{rubric['version'].replace(':', '-')}.json"
        with snapshot.open("w", encoding="utf-8") as f:
            json.dump(rubric, f, ensure_ascii=False, indent=2)
        with ACTIVE_RUBRIC_PATH.open("w", encoding="utf-8") as f:
            json.dump(rubric, f, ensure_ascii=False, indent=2)
        return rubric

    async def ingest_with_rag(self) -> Dict[str, Any]:
        """Phase 2 预留：向量检索后再归纳规则。首版直接调用 refresh_rubric。"""
        return await self.refresh_rubric()

    @staticmethod
    def read_manual_overrides_text() -> str:
        if MANUAL_OVERRIDES_PATH.exists():
            return MANUAL_OVERRIDES_PATH.read_text(encoding="utf-8")
        return ""

    @staticmethod
    def save_manual_overrides_text(content: str) -> None:
        RULES_DIR.mkdir(parents=True, exist_ok=True)
        MANUAL_OVERRIDES_PATH.write_text(content, encoding="utf-8")
