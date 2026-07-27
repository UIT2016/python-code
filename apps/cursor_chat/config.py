from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from cursor_chat.exceptions import ChatConfigError

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_FILES = ("config.example.json", "config.local.json")


@dataclass
class KnowledgeConfig:
    """知识库配置（与向量库实现解耦，便于后续替换）。"""

    directories: List[str] = field(default_factory=list)
    top_k: int = 5
    max_chunk_chars: int = 1500


@dataclass
class ChatConfig:
    """聊天模块配置。api_key 优先从配置读取，其次 CURSOR_API_KEY 环境变量。"""

    api_key: str
    model: str = "composer-2.5"
    cwd: str = "."
    mode: str = "plan"
    # local=官方 Python SDK；cloud=官方 Cloud Agents REST；auto=先 local 失败再 cloud
    runtime: str = "cloud"
    base_url: Optional[str] = None
    auth_token: Optional[str] = None
    timeout: float = 300.0
    max_retries: int = 2
    system_prompt: Optional[str] = None
    knowledge: KnowledgeConfig = field(default_factory=KnowledgeConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatConfig":
        api_key = (data.get("api_key") or os.environ.get("CURSOR_API_KEY") or "").strip()
        if not api_key:
            raise ChatConfigError(
                "未配置 api_key。请在 config.local.json 填写 api_key，或设置环境变量 CURSOR_API_KEY。"
            )

        kb_raw = data.get("knowledge") or {}
        knowledge = KnowledgeConfig(
            directories=[str(p) for p in kb_raw.get("directories", [])],
            top_k=int(kb_raw.get("top_k", 5)),
            max_chunk_chars=int(kb_raw.get("max_chunk_chars", 1500)),
        )

        base_url = (data.get("base_url") or "").strip() or None
        auth_token = (data.get("auth_token") or "").strip() or None
        cwd = str(data.get("cwd") or ".").strip() or "."
        mode = str(data.get("mode") or "plan").strip() or "plan"
        if mode not in ("agent", "plan"):
            raise ChatConfigError(f"不支持的 mode: {mode}，可选 agent / plan")

        runtime = str(data.get("runtime") or "cloud").strip().lower() or "cloud"
        if runtime not in ("cloud", "local", "auto"):
            raise ChatConfigError(f"不支持的 runtime: {runtime}，可选 cloud / local / auto")

        return cls(
            api_key=api_key,
            model=str(data.get("model") or "composer-2.5"),
            cwd=cwd,
            mode=mode,
            runtime=runtime,
            base_url=base_url,
            auth_token=auth_token,
            timeout=float(data.get("timeout", 300.0)),
            max_retries=int(data.get("max_retries", 2)),
            system_prompt=(data.get("system_prompt") or None),
            knowledge=knowledge,
        )


def _merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_dict(out[key], value)
        else:
            out[key] = value
    return out


def load_chat_config(base_dir: Optional[Path] = None) -> ChatConfig:
    """加载 config.example.json + config.local.json（后者覆盖前者）。"""
    base = base_dir or BASE_DIR
    example_path = base / "config.example.json"
    local_path = base / "config.local.json"

    if not example_path.exists():
        raise ChatConfigError(f"缺少示例配置: {example_path}")

    with example_path.open(encoding="utf-8") as f:
        config_data = json.load(f)

    if local_path.exists():
        with local_path.open(encoding="utf-8") as f:
            local_data = json.load(f)
        config_data = _merge_dict(config_data, local_data)

    return ChatConfig.from_dict(config_data)
