"""LLM 配置：复用 apps/message/config_loader 与 packages/lite_agent_sdk。"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

TOOL_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = TOOL_DIR.parent.parent
MESSAGE_DIR = REPO_ROOT / "apps" / "message"
PACKAGES_DIR = REPO_ROOT / "packages"

for _p in (MESSAGE_DIR, PACKAGES_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from config_loader import load_config, resolve_llm_config  # noqa: E402
from lite_agent_sdk.base import create_llm_client, create_llm_client_from_cfg, llm_chat  # noqa: E402

__all__ = [
    "load_llm_config",
    "resolve_llm_config",
    "create_llm_client",
    "create_llm_client_from_cfg",
    "llm_chat",
    "MESSAGE_DIR",
    "TOOL_DIR",
    "REPO_ROOT",
    "PACKAGES_DIR",
]


def load_llm_config(provider: Optional[str] = None) -> Dict[str, Any]:
    cfg = load_config(MESSAGE_DIR)
    llm = resolve_llm_config(cfg, provider=provider)
    return {**cfg, **llm, "active_provider": llm["provider"]}
