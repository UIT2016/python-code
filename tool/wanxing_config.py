"""万行一言 MCP 配置加载。"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

TOOL_DIR = Path(__file__).resolve().parent
EXAMPLE_PATH = TOOL_DIR / "wanxing.example.json"
LOCAL_PATH = TOOL_DIR / "wanxing.local.json"


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_wanxing_config() -> Dict[str, Any]:
    if not EXAMPLE_PATH.exists():
        raise FileNotFoundError(f"缺少 {EXAMPLE_PATH}")
    cfg = dict(_read_json(EXAMPLE_PATH))
    if LOCAL_PATH.exists():
        cfg.update(_read_json(LOCAL_PATH))
    env_key = os.environ.get("WANXING_API_KEY", "").strip()
    if env_key:
        cfg["wanxing_api_key"] = env_key
    return cfg


def save_research_response_enabled(cfg: Optional[Dict[str, Any]] = None) -> bool:
    cfg = cfg or load_wanxing_config()
    return bool(cfg.get("save_research_response", False))
