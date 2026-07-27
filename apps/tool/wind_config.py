"""Wind AIFin Market 与 Embedding 配置加载。"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

TOOL_DIR = Path(__file__).resolve().parent
EXAMPLE_PATH = TOOL_DIR / "wind.example.json"
LOCAL_PATH = TOOL_DIR / "wind.local.json"

# AIFin Market 开发者中心发 Key；实际 MCP/Alice 调用走 mcp.wind.com.cn（见 portal wind-mcp-skill / wind-alice 文档）
DEFAULT_API_BASE = "https://aifinmarket.wind.com.cn"
DEFAULT_ALICE_URL = "https://mcp.wind.com.cn/skills/alice"
DEFAULT_MCP_SERVERS: Dict[str, str] = {
    "stock_data": "https://mcp.wind.com.cn/vserver_stock_data/mcp/",
    "fund_data": "https://mcp.wind.com.cn/vserver_fund_data/mcp/",
    "index_data": "https://mcp.wind.com.cn/vserver_index_data/mcp/",
    "bond_data": "https://mcp.wind.com.cn/vserver_bond_data/mcp/",
    "financial_docs": "https://mcp.wind.com.cn/vserver_financial_docs/mcp/",
    "economic_data": "https://mcp.wind.com.cn/vserver_economic_data/mcp/",
    "analytics_data": "https://mcp.wind.com.cn/vserver_analytics_data/mcp/",
}


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_wind_config() -> Dict[str, Any]:
    if not EXAMPLE_PATH.exists():
        raise FileNotFoundError(f"缺少 {EXAMPLE_PATH}")
    cfg = dict(_read_json(EXAMPLE_PATH))
    if LOCAL_PATH.exists():
        cfg.update(_read_json(LOCAL_PATH))
    env_key = os.environ.get("WIND_API_KEY", "").strip()
    if env_key:
        cfg["wind_api_key"] = env_key
    embed_env = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if embed_env and not cfg.get("embedding_api_key"):
        cfg["embedding_api_key"] = embed_env
    return normalize_wind_config(cfg)


def normalize_wind_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """按 AIFin Market 文档补齐端点：Key 来自 portal，调用走 mcp.wind.com.cn。"""
    out = dict(cfg)
    out["wind_api_base"] = (out.get("wind_api_base") or DEFAULT_API_BASE).rstrip("/")
    out["wind_alice_api_url"] = (out.get("wind_alice_api_url") or DEFAULT_ALICE_URL).strip()
    servers = dict(DEFAULT_MCP_SERVERS)
    custom = out.get("wind_mcp_servers")
    if isinstance(custom, dict):
        for key, value in custom.items():
            if value:
                servers[key] = str(value).rstrip("/") + "/"
    out["wind_mcp_servers"] = servers
    return out


def save_wind_response_enabled(cfg: Optional[Dict[str, Any]] = None) -> bool:
    cfg = cfg or load_wind_config()
    return bool(cfg.get("save_wind_response", False))
