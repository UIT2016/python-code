import json
from pathlib import Path
from typing import Any, Dict, Optional

LLM_PRESETS: Dict[str, Dict[str, str]] = {
    "deepseek": {
        "api_url": "https://api.deepseek.com",
        "model": "deepseek-v4-pro",
    },
    "qwen": {
        "api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
    },
}
SUPPORTED_LLM_PROVIDERS = tuple(LLM_PRESETS.keys())


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _shallow_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    out.update(override)
    return out


def _normalize_provider(name: Any) -> str:
    provider = str(name or "deepseek").strip().lower()
    if provider not in LLM_PRESETS:
        raise ValueError(f"不支持的 llm_provider: {provider}，可选: {', '.join(SUPPORTED_LLM_PROVIDERS)}")
    return provider


def _provider_block(config: Dict[str, Any], provider: str) -> Dict[str, Any]:
    preset = dict(LLM_PRESETS[provider])
    nested = (config.get("llm") or {}).get(provider) or {}
    if isinstance(nested, dict):
        preset.update({k: v for k, v in nested.items() if v not in (None, "")})

    if provider == "deepseek":
        for key in ("api_key", "api_url", "model"):
            legacy = config.get(key)
            if legacy and not preset.get(key):
                preset[key] = legacy
    return preset


def resolve_llm_config(config: Dict[str, Any], provider: Optional[str] = None) -> Dict[str, Any]:
    """解析 LLM 配置，返回当前 provider 及 deepseek/qwen 两套完整参数。"""
    active_provider = _normalize_provider(provider or config.get("llm_provider", "deepseek"))
    providers: Dict[str, Dict[str, str]] = {}
    for name in SUPPORTED_LLM_PROVIDERS:
        block = _provider_block(config, name)
        providers[name] = {
            "api_key": str(block.get("api_key") or ""),
            "api_url": str(block.get("api_url") or LLM_PRESETS[name]["api_url"]),
            "model": str(block.get("model") or LLM_PRESETS[name]["model"]),
        }

    active = providers[active_provider]
    if not active["api_key"]:
        raise ValueError(
            f"config.local.json 中需填写 {active_provider} 的 api_key"
            f"（llm.{active_provider}.api_key 或 deepseek 旧字段 api_key）"
        )
    return {
        "provider": active_provider,
        "providers": providers,
        "api_key": active["api_key"],
        "api_url": active["api_url"],
        "model": active["model"],
    }


def load_config(base_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    从仓库内示例与本地文件加载配置：
      - config.example.json：结构与非敏感默认值（可提交）
      - config.local.json：token、cookie、api_key 等（勿提交，见 .gitignore）
      - header.example.json：请求头模板，含 {{TOKEN}}、{{COOKIE}}
      - header.local.json：可选，覆盖或增补请求头键值
    """
    base = base_dir or Path(__file__).resolve().parent
    example_cfg = base / "config.example.json"
    local_cfg = base / "config.local.json"
    example_hdr = base / "header.example.json"
    local_hdr = base / "header.local.json"

    if not example_cfg.exists():
        raise FileNotFoundError(f"缺少示例配置 {example_cfg}")
    if not local_cfg.exists():
        raise FileNotFoundError(
            f"缺少本地配置 {local_cfg}，请复制 config.example.json 为 config.local.json 并填写 token、cookie、api_key 等"
        )
    if not example_hdr.exists():
        raise FileNotFoundError(f"缺少请求头示例 {example_hdr}")

    config = _shallow_merge(_read_json(example_cfg), _read_json(local_cfg))
    headers_template = _read_json(example_hdr)
    if local_hdr.exists():
        headers_template = _shallow_merge(headers_template, _read_json(local_hdr))

    headers: Dict[str, Any] = {}
    for key, value in headers_template.items():
        if isinstance(value, str):
            value = value.replace("{{TOKEN}}", str(config.get("token", "")))
            value = value.replace("{{COOKIE}}", str(config.get("cookie", "")))
        headers[key] = value

    llm = resolve_llm_config(config)

    def _clamp_msg_pagesize(raw: Any) -> int:
        try:
            n = int(raw)
        except (TypeError, ValueError):
            return 30
        return max(1, min(n, 2000))

    msg_pagesize = _clamp_msg_pagesize(config.get("msg_pagesize", 30))

    return {
        "headers": headers,
        "llm_provider": llm["provider"],
        "llm": llm["providers"],
        "api_key": llm["api_key"],
        "api_url": llm["api_url"],
        "model": llm["model"],
        "msg_api_url": config.get("msg_api_url", "https://mx2025.hhhuu.com/5/api/msg/list"),
        "msg_pagesize": msg_pagesize,
        "token": config.get("token"),
        "cookie": config.get("cookie"),
    }


def get_headers() -> Dict[str, Any]:
    return load_config()["headers"]


load_headers = get_headers


def get_api_key(provider: Optional[str] = None) -> Optional[str]:
    cfg = load_config()
    if provider:
        name = _normalize_provider(provider)
        return cfg["llm"][name]["api_key"]
    return cfg["api_key"]
