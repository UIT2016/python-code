import json
from pathlib import Path
from typing import Any, Dict, Optional


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _shallow_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    out.update(override)
    return out


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

    api_key = config.get("api_key")
    if not api_key:
        raise ValueError("config.local.json 中需填写 api_key")

    def _clamp_msg_pagesize(raw: Any) -> int:
        try:
            n = int(raw)
        except (TypeError, ValueError):
            return 30
        return max(1, min(n, 2000))

    msg_pagesize = _clamp_msg_pagesize(config.get("msg_pagesize", 30))

    return {
        "headers": headers,
        "api_key": api_key,
        "api_url": config.get("api_url", "https://api.deepseek.com"),
        "model": config.get("model", "deepseek-v4-pro"),
        "msg_api_url": config.get("msg_api_url", "https://mx2025.hhhuu.com/5/api/msg/list"),
        "msg_pagesize": msg_pagesize,
        "token": config.get("token"),
        "cookie": config.get("cookie"),
    }


def get_headers() -> Dict[str, Any]:
    return load_config()["headers"]


load_headers = get_headers


def get_api_key() -> Optional[str]:
    return load_config()["api_key"]
