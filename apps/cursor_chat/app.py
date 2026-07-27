#!/usr/bin/env python3
"""Flask 网页聊天入口。

用法（在仓库根目录）::

    python -m cursor_chat.app
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Generator

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from flask import Flask, Response, jsonify, render_template, request, stream_with_context

from cursor_chat.chat.cloud_rest import CloudRestClient
from cursor_chat.config import load_chat_config
from cursor_chat.download.ytdlp_util import (
    DOWNLOAD_DIR,
    DownloadError,
    build_format_options,
    download,
    extract_info,
)
from cursor_chat.exceptions import ChatConfigError, ChatError, ChatRunError, ChatStartupError
from cursor_chat.prompts.presets import DEFAULT_MODE, get_preset, list_modes
from cursor_chat.session_store import session_store

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5055


def _json_error(message: str, status: int = 400, **extra: Any):
    payload = {"ok": False, "error": message}
    payload.update(extra)
    return jsonify(payload), status


def _get_json() -> Dict[str, Any]:
    return request.get_json(silent=True) or {}


def _rel_download_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(BASE_DIR.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def create_app() -> Flask:
    app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))

    @app.route("/", methods=["GET"])
    def index():
        return render_template("chat.html")

    @app.route("/api/health", methods=["GET"])
    def health():
        return jsonify(session_store.health())

    @app.route("/api/modes", methods=["GET"])
    def modes():
        return jsonify({"ok": True, "modes": list_modes(), "default": DEFAULT_MODE})

    @app.route("/api/models", methods=["GET"])
    def models():
        current = ""
        try:
            cfg = load_chat_config()
            current = cfg.model or ""
            client = CloudRestClient(cfg.api_key, timeout=float(cfg.timeout))
            ids = client.list_models()
            return jsonify({"ok": True, "models": [{"id": mid} for mid in ids], "current": current})
        except ChatStartupError as exc:
            return _json_error(str(exc), 502, retryable=exc.is_retryable, current=current, models=[])
        except ChatConfigError as exc:
            return _json_error(str(exc), 500, current=current, models=[])
        except ChatError as exc:
            return _json_error(str(exc), 500, current=current, models=[])

    @app.route("/api/model", methods=["POST"])
    def set_model():
        data = _get_json()
        session_id = (data.get("session_id") or "").strip()
        model = (data.get("model") or "").strip()
        if not session_id:
            return _json_error("缺少 session_id")
        if not model:
            return _json_error("缺少 model")
        service = session_store.get(session_id)
        if service is None:
            return _json_error("会话不存在或已过期", 404)
        try:
            agent_id = service.set_model(model)
            return jsonify(
                {
                    "ok": True,
                    "session_id": session_id,
                    "agent_id": agent_id,
                    "model": service.config.model,
                    "mode": service.task_mode,
                }
            )
        except ChatStartupError as exc:
            return _json_error(str(exc), 502, retryable=exc.is_retryable)
        except ChatError as exc:
            return _json_error(str(exc), 500)

    @app.route("/api/session", methods=["POST"])
    def create_session():
        data = _get_json()
        mode = (data.get("mode") or DEFAULT_MODE).strip()
        try:
            session_id, service = session_store.create(mode=mode)
            preset = get_preset(service.task_mode)
            return jsonify(
                {
                    "ok": True,
                    "session_id": session_id,
                    "agent_id": service.agent_id,
                    "model": service.config.model,
                    "mode": service.task_mode,
                    "system_prompt": service.get_system_prompt() or "",
                    "placeholder": preset.placeholder,
                    "hint": preset.hint,
                }
            )
        except ChatStartupError as exc:
            return _json_error(str(exc), 502, retryable=exc.is_retryable)
        except ChatError as exc:
            return _json_error(str(exc), 500)

    @app.route("/api/mode", methods=["POST"])
    def set_mode():
        data = _get_json()
        session_id = (data.get("session_id") or "").strip()
        mode = (data.get("mode") or "").strip()
        if not session_id:
            return _json_error("缺少 session_id")
        if not mode:
            return _json_error("缺少 mode")
        service = session_store.get(session_id)
        if service is None:
            return _json_error("会话不存在或已过期", 404)
        try:
            agent_id = service.set_task_mode(mode, renew_session=True)
            preset = get_preset(service.task_mode)
            return jsonify(
                {
                    "ok": True,
                    "session_id": session_id,
                    "agent_id": agent_id,
                    "model": service.config.model,
                    "mode": service.task_mode,
                    "system_prompt": service.get_system_prompt() or "",
                    "placeholder": preset.placeholder,
                    "hint": preset.hint,
                }
            )
        except ChatError as exc:
            return _json_error(str(exc), 500)

    @app.route("/api/session/clear", methods=["POST"])
    def clear_session():
        data = _get_json()
        session_id = (data.get("session_id") or "").strip()
        if not session_id:
            return _json_error("缺少 session_id")
        service = session_store.get(session_id)
        if service is None:
            return _json_error("会话不存在或已过期", 404)
        try:
            agent_id = service.clear_history()
            return jsonify(
                {
                    "ok": True,
                    "session_id": session_id,
                    "agent_id": agent_id,
                    "mode": service.task_mode,
                }
            )
        except ChatError as exc:
            return _json_error(str(exc), 500)

    @app.route("/api/system-prompt", methods=["POST"])
    def set_system_prompt():
        data = _get_json()
        session_id = (data.get("session_id") or "").strip()
        if not session_id:
            return _json_error("缺少 session_id")
        service = session_store.get(session_id)
        if service is None:
            return _json_error("会话不存在或已过期", 404)
        prompt = data.get("system_prompt")
        if prompt is None:
            return _json_error("缺少 system_prompt")
        service.set_system_prompt(str(prompt))
        return jsonify(
            {
                "ok": True,
                "session_id": session_id,
                "system_prompt": service.get_system_prompt() or "",
            }
        )

    @app.route("/api/chat", methods=["POST"])
    def chat():
        data = _get_json()
        session_id = (data.get("session_id") or "").strip()
        message = (data.get("message") or "").strip()
        use_knowledge = bool(data.get("use_knowledge", False))
        if not session_id:
            return _json_error("缺少 session_id")
        if not message:
            return _json_error("消息不能为空")
        service = session_store.get(session_id)
        if service is None:
            return _json_error("会话不存在或已过期", 404)
        try:
            response = service.chat(message, stream=False, use_knowledge=use_knowledge)
            return jsonify(
                {
                    "ok": True,
                    "session_id": session_id,
                    "agent_id": response.agent_id,
                    "text": response.text,
                }
            )
        except ChatStartupError as exc:
            return _json_error(str(exc), 502, retryable=exc.is_retryable)
        except ChatRunError as exc:
            return _json_error(str(exc), 502, run_id=exc.run_id, agent_id=exc.agent_id)
        except ChatError as exc:
            return _json_error(str(exc), 500)

    @app.route("/api/chat/stream", methods=["POST"])
    def chat_stream():
        data = _get_json()
        session_id = (data.get("session_id") or "").strip()
        message = (data.get("message") or "").strip()
        use_knowledge = bool(data.get("use_knowledge", False))
        if not session_id:
            return _json_error("缺少 session_id")
        if not message:
            return _json_error("消息不能为空")
        service = session_store.get(session_id)
        if service is None:
            return _json_error("会话不存在或已过期", 404)

        def event_stream() -> Generator[str, None, None]:
            try:
                stream = service.chat(message, stream=True, use_knowledge=use_knowledge)
                for chunk in stream:
                    payload = json.dumps({"type": "delta", "text": chunk}, ensure_ascii=False)
                    yield f"data: {payload}\n\n"
                done = json.dumps(
                    {"type": "done", "agent_id": service.agent_id or ""},
                    ensure_ascii=False,
                )
                yield f"data: {done}\n\n"
            except ChatStartupError as exc:
                err = json.dumps(
                    {"type": "error", "error": str(exc), "retryable": exc.is_retryable},
                    ensure_ascii=False,
                )
                yield f"data: {err}\n\n"
            except ChatRunError as exc:
                err = json.dumps(
                    {
                        "type": "error",
                        "error": str(exc),
                        "run_id": exc.run_id,
                        "agent_id": exc.agent_id,
                    },
                    ensure_ascii=False,
                )
                yield f"data: {err}\n\n"
            except ChatError as exc:
                err = json.dumps({"type": "error", "error": str(exc)}, ensure_ascii=False)
                yield f"data: {err}\n\n"

        return Response(
            stream_with_context(event_stream()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.route("/api/download/parse", methods=["POST"])
    def download_parse():
        data = _get_json()
        url = (data.get("url") or "").strip()
        if not url:
            return _json_error("缺少 url")
        try:
            info = extract_info(url)
            options = build_format_options(info)
            return jsonify(
                {
                    "ok": True,
                    "title": info.get("title") or "",
                    "id": info.get("id") or "",
                    "formats": [
                        {
                            "index": o.index,
                            "label": o.label,
                            "format_id": o.format_id,
                            "ext": o.ext,
                        }
                        for o in options
                    ],
                }
            )
        except DownloadError as exc:
            return _json_error(str(exc), 502)

    @app.route("/api/download", methods=["POST"])
    def download_file():
        data = _get_json()
        url = (data.get("url") or "").strip()
        format_spec = (data.get("format_spec") or "bestaudio/best").strip() or "bestaudio/best"
        if not url:
            return _json_error("缺少 url")
        try:
            path = download(url, format_spec=format_spec)
            return jsonify(
                {
                    "ok": True,
                    "path": _rel_download_path(path),
                    "abs_path": str(path.resolve()),
                    "name": path.name,
                    "download_dir": str(DOWNLOAD_DIR),
                }
            )
        except DownloadError as exc:
            return _json_error(str(exc), 502)

    return app


def main() -> None:
    app = create_app()
    print(f"cursor_chat Web: http://{DEFAULT_HOST}:{DEFAULT_PORT}")
    app.run(host=DEFAULT_HOST, port=DEFAULT_PORT, debug=False, threaded=True)


if __name__ == "__main__":
    main()
