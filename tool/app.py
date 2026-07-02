import os
import sys
import asyncio
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from asr_transcriber import resolve_audio_path, transcribe_audio_file, transcribe_audio_files
from task_store import task_store
from transcript_agent.agents.rule_curator import RuleCuratorAgent, load_active_rubric
from transcript_agent.base import PROCESSED_DIR, TRANSCRIPT_DIR
from transcript_agent.llm_config import create_llm_client_from_cfg, load_llm_config
from transcript_agent.pipeline import (
    TranscriptOrchestrator,
    list_processed_files,
    list_transcript_files,
)
from ytdlp_downloader import (
    DEFAULT_COOKIE_FILE,
    DownloadError,
    build_format_options,
    download_by_format_index,
    extract_info,
    list_downloaded_audio,
    resolve_cookiefile,
)

DEFAULT_COOKIE_NAME = DEFAULT_COOKIE_FILE.name


def _format_options_to_dict(options) -> List[Dict[str, Any]]:
    return [asdict(o) for o in options]


def _result_to_dict(result) -> Dict[str, Any]:
    d = {
        "file_name": result.file_name,
        "elapsed_sec": result.elapsed_sec,
        "txt": result.txt_path.name,
    }
    if result.timed_path:
        d["timed"] = result.timed_path.name
    if result.json_path:
        d["json"] = result.json_path.name
    return d


def _run_download_task(
    task_id: str,
    url: str,
    format_index: int,
    cookie_file: str,
) -> None:
    updater = task_store.make_updater(task_id)
    try:
        updater(1, "正在解析视频信息...")
        info = extract_info(url, cookie_file)
        options = build_format_options(info)
        updater(3, "开始下载...")
        saved = download_by_format_index(
            url,
            format_index,
            options,
            cookie_file,
            on_progress=updater,
        )
        rel = str(saved.relative_to(BASE_DIR)).replace("\\", "/")
        task_store.update(
            task_id,
            status="done",
            progress=100,
            message=f"下载完成: {saved.name}",
            result={"path": rel, "name": saved.name},
        )
    except Exception as exc:
        task_store.update(task_id, status="error", progress=100, message="下载失败", error=str(exc))


def _run_transcribe_task(
    task_id: str,
    audio_paths: List[str],
    include_timed: bool,
    include_json: bool,
) -> None:
    updater = task_store.make_updater(task_id)
    try:
        paths = [resolve_audio_path(p) for p in audio_paths]
        if len(paths) == 1:
            result = transcribe_audio_file(
                paths[0],
                include_timed=include_timed,
                include_json=include_json,
                on_progress=updater,
            )
            task_store.update(
                task_id,
                status="done",
                progress=100,
                message=f"转写完成，耗时 {result.elapsed_sec}s",
                result={"items": [_result_to_dict(result)], "elapsed_sec": result.elapsed_sec},
            )
        else:
            results = transcribe_audio_files(
                paths,
                include_timed=include_timed,
                include_json=include_json,
                on_progress=updater,
            )
            total_elapsed = round(sum(r.elapsed_sec for r in results), 1)
            task_store.update(
                task_id,
                status="done",
                progress=100,
                message=f"批量转写完成，共 {len(results)} 个文件，总耗时 {total_elapsed}s",
                result={
                    "items": [_result_to_dict(r) for r in results],
                    "elapsed_sec": total_elapsed,
                    "count": len(results),
                },
            )
    except Exception as exc:
        task_store.update(task_id, status="error", progress=100, message="转写失败", error=str(exc))


def _resolve_transcript_path(relative_or_name: str) -> Path:
    raw = relative_or_name.strip()
    path = Path(raw)
    if not path.is_absolute():
        path = BASE_DIR / raw
    path = path.resolve()
    if path.parent != TRANSCRIPT_DIR.resolve():
        raise ValueError("仅允许处理 transcripts 目录下的 .txt 文件")
    if not path.is_file():
        raise FileNotFoundError(f"转录文件不存在: {path.name}")
    if path.suffix.lower() != ".txt" or path.name.endswith(".timed.txt"):
        raise ValueError("请选择 .txt 转录文件（非 .timed.txt）")
    return path


def _run_refine_task(task_id: str, transcript_paths: List[str], provider: str) -> None:
    updater = task_store.make_updater(task_id)
    try:
        orchestrator = TranscriptOrchestrator(provider=provider or None)
        results = []
        total = len(transcript_paths)
        for idx, rel in enumerate(transcript_paths):
            base_pct = int(idx * 100 / total)

            def file_progress(pct: int, msg: str, _base: int = base_pct, _idx: int = idx) -> None:
                overall = _base + int(pct / total)
                updater(overall, f"[{_idx + 1}/{total}] {msg}")

            path = _resolve_transcript_path(rel)
            updater(base_pct, f"[{idx + 1}/{total}] 开始精炼 {path.name}")

            async def _run_one() -> Dict[str, Any]:
                return await orchestrator.run(path, on_progress=file_progress)

            result = asyncio.run(_run_one())
            results.append(result)

        passed_count = sum(1 for r in results if r.get("passed"))
        task_store.update(
            task_id,
            status="done",
            progress=100,
            message=f"精炼完成：{passed_count}/{total} 通过审计",
            result={"items": results, "count": total, "passed_count": passed_count},
        )
    except Exception as exc:
        task_store.update(task_id, status="error", progress=100, message="精炼失败", error=str(exc))


def _run_refresh_rules_task(task_id: str, provider: str) -> None:
    updater = task_store.make_updater(task_id)
    try:
        updater(10, "加载配置与知识库...")

        async def _run() -> Dict[str, Any]:
            cfg = load_llm_config(provider or None)
            llm = create_llm_client_from_cfg(cfg, provider=cfg["active_provider"], temperature=0.2)
            return await RuleCuratorAgent(llm).refresh_rubric()

        rubric = asyncio.run(_run())
        updater(100, f"规则已更新: {rubric.get('version')}")
        task_store.update(
            task_id,
            status="done",
            progress=100,
            message=f"审计规则已刷新: {rubric.get('version')}",
            result={"version": rubric.get("version"), "pass_threshold": rubric.get("pass_threshold")},
        )
    except Exception as exc:
        task_store.update(task_id, status="error", progress=100, message="规则刷新失败", error=str(exc))


def create_app() -> Flask:
    app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))
    app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-tool-web-change-in-prod")

    @app.route("/", methods=["GET"])
    def index():
        return render_template(
            "index.html",
            audio_files=list_downloaded_audio(),
            transcript_files=list_transcript_files(),
            default_cookie=DEFAULT_COOKIE_NAME,
        )

    @app.route("/api/parse", methods=["POST"])
    def api_parse():
        data = request.get_json(silent=True) or {}
        url = (data.get("url") or "").strip()
        cookie_file = (data.get("cookie_file") or DEFAULT_COOKIE_NAME).strip()
        if not url:
            return jsonify({"error": "URL 不能为空"}), 400
        try:
            info = extract_info(url, cookie_file)
            options = build_format_options(info)
            return jsonify(
                {
                    "title": info.get("title") or "未知标题",
                    "duration": info.get("duration") or 0,
                    "options": _format_options_to_dict(options),
                    "cookie_ok": resolve_cookiefile(cookie_file) is not None,
                }
            )
        except DownloadError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            return jsonify({"error": f"解析失败: {exc}"}), 500

    @app.route("/api/audio-files")
    def api_audio_files():
        return jsonify(list_downloaded_audio())

    @app.route("/api/download", methods=["POST"])
    def api_download():
        data = request.get_json(silent=True) or {}
        url = (data.get("url") or "").strip()
        cookie_file = (data.get("cookie_file") or DEFAULT_COOKIE_NAME).strip()
        label = (data.get("label") or url[:60] or "视频下载").strip()
        try:
            format_index = int(data.get("format_index"))
        except (TypeError, ValueError):
            return jsonify({"error": "无效的 format_index"}), 400
        if not url:
            return jsonify({"error": "URL 不能为空"}), 400

        task_id = task_store.create("download", meta={"label": label})
        threading.Thread(
            target=_run_download_task,
            args=(task_id, url, format_index, cookie_file),
            daemon=True,
        ).start()
        return jsonify({"task_id": task_id})

    @app.route("/api/transcribe", methods=["POST"])
    def api_transcribe():
        data = request.get_json(silent=True) or {}
        audio_files = data.get("audio_files") or []
        if isinstance(audio_files, str):
            audio_files = [audio_files]
        audio_files = [str(f).strip() for f in audio_files if str(f).strip()]
        if not audio_files:
            return jsonify({"error": "请至少选择一个音频文件"}), 400

        include_timed = bool(data.get("include_timed"))
        include_json = bool(data.get("include_json"))
        task_type = "transcribe_batch" if len(audio_files) > 1 else "transcribe"
        names = [Path(p).name for p in audio_files]
        label = (data.get("label") or "").strip() or (names[0] if len(names) == 1 else f"批量转写 {len(names)} 个文件")
        task_id = task_store.create(task_type, meta={"items": audio_files, "label": label})
        threading.Thread(
            target=_run_transcribe_task,
            args=(task_id, audio_files, include_timed, include_json),
            daemon=True,
        ).start()
        return jsonify({"task_id": task_id})

    @app.route("/api/transcript/files")
    def api_transcript_files():
        return jsonify(list_transcript_files())

    @app.route("/api/transcript/rubric")
    def api_transcript_rubric():
        rubric = load_active_rubric()
        return jsonify(rubric)

    @app.route("/api/transcript/manual-rules", methods=["GET"])
    def api_transcript_manual_rules_get():
        return jsonify({"content": RuleCuratorAgent.read_manual_overrides_text()})

    @app.route("/api/transcript/manual-rules", methods=["POST"])
    def api_transcript_manual_rules_save():
        data = request.get_json(silent=True) or {}
        content = data.get("content")
        if content is None:
            return jsonify({"error": "content 不能为空"}), 400
        RuleCuratorAgent.save_manual_overrides_text(str(content))
        return jsonify({"ok": True})

    @app.route("/api/transcript/refresh-rules", methods=["POST"])
    def api_transcript_refresh_rules():
        data = request.get_json(silent=True) or {}
        provider = (data.get("provider") or "").strip()
        task_id = task_store.create("refresh_rules", meta={"label": "刷新审计规则"})
        threading.Thread(
            target=_run_refresh_rules_task,
            args=(task_id, provider),
            daemon=True,
        ).start()
        return jsonify({"task_id": task_id})

    @app.route("/api/transcript/refine", methods=["POST"])
    def api_transcript_refine():
        data = request.get_json(silent=True) or {}
        files = data.get("transcript_files") or []
        if isinstance(files, str):
            files = [files]
        files = [str(f).strip() for f in files if str(f).strip()]
        if not files:
            return jsonify({"error": "请至少选择一个转录文件"}), 400
        provider = (data.get("provider") or "").strip()
        names = [Path(f).name for f in files]
        label = (data.get("label") or "").strip() or (
            names[0] if len(names) == 1 else f"批量精炼 {len(names)} 个文件"
        )
        task_type = "refine_batch" if len(files) > 1 else "refine"
        task_id = task_store.create(task_type, meta={"items": files, "label": label})
        threading.Thread(
            target=_run_refine_task,
            args=(task_id, files, provider),
            daemon=True,
        ).start()
        return jsonify({"task_id": task_id})

    @app.route("/api/transcript/processed")
    def api_transcript_processed():
        return jsonify(list_processed_files())

    @app.route("/api/transcript/processed/<path:name>")
    def api_transcript_processed_content(name: str):
        safe_name = Path(name).name
        path = PROCESSED_DIR / safe_name
        if not path.is_file() or path.resolve().parent != PROCESSED_DIR.resolve():
            return jsonify({"error": "文件不存在"}), 404
        return jsonify({"name": safe_name, "content": path.read_text(encoding="utf-8")})

    @app.route("/api/task/<task_id>")
    def api_task(task_id: str):
        task = task_store.get(task_id)
        if not task:
            return jsonify({"error": "任务不存在", "status": "error"}), 404
        return jsonify(task)

    @app.errorhandler(404)
    def handle_404(err):
        if request.path.startswith("/api/"):
            return jsonify({"error": "接口不存在", "status": "error", "path": request.path}), 404
        if isinstance(err, HTTPException):
            return err.get_response()
        return err

    @app.errorhandler(500)
    def handle_500(err):
        if request.path.startswith("/api/"):
            return jsonify({"error": "服务器内部错误", "status": "error"}), 500
        if isinstance(err, HTTPException):
            return err.get_response()
        return err

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5001, debug=True, threaded=True, use_reloader=False)
