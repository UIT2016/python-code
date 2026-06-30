"""使用 Qwen3-ASR-Flash-Filetrans 将本地音频转写为文字。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import requests
except ImportError:
    raise SystemExit("缺少 requests，请先执行: pip install -r tool/requirements.txt")

BASE_DIR = Path(__file__).resolve().parent
API_KEY_FILE = BASE_DIR / "api_key.local.json"
TRANSCRIPT_DIR = BASE_DIR / "transcripts"
ASR_MODEL = "qwen3-asr-flash-filetrans"
POLL_INTERVAL_SEC = 3
POLL_TIMEOUT_SEC = 3600

AUDIO_EXTENSIONS = {".mp3", ".m4a", ".wav", ".opus", ".ogg", ".flac", ".aac", ".wma"}


class AsrError(Exception):
    pass


def load_api_config() -> Dict[str, str]:
    if not API_KEY_FILE.exists():
        example = BASE_DIR / "api_key.example.json"
        hint = f"请复制 {example.name} 为 {API_KEY_FILE.name} 并填入 api_key"
        raise AsrError(hint)
    with API_KEY_FILE.open(encoding="utf-8") as f:
        config = json.load(f)
    api_key = (config.get("api_key") or "").strip()
    if not api_key:
        raise AsrError(f"请在 {API_KEY_FILE.name} 中填写 api_key")
    base_url = (config.get("base_url") or "https://dashscope.aliyuncs.com/api/v1").rstrip("/")
    return {"api_key": api_key, "base_url": base_url}


def _auth_headers(api_key: str, *, resolve_oss: bool = False) -> Dict[str, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if resolve_oss:
        headers["X-DashScope-OssResourceResolve"] = "enable"
    return headers


def upload_local_file(api_key: str, base_url: str, file_path: Path) -> str:
    """上传本地文件到 DashScope 临时 OSS，返回 oss:// URL。"""
    policy_url = f"{base_url}/uploads"
    resp = requests.get(
        policy_url,
        headers=_auth_headers(api_key),
        params={"action": "getPolicy", "model": ASR_MODEL},
        timeout=60,
    )
    if resp.status_code != 200:
        raise AsrError(f"获取上传凭证失败: {resp.text}")
    policy_data = resp.json().get("data") or {}
    file_name = file_path.name
    key = f"{policy_data['upload_dir']}/{file_name}"
    with file_path.open("rb") as file_obj:
        upload_resp = requests.post(
            policy_data["upload_host"],
            files={
                "OSSAccessKeyId": (None, policy_data["oss_access_key_id"]),
                "Signature": (None, policy_data["signature"]),
                "policy": (None, policy_data["policy"]),
                "x-oss-object-acl": (None, policy_data["x_oss_object_acl"]),
                "x-oss-forbid-overwrite": (None, policy_data["x_oss_forbid_overwrite"]),
                "key": (None, key),
                "success_action_status": (None, "200"),
                "file": (file_name, file_obj),
            },
            timeout=600,
        )
    if upload_resp.status_code != 200:
        raise AsrError(f"上传音频失败: {upload_resp.text}")
    return f"oss://{key}"


def submit_transcription_task(api_key: str, base_url: str, file_url: str) -> str:
    url = f"{base_url}/services/audio/asr/transcription"
    headers = _auth_headers(api_key, resolve_oss=True)
    headers["X-DashScope-Async"] = "enable"
    payload = {
        "model": ASR_MODEL,
        "input": {"file_url": file_url},
        "parameters": {"channel_id": [0], "enable_itn": True},
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        raise AsrError(f"提交转写任务失败: {resp.text}")
    output = resp.json().get("output") or {}
    task_id = output.get("task_id")
    if not task_id:
        raise AsrError(f"未获取到 task_id: {resp.text}")
    return task_id


def poll_task_result(api_key: str, base_url: str, task_id: str) -> Dict[str, Any]:
    url = f"{base_url}/tasks/{task_id}"
    headers = _auth_headers(api_key)
    deadline = time.monotonic() + POLL_TIMEOUT_SEC
    while time.monotonic() < deadline:
        resp = requests.get(url, headers=headers, timeout=60)
        if resp.status_code != 200:
            raise AsrError(f"查询任务失败: {resp.text}")
        data = resp.json()
        output = data.get("output") or {}
        status = (output.get("task_status") or "").upper()
        if status == "SUCCEEDED":
            return data
        if status in ("FAILED", "UNKNOWN"):
            message = output.get("message") or data.get("message") or resp.text
            raise AsrError(f"转写任务失败 ({status}): {message}")
        time.sleep(POLL_INTERVAL_SEC)
    raise AsrError(f"转写超时（>{POLL_TIMEOUT_SEC} 秒），task_id={task_id}")


def fetch_transcription_json(transcription_url: str) -> Dict[str, Any]:
    resp = requests.get(transcription_url, timeout=120)
    if resp.status_code != 200:
        raise AsrError(f"下载识别结果失败: {resp.text}")
    return resp.json()


def _format_ms(ms: int) -> str:
    sec = ms // 1000
    return f"{sec // 60:02d}:{sec % 60:02d}"


def extract_texts(result: Dict[str, Any]) -> tuple[str, str]:
    """返回 (纯文本, 带时间戳文本)。"""
    transcripts = result.get("transcripts") or []
    plain_parts: List[str] = []
    timed_parts: List[str] = []
    for track in transcripts:
        text = (track.get("text") or "").strip()
        if text:
            plain_parts.append(text)
        for sentence in track.get("sentences") or []:
            sent_text = (sentence.get("text") or "").strip()
            if not sent_text:
                continue
            begin = sentence.get("begin_time", 0)
            end = sentence.get("end_time", 0)
            timed_parts.append(f"[{_format_ms(begin)}-{_format_ms(end)}] {sent_text}")
    plain = "\n".join(plain_parts)
    timed = "\n".join(timed_parts) if timed_parts else plain
    return plain, timed


def save_transcript(
    source_file: Path,
    plain_text: str,
    timed_text: str,
    raw_result: Optional[Dict[str, Any]] = None,
) -> Path:
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    stem = source_file.stem
    txt_path = TRANSCRIPT_DIR / f"{stem}.txt"
    timed_path = TRANSCRIPT_DIR / f"{stem}.timed.txt"
    txt_path.write_text(plain_text, encoding="utf-8")
    timed_path.write_text(timed_text, encoding="utf-8")
    if raw_result is not None:
        json_path = TRANSCRIPT_DIR / f"{stem}.json"
        json_path.write_text(json.dumps(raw_result, ensure_ascii=False, indent=2), encoding="utf-8")
    return txt_path


def transcribe_audio_file(audio_path: Path) -> Path:
    if not audio_path.exists():
        raise AsrError(f"音频文件不存在: {audio_path}")
    if audio_path.suffix.lower() not in AUDIO_EXTENSIONS:
        raise AsrError(f"不支持的音频格式: {audio_path.suffix}")

    config = load_api_config()
    api_key = config["api_key"]
    base_url = config["base_url"]

    print("\n正在上传音频到 DashScope 临时存储...")
    oss_url = upload_local_file(api_key, base_url, audio_path)
    print("上传完成，正在提交转写任务...")
    task_id = submit_transcription_task(api_key, base_url, oss_url)
    print(f"任务已提交 (task_id: {task_id})，等待识别结果...")
    task_result = poll_task_result(api_key, base_url, task_id)
    output = task_result.get("output") or {}
    result_block = output.get("result") or {}
    transcription_url = result_block.get("transcription_url")
    if not transcription_url:
        raise AsrError(f"未获取到 transcription_url: {json.dumps(task_result, ensure_ascii=False)}")
    print("识别完成，正在下载并保存文字...")
    raw = fetch_transcription_json(transcription_url)
    plain, timed = extract_texts(raw)
    if not plain.strip():
        raise AsrError("识别结果为空")
    saved = save_transcript(audio_path, plain, timed, raw)
    return saved
