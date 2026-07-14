#!/usr/bin/env python3
"""视频/音频下载（基于 yt-dlp，支持 B 站等站点）。"""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

ProgressCallback = Callable[[int, str], None]

try:
    import yt_dlp
except ImportError:
    yt_dlp = None  # type: ignore

BASE_DIR = Path(__file__).resolve().parent
DOWNLOAD_DIR = BASE_DIR / "download"
DEFAULT_COOKIE_FILE = BASE_DIR / "bilibili_cookies.local.txt"
AUDIO_EXTENSIONS = {".mp3", ".m4a", ".wav", ".opus", ".ogg", ".flac", ".aac", ".wma"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".flv", ".avi", ".mov"}


class DownloadError(Exception):
    pass


@dataclass
class FormatOption:
    index: int
    label: str
    format_spec: str
    audio_codec: Optional[str] = None
    need_ffmpeg: bool = False


def _ensure_ytdlp() -> None:
    if yt_dlp is None:
        raise DownloadError("缺少 yt-dlp，请先执行: pip install -r tool/requirements.txt")


def _human_size(num: Optional[int]) -> str:
    if not num:
        return "未知大小"
    units = ["B", "KB", "MB", "GB"]
    size = float(num)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{num}B"


def _is_bilibili_url(url: str) -> bool:
    return "bilibili.com" in url.lower()


def resolve_cookiefile(cookie_file: Optional[str] = None) -> Optional[Path]:
    """解析 Cookie 文件路径（相对 tool 目录或绝对路径）。"""
    if cookie_file and cookie_file.strip():
        path = Path(cookie_file.strip())
        if not path.is_absolute():
            path = BASE_DIR / path
        return path if path.is_file() else None
    return DEFAULT_COOKIE_FILE if DEFAULT_COOKIE_FILE.is_file() else None


def _audio_format_spec(url: str) -> str:
    if _is_bilibili_url(url):
        return "bv*+ba/b"
    return "bestaudio/best"


def base_ydl_opts(url: str, cookie_file: Optional[str] = None) -> Dict[str, Any]:
    opts: Dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }
    if _is_bilibili_url(url):
        opts["http_headers"] = {
            "Origin": "https://www.bilibili.com",
            "Referer": "https://www.bilibili.com/",
        }
        cookiefile = resolve_cookiefile(cookie_file)
        if cookiefile:
            opts["cookiefile"] = str(cookiefile)
    return opts


def extract_info(url: str, cookie_file: Optional[str] = None) -> Dict[str, Any]:
    _ensure_ytdlp()
    opts = base_ydl_opts(url, cookie_file)
    opts["extract_flat"] = False
    with yt_dlp.YoutubeDL(opts) as ydl:
        return ydl.extract_info(url, download=False)


def build_format_options(info: Dict[str, Any]) -> List[FormatOption]:
    options: List[FormatOption] = []
    idx = 1

    options.append(
        FormatOption(index=idx, label="最佳 MP4（视频+音频，推荐）", format_spec="bv*+ba/b")
    )
    idx += 1

    seen_heights: set[int] = set()
    video_candidates: List[Dict[str, Any]] = []
    for fmt in info.get("formats") or []:
        if fmt.get("vcodec") in (None, "none"):
            continue
        height = fmt.get("height")
        ext = (fmt.get("ext") or "").lower()
        if not height or height in seen_heights:
            continue
        if ext not in ("mp4", "flv", "mkv", "webm"):
            continue
        seen_heights.add(height)
        video_candidates.append(fmt)

    for fmt in sorted(video_candidates, key=lambda x: x.get("height") or 0, reverse=True):
        height = fmt.get("height")
        ext = (fmt.get("ext") or "mp4").upper()
        fps = fmt.get("fps")
        size = _human_size(fmt.get("filesize") or fmt.get("filesize_approx"))
        fps_text = f" {fps}fps" if fps else ""
        options.append(
            FormatOption(
                index=idx,
                label=f"MP4 {height}p{fps_text} ({ext}, 约 {size})",
                format_spec=f"bestvideo[height<={height}]+bestaudio/best[height<={height}]",
            )
        )
        idx += 1

    options.append(
        FormatOption(
            index=idx,
            label="MP3 音频（需 ffmpeg）",
            format_spec="bestaudio/best",
            audio_codec="mp3",
            need_ffmpeg=True,
        )
    )
    idx += 1

    options.append(
        FormatOption(
            index=idx,
            label="M4A 音频（需 ffmpeg）",
            format_spec="bestaudio/best",
            audio_codec="m4a",
            need_ffmpeg=True,
        )
    )
    return options


def list_downloaded_audio() -> List[Dict[str, str]]:
    if not DOWNLOAD_DIR.is_dir():
        return []
    files: List[Path] = []
    for ext in AUDIO_EXTENSIONS:
        files.extend(DOWNLOAD_DIR.glob(f"*{ext}"))
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [{"name": p.name, "path": str(p.relative_to(BASE_DIR)).replace("\\", "/")} for p in files]


def _make_download_hook(on_progress: Optional[ProgressCallback]) -> Callable[[Dict[str, Any]], None]:
    last_pct = [-1]

    def hook(data: Dict[str, Any]) -> None:
        if not on_progress:
            return
        status = data.get("status")
        if status == "downloading":
            total = data.get("total_bytes") or data.get("total_bytes_estimate")
            downloaded = data.get("downloaded_bytes") or 0
            if total:
                pct = int(downloaded * 100 / total)
            else:
                pct = last_pct[0] if last_pct[0] >= 0 else 5
            pct = max(1, min(99, pct))
            if pct != last_pct[0]:
                last_pct[0] = pct
                speed = data.get("_speed_str") or ""
                on_progress(pct, f"下载中 {data.get('_percent_str', '')} {speed}".strip())
        elif status == "finished":
            on_progress(99, "正在合并文件...")

    return hook


def download(
    url: str,
    option: FormatOption,
    cookie_file: Optional[str] = None,
    on_progress: Optional[ProgressCallback] = None,
) -> Path:
    _ensure_ytdlp()
    if option.need_ffmpeg and not shutil.which("ffmpeg"):
        raise DownloadError("未检测到 ffmpeg，无法导出 MP3/M4A")

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    outtmpl = str(DOWNLOAD_DIR / "%(title).80B [%(id)s].%(ext)s")

    ydl_opts: Dict[str, Any] = {
        **base_ydl_opts(url, cookie_file),
        "format": _audio_format_spec(url) if option.audio_codec else option.format_spec,
        "outtmpl": outtmpl,
    }
    if on_progress:
        ydl_opts["progress_hooks"] = [_make_download_hook(on_progress)]

    if option.audio_codec:
        ydl_opts["postprocessors"] = [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": option.audio_codec,
                "preferredquality": "192",
            }
        ]

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filepath = Path(ydl.prepare_filename(info))
        if option.audio_codec:
            audio_path = filepath.with_suffix(f".{option.audio_codec}")
            if audio_path.exists():
                return audio_path
        return filepath


def download_by_format_index(
    url: str,
    format_index: int,
    options: List[FormatOption],
    cookie_file: Optional[str] = None,
    on_progress: Optional[ProgressCallback] = None,
) -> Path:
    selected = next((o for o in options if o.index == format_index), None)
    if not selected:
        raise DownloadError(f"无效的格式序号: {format_index}")
    return download(url, selected, cookie_file=cookie_file, on_progress=on_progress)


def _prompt_url() -> str:
    while True:
        url = input("\n请输入视频 URL: ").strip()
        if url:
            return url
        print("URL 不能为空。")


def _choose_option(options: List[FormatOption]) -> Optional[FormatOption]:
    valid = {opt.index for opt in options}
    while True:
        raw = input("\n请输入序号 (0 取消): ").strip()
        if raw == "0":
            return None
        try:
            choice = int(raw)
        except ValueError:
            print("请输入有效数字。")
            continue
        if choice in valid:
            return next(opt for opt in options if opt.index == choice)
        print(f"无效序号，请输入 0 或 {min(valid)}-{max(valid)}。")


def main() -> None:
    if yt_dlp is None:
        print("缺少 yt-dlp，请先执行: pip install -r tool/requirements.txt")
        sys.exit(1)

    print("=" * 56)
    print("  yt-dlp 下载工具（CLI）— Web 版请运行: python tool/app.py")
    print("=" * 56)

    url = _prompt_url()
    cookie = input(f"Cookie 文件 (默认 {DEFAULT_COOKIE_FILE.name}，回车跳过): ").strip() or None
    cookie_resolved = resolve_cookiefile(cookie)
    if _is_bilibili_url(url):
        print(f"B 站 Cookie: {cookie_resolved.name if cookie_resolved else '未配置'}")

    try:
        info = extract_info(url, cookie)
        options = build_format_options(info)
        print(f"\n标题: {info.get('title') or '未知'}")
        for opt in options:
            print(f"  [{opt.index}] {opt.label}")
        selected = _choose_option(options)
        if not selected:
            print("已取消。")
            return
        saved = download(url, selected, cookie_file=cookie)
    except Exception as exc:
        print(f"\n失败: {exc}")
        sys.exit(1)

    print(f"\n下载完成: {saved}")


if __name__ == "__main__":
    main()
