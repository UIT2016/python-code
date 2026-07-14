"""cursor_chat 内独立的 yt-dlp 薄封装（不依赖 tool/ 模块）。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
DOWNLOAD_DIR = BASE_DIR / "download_out"


@dataclass
class FormatOption:
    index: int
    label: str
    format_id: str
    ext: str = ""
    filesize: Optional[int] = None


class DownloadError(Exception):
    pass


def _ensure_yt_dlp():
    try:
        import yt_dlp  # noqa: F401
    except ImportError as exc:
        raise DownloadError("未安装 yt-dlp，请执行: pip install -r cursor_chat/requirements.txt") from exc


def _human_size(num: Optional[int]) -> str:
    if not num:
        return "未知大小"
    size = float(num)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{num}B"


def extract_info(url: str) -> Dict[str, Any]:
    _ensure_yt_dlp()
    import yt_dlp

    opts: Dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
    }
    if "bilibili.com" in url.lower():
        opts["http_headers"] = {
            "Referer": "https://www.bilibili.com",
            "Origin": "https://www.bilibili.com",
        }
    with yt_dlp.YoutubeDL(opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
        except Exception as exc:
            raise DownloadError(f"解析失败: {exc}") from exc
    if not isinstance(info, dict):
        raise DownloadError("解析结果无效")
    return info


def build_format_options(info: Dict[str, Any]) -> List[FormatOption]:
    options: List[FormatOption] = [
        FormatOption(index=0, label="默认：仅音频 (bestaudio)", format_id="bestaudio/best", ext="audio"),
        FormatOption(index=1, label="默认：最佳视频+音频", format_id="bv*+ba/b", ext="video"),
    ]
    formats = info.get("formats") or []
    idx = 2
    for fmt in formats:
        if not isinstance(fmt, dict):
            continue
        fid = str(fmt.get("format_id") or "")
        if not fid:
            continue
        height = fmt.get("height")
        acodec = fmt.get("acodec")
        vcodec = fmt.get("vcodec")
        ext = str(fmt.get("ext") or "")
        note = fmt.get("format_note") or ""
        size = fmt.get("filesize") or fmt.get("filesize_approx")
        kind = []
        if vcodec and vcodec != "none":
            kind.append(f"{height}p" if height else "video")
        if acodec and acodec != "none":
            kind.append("audio")
        label = f"{fid} | {'+'.join(kind) or 'fmt'} | {ext} | {note} | {_human_size(size)}"
        options.append(FormatOption(index=idx, label=label, format_id=fid, ext=ext, filesize=size))
        idx += 1
        if idx >= 40:
            break
    return options


def download(
    url: str,
    *,
    format_spec: str = "bestaudio/best",
    out_dir: Optional[Path] = None,
) -> Path:
    _ensure_yt_dlp()
    import yt_dlp

    target = Path(out_dir) if out_dir else DOWNLOAD_DIR
    target.mkdir(parents=True, exist_ok=True)
    outtmpl = str(target / "%(title).80B [%(id)s].%(ext)s")

    opts: Dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "outtmpl": outtmpl,
        "format": format_spec,
    }
    if "bilibili.com" in url.lower():
        opts["http_headers"] = {
            "Referer": "https://www.bilibili.com",
            "Origin": "https://www.bilibili.com",
        }
        # 仅音频时尽量合并轨再抽
        if format_spec.startswith("bestaudio"):
            opts["format"] = "bv*+ba/b"
            opts["postprocessors"] = [
                {"key": "FFmpegExtractAudio", "preferredcodec": "m4a", "preferredquality": "192"}
            ]

    before = {p.resolve() for p in target.glob("*") if p.is_file()}
    with yt_dlp.YoutubeDL(opts) as ydl:
        try:
            ydl.download([url])
        except Exception as exc:
            raise DownloadError(f"下载失败: {exc}") from exc

    after = [p for p in target.glob("*") if p.is_file() and p.resolve() not in before]
    if not after:
        # 回退：取目录最新文件
        files = sorted(target.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
            raise DownloadError("下载完成但未找到输出文件（可能需要本机 ffmpeg）")
        return files[0]
    after.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return after[0]
