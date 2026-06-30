#!/usr/bin/env python3
"""交互式视频/音频下载工具（基于 yt-dlp，支持 B 站等站点）。"""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    import yt_dlp
except ImportError:
    print("缺少依赖，请先执行: pip install -r tool/requirements.txt")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("缺少 tqdm，请先执行: pip install -r tool/requirements.txt")
    sys.exit(1)

BASE_DIR = Path(__file__).resolve().parent
DOWNLOAD_DIR = BASE_DIR / "download"


@dataclass
class FormatOption:
    index: int
    label: str
    format_spec: str
    audio_codec: Optional[str] = None
    need_ffmpeg: bool = False


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


def _prompt_url() -> str:
    while True:
        url = input("\n请输入视频 URL（B 站示例: https://www.bilibili.com/video/BVxxx）: ").strip()
        if url:
            return url
        print("URL 不能为空，请重新输入。")


def _extract_info(url: str) -> Dict[str, Any]:
    print("\n正在解析视频信息，请稍候...")
    opts = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "extract_flat": False,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        return ydl.extract_info(url, download=False)


def _build_format_options(info: Dict[str, Any]) -> List[FormatOption]:
    title = info.get("title") or "未知标题"
    duration = info.get("duration") or 0
    print(f"\n标题: {title}")
    if duration:
        print(f"时长: {duration // 60:02d}:{duration % 60:02d}")

    options: List[FormatOption] = []
    idx = 1

    options.append(
        FormatOption(
            index=idx,
            label="最佳 MP4（视频+音频，推荐）",
            format_spec="bv*+ba/b",
        )
    )
    idx += 1

    seen_heights: set[int] = set()
    formats = info.get("formats") or []
    video_candidates: List[Dict[str, Any]] = []
    for fmt in formats:
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
            label="MP3 音频（需本机安装 ffmpeg）",
            format_spec="bestaudio/best",
            audio_codec="mp3",
            need_ffmpeg=True,
        )
    )
    idx += 1

    options.append(
        FormatOption(
            index=idx,
            label="M4A 音频（需本机安装 ffmpeg）",
            format_spec="bestaudio/best",
            audio_codec="m4a",
            need_ffmpeg=True,
        )
    )

    return options


def _print_options(options: List[FormatOption]) -> None:
    print("\n可选下载格式:")
    for opt in options:
        print(f"  [{opt.index}] {opt.label}")
    print("  [0] 退出")


def _choose_option(options: List[FormatOption]) -> Optional[FormatOption]:
    valid = {opt.index for opt in options}
    while True:
        raw = input("\n请输入序号: ").strip()
        if raw == "0":
            return None
        try:
            choice = int(raw)
        except ValueError:
            print("请输入有效数字。")
            continue
        if choice in valid:
            return next(opt for opt in options if opt.index == choice)
        print(f"无效序号，请输入 0 或 {min(valid)}-{max(valid)} 之间的数字。")


def _make_progress_hook() -> Callable[[Dict[str, Any]], None]:
    bar: Optional[tqdm] = None
    last_downloaded = 0

    def hook(data: Dict[str, Any]) -> None:
        nonlocal bar, last_downloaded
        status = data.get("status")
        if status == "downloading":
            total = data.get("total_bytes") or data.get("total_bytes_estimate")
            downloaded = data.get("downloaded_bytes") or 0
            if bar is None:
                desc = (data.get("info_dict") or {}).get("title") or "下载中"
                bar = tqdm(
                    total=total if total else None,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=desc[:40],
                    leave=True,
                )
                last_downloaded = 0
            delta = downloaded - last_downloaded
            if delta > 0:
                bar.update(delta)
                last_downloaded = downloaded
            if total and bar.total != total:
                bar.total = total
                bar.refresh()
        elif status == "finished":
            if bar is not None:
                if bar.total and bar.n < bar.total:
                    bar.update(bar.total - bar.n)
                bar.close()
                bar = None
                last_downloaded = 0
            filename = data.get("filename") or ""
            if filename:
                print(f"\n片段完成: {Path(filename).name}")

    return hook


def _download(url: str, option: FormatOption) -> Path:
    if option.need_ffmpeg and not shutil.which("ffmpeg"):
        print("\n错误: 未检测到 ffmpeg，无法导出 MP3/M4A。请先安装 ffmpeg 并加入 PATH。")
        sys.exit(1)

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    outtmpl = str(DOWNLOAD_DIR / "%(title).80B [%(id)s].%(ext)s")

    ydl_opts: Dict[str, Any] = {
        "format": option.format_spec,
        "outtmpl": outtmpl,
        "noplaylist": True,
        "progress_hooks": [_make_progress_hook()],
        "quiet": True,
        "no_warnings": True,
    }

    if option.audio_codec:
        ydl_opts["postprocessors"] = [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": option.audio_codec,
                "preferredquality": "192",
            }
        ]

    print(f"\n保存目录: {DOWNLOAD_DIR}")
    print(f"开始下载: {option.label}\n")

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filepath = Path(ydl.prepare_filename(info))
        if option.audio_codec:
            audio_path = filepath.with_suffix(f".{option.audio_codec}")
            if audio_path.exists():
                return audio_path
        return filepath


def main() -> None:
    print("=" * 56)
    print("  yt-dlp 交互式下载工具（支持 B 站 / YouTube 等）")
    print("=" * 56)

    url = _prompt_url()
    try:
        info = _extract_info(url)
    except Exception as exc:
        print(f"\n解析失败: {exc}")
        sys.exit(1)

    options = _build_format_options(info)
    _print_options(options)
    selected = _choose_option(options)
    if selected is None:
        print("已取消。")
        return

    try:
        saved = _download(url, selected)
    except Exception as exc:
        print(f"\n下载失败: {exc}")
        sys.exit(1)

    print(f"\n下载完成: {saved}")


if __name__ == "__main__":
    main()
