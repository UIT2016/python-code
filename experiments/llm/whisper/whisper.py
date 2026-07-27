import os

import yt_dlp
from faster_whisper import WhisperModel


def get_available_formats(url):
    """获取并展示可用的格式列表"""
    print("\n🔍 正在解析视频格式信息...")

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,  # 必须为 False 才能获取 formats 详情
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

            if not info:
                print("❌ 无法获取视频信息。")
                return None, None

            formats = info.get("formats", [])
            if not formats:
                print("❌ 未找到任何可用格式。")
                return None, None

            # 过滤并整理格式列表，方便用户阅读
            # 我们只展示有清晰度或有音质的关键格式
            visible_formats = []

            for f in formats:
                f_id = f.get("format_id")
                ext = f.get("ext")
                resolution = f.get("resolution") or f.get("height", "")
                if resolution and isinstance(resolution, int):
                    resolution = f"{f.get('width', '?')}x{resolution}"

                # 判断类型
                vcodec = f.get("vcodec")
                acodec = f.get("acodec")

                f_type = "Unknown"
                if vcodec != "none" and acodec != "none":
                    f_type = "🎬 音视频混合"
                elif vcodec != "none":
                    f_type = "📺 纯视频 (无音)"
                elif acodec != "none":
                    f_type = "🎵 纯音频"

                # 构建显示字符串
                note = f.get("note", "")
                display_str = f"[ID: {f_id:<4}] {f_type:<12} | {ext:<4} | {resolution:<12} | {note}"
                visible_formats.append(
                    {"id": f_id, "format_obj": f, "display": display_str, "is_video": vcodec != "none"}
                )

            return info, visible_formats

    except Exception as e:
        print(f"❌ 解析失败: {e}")
        return None, None


def download_selected(url, format_id, output_folder="downloads"):
    """下载指定格式"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 动态构建输出模板，包含格式ID以防重名
    outtmpl = os.path.join(output_folder, "%(title)s_[%(format_id)s].%(ext)s")

    ydl_opts = {
        "format": format_id,
        "outtmpl": outtmpl,
        "quiet": False,  # 下载时显示进度条
        "no_warnings": False,
    }

    print(f"\n🚀 开始下载格式 ID: {format_id} ...")

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            return filename, info
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return None, None


def extract_audio_from_video(video_path, output_mp3_path):
    """使用 yt-dlp (内置 ffmpeg) 从视频中提取音频"""
    print("\n🎬 检测到视频文件，正在提取音频至 MP3...")

    # 使用 yt-dlp 的后处理功能来提取，比直接调 ffmpeg 更稳健
    ydl_opts = {
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "outtmpl": output_mp3_path.replace(".mp3", ""),  # 去掉扩展名让 yt-dlp 自己加
        "quiet": False,
    }

    # 创建一个临时的 info 字典欺骗 yt-dlp 认为我们要处理这个本地文件
    # 但更简单的方法是直接运行一个针对本地文件的下载任务
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # 传入本地文件路径作为 "url"
            ydl.download([video_path])

        # yt-dlp 提取后通常会生成新的 mp3 文件，我们需要找到它
        # 默认行为是替换原文件还是新建？FFmpegExtractAudio 默认是新建
        expected_mp3 = video_path.rsplit(".", 1)[0] + ".mp3"

        # 如果文件名里有格式ID，逻辑会稍微复杂点，这里简化处理：
        # 直接查找目录下最新生成的 mp3
        base_name = os.path.basename(video_path).rsplit(".", 1)[0]
        dir_name = os.path.dirname(video_path)

        # 尝试几种可能的命名
        candidates = [
            expected_mp3,
            os.path.join(dir_name, base_name.split("_[")[0] + ".mp3"),  # 清理掉格式后缀
        ]

        for cand in candidates:
            if os.path.exists(cand):
                print(f"✅ 音频提取成功: {cand}")
                return cand

        # 如果都没找到，可能是同名覆盖或者是其他情况，报错
        print("⚠️ 未能自动定位生成的 MP3 文件，请检查 downloads 文件夹。")
        return None

    except Exception as e:
        print(f"❌ 音频提取失败: {e}")
        return None


def transcribe_audio(audio_path, model_size="large-v3"):
    """转录音频"""
    print(f"\n🤖 正在加载 Whisper 模型 ({model_size})...")
    print("💡 提示：首次运行会下载模型，之后会缓存。利用 Ryzen AI 9 CPU int8 模式。")

    try:
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None, None

    print("📝 开始转录 (这可能需要几分钟)...")

    # 语言设置为 "zh" 以提高中文准确率，也可以设为 None 自动检测
    segments, info = model.transcribe(audio_path, beam_size=5, language="zh", vad_filter=True)

    base_name = audio_path.rsplit(".", 1)[0]
    output_txt = base_name + ".txt"
    output_srt = base_name + ".srt"

    count = 1
    with open(output_txt, "w", encoding="utf-8") as f, open(output_srt, "w", encoding="utf-8") as srt_f:
        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue

            # 写入 TXT
            f.write(text + "\n")

            # 写入 SRT
            start = format_timestamp(segment.start)
            end = format_timestamp(segment.end)
            srt_f.write(f"{count}\n{start} --> {end}\n{text}\n\n")
            count += 1

    print("✅ 转录完成！")
    return output_txt, output_srt


def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def main():
    print("=" * 60)
    print("🎥 视频下载 & 转录助手 (交互式版)")
    print("=" * 60)

    url = input("\n🔗 请输入视频网址 (B站/YouTube/抖音等): ").strip()
    if not url:
        print("❌ 未输入网址。")
        return

    # 1. 获取格式列表
    info, formats = get_available_formats(url)
    if not formats:
        return

    print("\n📋 可用格式列表 (仅显示前 20 个关键格式):")
    print("-" * 80)

    # 分离视频和音频以便展示
    video_formats = [f for f in formats if f["is_video"]]
    audio_formats = [f for f in formats if not f["is_video"]]

    # 打印音频
    if audio_formats:
        print("【🎵 纯音频流】")
        for f in audio_formats[:10]:  # 最多显示10个
            print(f"  {f['display']}")

    # 打印视频
    if video_formats:
        print("\n【🎬 视频流 (含音视频混合或纯视频)】")
        # 按分辨率排序展示几个代表性的
        sorted_vids = sorted(video_formats, key=lambda x: int(x["format_obj"].get("height", 0) or 0), reverse=True)
        for f in sorted_vids[:10]:
            print(f"  {f['display']}")

    print("-" * 80)
    print("💡 输入提示:")
    print("  - 输入具体的 ID (如 140, 22) 下载指定格式")
    print("  - 输入 'a' 自动下载最佳音频")
    print("  - 输入 'v' 自动下载最佳视频 (最高画质)")
    print("  - 输入 'q' 退出")

    choice = input("\n请选择格式 ID: ").strip().lower()

    if choice == "q":
        return

    final_format_id = ""

    if choice == "a":
        # 寻找最佳音频
        # 优先找 m4a, mp3, opus
        best_audio = next((f for f in audio_formats if f["format_obj"]["ext"] in ["m4a", "mp3"]), None)
        if not best_audio:
            best_audio = audio_formats[0] if audio_formats else None

        if best_audio:
            final_format_id = best_audio["id"]
            print(f"✅ 已选择最佳音频: ID {final_format_id}")
        else:
            print("❌ 未找到可用的纯音频流。")
            return

    elif choice == "v":
        # 寻找最佳视频 (通常 yt-dlp 的 'best' 策略很好，这里手动选一个分辨率最高的)
        if video_formats:
            # 找分辨率最高的
            best_vid = sorted(video_formats, key=lambda x: int(x["format_obj"].get("height", 0) or 0), reverse=True)[0]
            final_format_id = best_vid["id"]
            print(f"✅ 已选择最佳视频: ID {final_format_id} ({best_vid['format_obj'].get('resolution')})")
        else:
            print("❌ 未找到视频流。")
            return
    else:
        # 验证用户输入的 ID 是否存在
        valid_ids = [f["id"] for f in formats]
        if choice in valid_ids:
            final_format_id = choice
        else:
            print(f"❌ 无效的格式 ID: {choice}。请在列表中选择一个。")
            return

    # 2. 执行下载
    downloaded_file, info_dict = download_selected(url, final_format_id)

    if not downloaded_file or not os.path.exists(downloaded_file):
        print("❌ 下载未完成或文件不存在。")
        return

    # 3. 判断是否需要提取音频
    file_ext = os.path.splitext(downloaded_file)[1].lower()
    audio_extensions = [".mp3", ".m4a", ".wav", ".opus", ".ogg", ".flac"]

    final_audio_path = ""

    if file_ext in audio_extensions:
        print(f"\n✨ 下载的是音频文件 ({file_ext})，无需提取。")
        # 如果不是 mp3 但 whisper 支持，也可以直接用，但为了统一转成 mp3 吧
        if file_ext != ".mp3":
            print("🔄 正在转换为 MP3 以保持一致性...")
            # 复用提取逻辑转换格式
            temp_mp3 = downloaded_file.rsplit(".", 1)[0] + ".mp3"
            # 这里简单重命名或调用 ffmpeg，为了代码简洁，假设 whisper 也能读 m4a
            # 如果需要强制 mp3，可以调用 extract_audio_from_video 逻辑（虽然它是视频来的名字，但 ffmpeg 不care）
            final_audio_path = downloaded_file
        else:
            final_audio_path = downloaded_file
    else:
        # 是视频，需要提取
        target_mp3_name = downloaded_file.rsplit(".", 1)[0] + ".mp3"
        final_audio_path = extract_audio_from_video(downloaded_file, target_mp3_name)

        if not final_audio_path:
            print("❌ 音频提取失败，无法继续转录。")
            return

    # 4. 转录
    if final_audio_path:
        txt_file, srt_file = transcribe_audio(final_audio_path)

        print("\n" + "=" * 60)
        print("🎉 全部完成！")
        print("=" * 60)
        print(f"📂 原始文件: {downloaded_file}")
        if final_audio_path != downloaded_file:
            print(f"🎵 音频文件: {final_audio_path}")
        print(f"📄 文本稿: {txt_file}")
        print(f"📺 字幕文件: {srt_file}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 用户中断操作。")
    except Exception as e:
        print(f"\n💥 发生未知错误: {e}")
        import traceback

        traceback.print_exc()
