import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from config_loader import load_config
from flask import Flask, flash, render_template, request
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

BASE_DIR = Path(__file__).resolve().parent

cfg = load_config(BASE_DIR)

API_URL = cfg["msg_api_url"]
HEADERS = cfg["headers"]
DEEPSEEK_API_KEY = cfg["api_key"]
DEEPSEEK_API_BASE = cfg["api_url"]
MODEL = cfg["model"]
PAGE_SIZE = 30


def fetch_messages(rid: int, msgid: int = 0, pagesize: int = PAGE_SIZE) -> List[Dict]:
    """
    获取指定聊天室的消息列表
    :param rid: 房间ID
    :param msgid: 起始消息ID，0表示最新
    :param pagesize: 每页条数
    :return: 解析后的消息列表
    """
    payload = {
        "rid": rid,
        "msgid": msgid,
        "pagesize": pagesize,
        "tt": int(time.time() * 1000),
    }
    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("code") != 200:
            print(f"  API 错误 (rid={rid}): {data}")
            return []
        parsed_list = []
        for item in data.get("list", []):
            msg_json_str = item.get("msg", "[]")
            try:
                msg_content = json.loads(msg_json_str)
                full_text = "\n".join([part["msg"] for part in msg_content if part.get("type") == "text"])
                parsed_item = {
                    "id": item["id"],
                    "createtime": item["createtime"],
                    "datetime": datetime.fromtimestamp(item["createtime"] / 1000).strftime("%Y-%m-%d %H:%M:%S"),
                    "raw_msg": full_text,
                }
                parsed_list.append(parsed_item)
            except json.JSONDecodeError:
                print(f"  消息 {item['id']} 的 msg 字段解析失败")
                continue
        return parsed_list
    except Exception as e:
        print(f"  请求失败 (rid={rid}): {e}")
        return []


def filter_messages_from_date(messages: List[Dict], from_date: str) -> List[Dict]:
    """仅保留 createtime >= 所选日期本地 00:00:00 的消息，并按时间升序返回。"""
    cutoff = datetime.strptime(from_date.strip(), "%Y-%m-%d").replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    cutoff_ms = int(cutoff.timestamp() * 1000)
    kept = [m for m in messages if m["createtime"] >= cutoff_ms]
    return sorted(kept, key=lambda x: x["createtime"])


def build_analysis_prompt(messages: List[Dict], room_title: str = "") -> str:
    """根据消息生成 prompt（取最新30条）"""
    if not messages:
        return "无消息内容"
    sorted_msgs = sorted(messages, key=lambda x: x["createtime"])
    latest_msgs = sorted_msgs[-30:] if len(sorted_msgs) > 30 else sorted_msgs

    text_blocks = []
    for msg in latest_msgs:
        text_blocks.append(f"[{msg['datetime']}]\n{msg['raw_msg']}\n")
    combined = "\n---\n".join(text_blocks)

    room_info = f"（聊天室：{room_title}）\n" if room_title else ""
    return f"""
你是一位专业的股票投资分析助手。{room_info}以下是一组投资聊天室的历史消息，请综合分析：

1. 整体市场情绪（乐观/谨慎/恐慌）
2. 主要关注板块和个股
3. 关键操作策略建议
4. 潜在风险和机会
5. 对发言者核心观点的总结

消息内容：
{combined}

请给出清晰、实用的分析结论。
"""


def analyze_with_deepseek(messages: List[Dict], api_key: str, room_title: str = "") -> str:
    """调用 DeepSeek API 进行分析"""
    if not messages:
        return "该房间无有效消息，跳过分析。"
    try:
        llm = ChatOpenAI(
            model=MODEL,
            openai_api_key=api_key,
            openai_api_base=DEEPSEEK_API_BASE,
            temperature=0.7,
        )
        prompt = build_analysis_prompt(messages, room_title)
        response = llm.invoke([SystemMessage(content="你是一位专业的股票投资分析助手。"), HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        return f"AI 分析失败: {e}"


def _safe_title(title: str) -> str:
    return "".join(c for c in title if c.isalnum() or c in (" ", "-", "_")).rstrip()


def process_one_room(
    rid: int,
    title: str,
    api_key: str,
    output_dir: Path,
    from_date: Optional[str] = None,
) -> Tuple[str, int, str]:
    """
    拉取消息 → 可选按日期截断 → 分析与落盘。
    返回 (status, 用于分析的条数, 详情说明)。
    status: ok | skip | error
    """
    messages = fetch_messages(rid=rid, msgid=0, pagesize=PAGE_SIZE)
    if not messages:
        return "skip", 0, "无消息"
    if from_date:
        messages = filter_messages_from_date(messages, from_date)
    if not messages:
        return "skip", 0, f"在选定日期 {from_date} 之后无消息（当前仅拉取最近 {PAGE_SIZE} 条）"
    analysis = analyze_with_deepseek(messages, api_key, room_title=title)
    safe = _safe_title(title) or str(rid)
    msg_file = output_dir / f"{rid}_{safe}_messages.json"
    ana_file = output_dir / f"{rid}_{safe}_analysis.txt"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(msg_file, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)
    with open(ana_file, "w", encoding="utf-8") as f:
        f.write(analysis)
    n = len(messages)
    return "ok", n, f"{msg_file.name} / {ana_file.name}"


def main():
    rooms_file = BASE_DIR / "rooms.json"
    if not rooms_file.exists():
        print(f"错误：未找到 {rooms_file}，请先运行 get_room_list.py 获取房间列表。")
        return

    with open(rooms_file, "r", encoding="utf-8") as f:
        rooms = json.load(f)
    print(f"加载了 {len(rooms)} 个聊天室")

    api_key = DEEPSEEK_API_KEY
    output_dir = BASE_DIR / "analysis_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, room in enumerate(rooms, 1):
        rid = room["id"]
        title = room["title"]
        print(f"\n[{idx}/{len(rooms)}] 正在处理房间: {title} (ID: {rid})")

        status, n, detail = process_one_room(rid, title, api_key, output_dir, from_date=None)
        print(f"  获取并用于分析的消息: {n} 条 ({detail})")

        if status == "skip":
            print("  跳过")
            continue

        print("  已完成分析与保存")
        time.sleep(1)

    print("\n全部房间处理完成！")


def create_app() -> Flask:
    app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))
    app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-message-analysis-change-in-prod")

    @app.route("/", methods=["GET", "POST"])
    def index():
        today = datetime.now().strftime("%Y-%m-%d")
        rooms_file = BASE_DIR / "rooms.json"
        if not rooms_file.exists():
            flash("未找到 rooms.json，请先运行 get_room_list.py。", "error")
            return render_template("index.html", rooms=[], results=None, default_date=today, page_size=PAGE_SIZE)

        with open(rooms_file, "r", encoding="utf-8") as f:
            rooms: List[Dict[str, Any]] = json.load(f)

        results: Optional[List[Dict[str, Any]]] = None

        if request.method == "POST":
            raw_ids = request.form.getlist("room_ids")
            from_date = (request.form.get("from_date") or "").strip()
            if not raw_ids:
                flash("请至少选择一个聊天室。", "error")
                return render_template(
                    "index.html",
                    rooms=rooms,
                    results=None,
                    default_date=from_date or today,
                    page_size=PAGE_SIZE,
                )
            if not from_date:
                flash("请选择日期。", "error")
                return render_template(
                    "index.html",
                    rooms=rooms,
                    results=None,
                    default_date=today,
                    page_size=PAGE_SIZE,
                )
            try:
                datetime.strptime(from_date, "%Y-%m-%d")
            except ValueError:
                flash("日期格式无效，请使用 YYYY-MM-DD。", "error")
                return render_template(
                    "index.html",
                    rooms=rooms,
                    results=None,
                    default_date=from_date,
                    page_size=PAGE_SIZE,
                )

            id_set = {r["id"] for r in rooms}
            selected: List[Dict[str, Any]] = []
            for s in raw_ids:
                try:
                    rid = int(s)
                except ValueError:
                    continue
                if rid not in id_set:
                    continue
                selected.append(next(r for r in rooms if r["id"] == rid))

            if not selected:
                flash("所选房间无效。", "error")
                return render_template(
                    "index.html",
                    rooms=rooms,
                    results=None,
                    default_date=from_date,
                    page_size=PAGE_SIZE,
                )

            api_key = DEEPSEEK_API_KEY
            output_dir = BASE_DIR / "analysis_results"
            results = []
            for room in selected:
                rid, title = room["id"], room["title"]
                status, n, detail = process_one_room(rid, title, api_key, output_dir, from_date=from_date)
                label = {"ok": "成功", "skip": "跳过", "error": "失败"}.get(status, status)
                results.append({"id": rid, "title": title, "status": label, "msg_count": n, "detail": detail})
                time.sleep(1)

            flash(f"已处理 {len(results)} 个聊天室（日期截断：{from_date} 起）。", "message")

        form_date = (request.form.get("from_date") or "").strip() if request.method == "POST" else ""
        default_date = form_date or today
        return render_template(
            "index.html",
            rooms=rooms,
            results=results,
            default_date=default_date,
            page_size=PAGE_SIZE,
        )

    return app


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] in ("--web", "-w", "web"):
        create_app().run(host="127.0.0.1", port=5000, debug=True)
    else:
        main()
