import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from config_loader import load_config
from flask import Flask, flash, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MESSAGE_DATA_DIR = BASE_DIR / "message_data"

cfg = load_config(BASE_DIR)

API_URL = cfg["msg_api_url"]
HEADERS = cfg["headers"]
# 接口 pagesize 固定为 30，修改无效
API_PAGE_SIZE = 30
MAX_FETCH_PAGES = 500
PAGE_FETCH_DELAY_SEC = 0.3


def _date_cutoff_ms(from_date: str) -> int:
    cutoff = datetime.strptime(from_date.strip(), "%Y-%m-%d").replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return int(cutoff.timestamp() * 1000)


def _parse_createtime(value: Any) -> int:
    """解析 createtime：支持毫秒时间戳（int/数字字符串）或 'YYYY-MM-DD HH:MM:SS' 字符串。"""
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return 0
        if s.isdigit():
            return int(s)
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
            try:
                return int(datetime.strptime(s, fmt).timestamp() * 1000)
            except ValueError:
                continue
    return 0


def _parse_msg_content(msg_raw: Any) -> List[Dict[str, Any]]:
    """解析 msg 字段：接口返回 JSON 字符串，少数情况下可能已是 list。"""
    if msg_raw is None:
        return []
    if isinstance(msg_raw, list):
        return msg_raw
    if isinstance(msg_raw, str):
        if not msg_raw.strip():
            return []
        try:
            parsed = json.loads(msg_raw)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return [parsed]
    return []


def _extract_text(msg_content: List[Dict[str, Any]]) -> str:
    parts = []
    for part in msg_content:
        if isinstance(part, dict) and part.get("type") == "text":
            text = part.get("msg")
            if text:
                parts.append(str(text))
    return "\n".join(parts)


def fetch_messages(rid: int, msgid: int = 0) -> List[Dict]:
    """
    获取指定聊天室的一页消息（pagesize 固定 30）。
    :param rid: 房间ID
    :param msgid: 0 表示最新；翻页时传入上一批最老消息的 id
    :return: 解析后的消息列表
    """
    payload = {
        "rid": rid,
        "msgid": msgid,
        "pagesize": API_PAGE_SIZE,
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
            try:
                msg_content = _parse_msg_content(item.get("msg"))
                full_text = _extract_text(msg_content)
                createtime_ms = _parse_createtime(item.get("createtime"))
                if createtime_ms <= 0:
                    print(f"  消息 {item.get('id')} 的 createtime 无法解析: {item.get('createtime')!r}")
                    continue
                parsed_item = {
                    "id": item["id"],
                    "createtime": createtime_ms,
                    "datetime": datetime.fromtimestamp(createtime_ms / 1000).strftime("%Y-%m-%d %H:%M:%S"),
                    "raw_msg": full_text,
                }
                parsed_list.append(parsed_item)
            except Exception as e:
                print(f"  消息 {item.get('id')} 解析失败: {e}")
                continue
        return parsed_list
    except Exception as e:
        print(f"  请求失败 (rid={rid}): {e}")
        return []


def fetch_messages_paginated(
    rid: int,
    from_date: Optional[str] = None,
    *,
    max_pages: int = MAX_FETCH_PAGES,
) -> Tuple[List[Dict], int]:
    """
    分页拉取聊天室消息。
    首次 msgid=0；之后将上一批最老消息的 id 作为 msgid 继续请求，直到：
      - 返回空列表或不足一页
      - 本批最老消息早于 from_date（若指定）
      - 达到 max_pages
    返回 (消息列表按时间升序, 实际请求页数)。
    """
    cutoff_ms = _date_cutoff_ms(from_date) if from_date else None
    by_id: Dict[int, Dict] = {}
    msgid = 0
    page_count = 0

    while page_count < max_pages:
        batch = fetch_messages(rid=rid, msgid=msgid)
        page_count += 1
        if not batch:
            break

        oldest_id = min(m["id"] for m in batch)
        oldest_time = min(m["createtime"] for m in batch)

        for msg in batch:
            if cutoff_ms is not None and msg["createtime"] < cutoff_ms:
                continue
            by_id[msg["id"]] = msg

        if cutoff_ms is not None and oldest_time < cutoff_ms:
            break
        if len(batch) < API_PAGE_SIZE:
            break
        if oldest_id == msgid:
            break

        msgid = oldest_id
        time.sleep(PAGE_FETCH_DELAY_SEC)

    return sorted(by_id.values(), key=lambda m: m["createtime"]), page_count


def fetch_room_messages(
    rid: int,
    title: str,
    from_date: Optional[str] = None,
) -> Tuple[str, int, Optional[Dict[str, Any]], str]:
    """
    拉取单个聊天室消息，可选按日期截断并自动 msgid 翻页。
    返回 (status, 消息条数, 房间数据或 None, 详情说明)。
    status: ok | skip | error
    """
    if from_date:
        messages, pages = fetch_messages_paginated(rid, from_date=from_date)
    else:
        messages = fetch_messages(rid=rid, msgid=0)
        pages = 1

    if not messages:
        if from_date:
            return "skip", 0, None, f"在选定日期 {from_date} 之后无消息（已请求 {pages} 页）"
        return "skip", 0, None, "无消息"

    room_data = {
        "id": rid,
        "title": title,
        "message_count": len(messages),
        "messages": messages,
    }
    detail = f"获取 {len(messages)} 条消息"
    if pages > 1:
        detail += f"（{pages} 页）"
    return "ok", len(messages), room_data, detail


def save_messages_by_date(
    date_str: str,
    rooms_data: List[Dict[str, Any]],
    from_date: Optional[str] = None,
) -> Path:
    """将各聊天室消息汇总保存为以日期命名的 JSON 文件。"""
    MESSAGE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    total_messages = sum(r.get("message_count", 0) for r in rooms_data)
    payload = {
        "date": date_str,
        "from_date": from_date or date_str,
        "pagesize": API_PAGE_SIZE,
        "collected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "room_count": len(rooms_data),
        "total_messages": total_messages,
        "rooms": rooms_data,
    }
    out_file = MESSAGE_DATA_DIR / f"{date_str}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_file


def collect_rooms(
    rooms: List[Dict[str, Any]],
    date_str: str,
    from_date: Optional[str] = None,
) -> Tuple[Optional[Path], List[Dict[str, Any]]]:
    """批量拉取聊天室消息并保存到日期文件，返回 (输出路径, 各房间处理结果)。"""
    rooms_data: List[Dict[str, Any]] = []
    results: List[Dict[str, Any]] = []
    for room in rooms:
        rid, title = room["id"], room["title"]
        status, n, room_data, detail = fetch_room_messages(rid, title, from_date=from_date)
        label = {"ok": "成功", "skip": "跳过", "error": "失败"}.get(status, status)
        results.append({"id": rid, "title": title, "status": label, "msg_count": n, "detail": detail})
        if room_data:
            rooms_data.append(room_data)
        time.sleep(1)
    if not rooms_data:
        return None, results
    out_file = save_messages_by_date(date_str, rooms_data, from_date=from_date)
    for r in results:
        if r["status"] == "成功":
            r["detail"] = out_file.name
    return out_file, results


def main():
    rooms_file = BASE_DIR / "rooms.json"
    if not rooms_file.exists():
        print(f"错误：未找到 {rooms_file}，请先运行 get_room_list.py 获取房间列表。")
        return

    with open(rooms_file, "r", encoding="utf-8") as f:
        rooms = json.load(f)
    print(f"加载了 {len(rooms)} 个聊天室")

    today = datetime.now().strftime("%Y-%m-%d")
    print(f"\n开始拉取消息（msgid 翻页，pagesize={API_PAGE_SIZE}），汇总保存为 {today}.json")

    out_file, results = collect_rooms(rooms, date_str=today, from_date=today)
    for idx, (room, result) in enumerate(zip(rooms, results), 1):
        print(f"\n[{idx}/{len(rooms)}] {room['title']} (ID: {room['id']})")
        print(f"  状态: {result['status']}，消息: {result['msg_count']} 条 ({result['detail']})")

    if out_file:
        print(f"\n全部完成！汇总文件: {out_file}")
    else:
        print("\n未获取到任何有效消息，未生成汇总文件。")


def create_app() -> Flask:
    app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))
    app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-message-fetch-change-in-prod")

    @app.route("/", methods=["GET", "POST"])
    def index():
        today = datetime.now().strftime("%Y-%m-%d")

        rooms_file = BASE_DIR / "rooms.json"
        if not rooms_file.exists():
            flash("未找到 rooms.json，请先运行 get_room_list.py。", "error")
            return render_template("index.html", rooms=[], results=None, default_date=today)

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
                )
            if not from_date:
                flash("请选择日期。", "error")
                return render_template(
                    "index.html",
                    rooms=rooms,
                    results=None,
                    default_date=today,
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
                )

            out_file, results = collect_rooms(selected, date_str=from_date, from_date=from_date)
            if out_file:
                flash(
                    f"已处理 {len(results)} 个聊天室，汇总保存至 {out_file.name}（日期截断：{from_date} 起，pagesize={API_PAGE_SIZE} 自动 msgid 翻页）。",
                    "message",
                )
            else:
                flash("未获取到任何有效消息，未生成汇总文件。", "error")

        form_date = (request.form.get("from_date") or "").strip() if request.method == "POST" else ""
        default_date = form_date or today
        return render_template(
            "index.html",
            rooms=rooms,
            results=results,
            default_date=default_date,
        )

    return app


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] in ("--web", "-w", "web"):
        create_app().run(host="127.0.0.1", port=5000, debug=True)
    else:
        main()
