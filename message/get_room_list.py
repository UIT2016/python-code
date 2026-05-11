import json
from typing import Dict, List

import requests

from message.config_loader import load_headers

# ================== 配置（请根据实际情况更新） ==================
API_URL = "https://mx2025.hhhuu.com/5/api/room/list"

# 请求头（从浏览器复制，注意 token 和 cookie 可能过期）
HEADERS = load_headers()


def fetch_room_list() -> List[Dict]:
    """
    获取聊天室列表
    返回每条记录的 id 和 title
    """
    try:
        # 请求体：该接口可能不需要参数，发送空 JSON
        response = requests.post(API_URL, headers=HEADERS, json={}, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("code") != 200:
            print(f"API 错误: {data}")
            return []

        room_list = []
        for room in data.get("list", []):
            room_id = room.get("id")
            title = room.get("title")
            if room_id and title:
                room_list.append({"id": room_id, "title": title})
        return room_list

    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        return []
    except json.JSONDecodeError as e:
        print(f"JSON 解析失败: {e}")
        return []


def save_rooms_to_file(rooms: List[Dict], filename="rooms.json"):
    """保存房间列表到 JSON 文件"""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(rooms, f, ensure_ascii=False, indent=2)
    print(f"已保存 {len(rooms)} 个房间到 {filename}")


def main():
    print("正在获取聊天室列表...")
    rooms = fetch_room_list()
    if not rooms:
        print("未获取到任何房间，请检查 token/cookie 是否有效。")
        return

    print(f"成功获取 {len(rooms)} 个聊天室")
    # 打印前5条示例
    for i, room in enumerate(rooms[:5], 1):
        print(f"{i}. ID: {room['id']}\t名称: {room['title']}")

    save_rooms_to_file(rooms)


if __name__ == "__main__":
    main()
