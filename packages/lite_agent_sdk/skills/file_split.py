from typing import Any, Dict, List

from lite_agent_sdk.base import BaseSkill, MessageBatch, SkillContext


def _normalize_text(text: str) -> str:
    return text.replace("\\n", "\n").strip()


def _iter_room_messages(source_data: Dict[str, Any]):
    for room in source_data.get("rooms", []):
        title = str(room.get("title") or "")
        rid = room.get("id", "")
        for msg in room.get("messages", []):
            raw = _normalize_text(str(msg.get("raw_msg") or ""))
            if not raw or raw in ("收到一条语音消息", "。"):
                continue
            dt = msg.get("datetime", "")
            yield title, rid, dt, raw


class FileSplitSkill(BaseSkill):
    name = "FileSplitSkill"

    async def execute(self, ctx: SkillContext) -> None:
        chunks: List[str] = []
        room_titles: List[str] = []
        msg_count = 0
        batch_id = 0
        max_chars = max(1000, ctx.batch_size)

        def flush():
            nonlocal batch_id, chunks, room_titles, msg_count
            if not chunks:
                return
            ctx.batches.append(
                MessageBatch(
                    batch_id=batch_id,
                    text="\n".join(chunks),
                    message_count=msg_count,
                    room_titles=sorted(set(room_titles)),
                )
            )
            batch_id += 1
            chunks = []
            room_titles = []
            msg_count = 0

        for title, rid, dt, raw in _iter_room_messages(ctx.source_data):
            line = f"[{title}|{rid}|{dt}] {raw}"
            if chunks and sum(len(c) for c in chunks) + len(line) > max_chars:
                flush()
            chunks.append(line)
            room_titles.append(title)
            msg_count += 1

        flush()
        ctx.meta["split_batch_count"] = len(ctx.batches)
