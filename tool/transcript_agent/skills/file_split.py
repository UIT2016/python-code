from __future__ import annotations

from typing import List

from transcript_agent.base import BaseSkill, TextBatch, TranscriptContext


class FileSplitSkill(BaseSkill):
    name = "FileSplitSkill"

    async def execute(self, ctx: TranscriptContext) -> None:
        text = ctx.source_text.strip()
        if not text:
            ctx.batches = []
            ctx.meta["split_batch_count"] = 0
            return

        max_chars = max(2000, ctx.batch_size)
        paragraphs = [p.strip() for p in text.replace("\r\n", "\n").split("\n") if p.strip()]
        if not paragraphs:
            paragraphs = [text]

        chunks: List[str] = []
        current: List[str] = []
        current_len = 0

        def flush(batch_id: int) -> None:
            if not current:
                return
            joined = "\n".join(current)
            ctx.batches.append(TextBatch(batch_id=batch_id, text=joined, char_count=len(joined)))

        batch_id = 0
        for para in paragraphs:
            if current and current_len + len(para) + 1 > max_chars:
                flush(batch_id)
                batch_id += 1
                current = []
                current_len = 0
            if len(para) > max_chars:
                if current:
                    flush(batch_id)
                    batch_id += 1
                    current = []
                    current_len = 0
                for i in range(0, len(para), max_chars):
                    chunk = para[i : i + max_chars]
                    ctx.batches.append(TextBatch(batch_id=batch_id, text=chunk, char_count=len(chunk)))
                    batch_id += 1
                continue
            current.append(para)
            current_len += len(para) + 1

        flush(batch_id)
        if not ctx.batches:
            ctx.batches = [TextBatch(batch_id=0, text=text, char_count=len(text))]
        ctx.meta["split_batch_count"] = len(ctx.batches)
