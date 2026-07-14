"""基于关键词重叠的本地目录检索器（简单稳定，易于替换）。"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

from cursor_chat.knowledge.base import KnowledgeRetriever, RetrievedChunk
from cursor_chat.knowledge.loader import iter_knowledge_files, load_file_content

_TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]+|[a-zA-Z0-9_]+")


@dataclass
class _IndexedChunk:
    content: str
    source: str
    tokens: set[str]
    length: int


def _tokenize(text: str) -> set[str]:
    """中英文混合分词：英文按词，中文按字与双字片段。"""
    tokens: set[str] = set()
    for match in _TOKEN_PATTERN.finditer(text.lower()):
        word = match.group(0)
        tokens.add(word)
        if re.fullmatch(r"[\u4e00-\u9fff]+", word) and len(word) >= 2:
            tokens.update(word[i : i + 2] for i in range(len(word) - 1))
    return tokens


def _split_chunks(text: str, *, max_chunk_chars: int) -> List[str]:
    """按段落切分，超长段落再按字符窗口切分。"""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    for para in paragraphs:
        if len(para) <= max_chunk_chars:
            chunks.append(para)
            continue
        start = 0
        while start < len(para):
            chunks.append(para[start : start + max_chunk_chars])
            start += max_chunk_chars
    return chunks


def _score_overlap(query_tokens: set[str], chunk_tokens: set[str], chunk_len: int) -> float:
    if not query_tokens or not chunk_tokens:
        return 0.0
    overlap = len(query_tokens & chunk_tokens)
    if overlap == 0:
        return 0.0
    # 简单 TF 风格 + 长度归一，避免长文档天然占优
    return overlap / (math.log10(chunk_len + 10) + 1.0)


class LocalDirectoryRetriever(KnowledgeRetriever):
    """扫描本地目录，用关键词重叠检索相关片段。"""

    def __init__(
        self,
        directories: Sequence[str | Path],
        *,
        max_chunk_chars: int = 1500,
    ):
        self._directories = [str(Path(d)) for d in directories]
        self._max_chunk_chars = max(200, max_chunk_chars)
        self._chunks: List[_IndexedChunk] = []
        self.refresh()

    def refresh(self) -> None:
        indexed: list[_IndexedChunk] = []
        for path in iter_knowledge_files(self._directories):
            source, content = load_file_content(path)
            if not content:
                continue
            for chunk_text in _split_chunks(content, max_chunk_chars=self._max_chunk_chars):
                indexed.append(
                    _IndexedChunk(
                        content=chunk_text,
                        source=source,
                        tokens=_tokenize(chunk_text),
                        length=len(chunk_text),
                    )
                )
        self._chunks = indexed

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        if not query.strip() or not self._chunks:
            return []
        query_tokens = _tokenize(query)
        scored: list[tuple[float, _IndexedChunk]] = []
        for chunk in self._chunks:
            score = _score_overlap(query_tokens, chunk.tokens, chunk.length)
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            RetrievedChunk(content=chunk.content, source=chunk.source, score=score)
            for score, chunk in scored[: max(1, top_k)]
        ]

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    @property
    def directories(self) -> Iterable[str]:
        return tuple(self._directories)
