"""知识库检索抽象接口，便于后续替换为 FAISS / Chroma / Milvus / pgvector。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class RetrievedChunk:
    """检索到的知识片段。"""

    content: str
    source: str
    score: float = 0.0
    metadata: dict | None = None


class KnowledgeRetriever(ABC):
    """知识库检索器抽象基类。"""

    @abstractmethod
    def refresh(self) -> None:
        """重新加载/索引知识库。"""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        """根据查询返回最相关的片段。"""
