from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from ..models import MemoryDocument, SearchResult


class VectorStore(ABC):
    """Provider-agnostic interface for a vector store.

    Implement this to add a new backend (Pinecone, Weaviate, pgvector, etc.).
    ChromaVectorStore is the default implementation.
    """

    @abstractmethod
    def upsert(self, documents: List[MemoryDocument]) -> None:
        """Insert or update documents in the store."""
        ...

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int,
        filters: Optional[Dict] = None,
    ) -> List[SearchResult]:
        """Return top-K most similar documents, optionally filtered by metadata."""
        ...

    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """Delete documents by ID."""
        ...

    @abstractmethod
    def count(self) -> int:
        """Return total number of stored documents."""
        ...
