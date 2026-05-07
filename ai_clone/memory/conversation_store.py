from __future__ import annotations
import uuid
from datetime import datetime
from typing import List, Optional
from .embeddings.base import EmbeddingProvider
from .models import MemoryDocument, SearchResult
from .store.base import VectorStore


class ConversationStore:
    """Stores and retrieves conversation messages using a pluggable VectorStore + EmbeddingProvider.

    Usage:
        store = ConversationStore(
            vector_store=ChromaVectorStore(path=Path("./data/chroma")),
            embedder=SentenceTransformerEmbedder(),
        )
        store.add_message(text="Hello", speaker="user", session_id="s1", timestamp=datetime.utcnow())
        results = store.search("how are you", top_k=5)
    """

    def __init__(self, vector_store: VectorStore, embedder: EmbeddingProvider) -> None:
        self._store = vector_store
        self._embedder = embedder

    def add_message(
        self,
        text: str,
        speaker: str,
        session_id: str,
        timestamp: datetime,
    ) -> MemoryDocument:
        """Embed and persist a single message. Returns the stored document."""
        embedding = self._embedder.embed([text])[0]
        doc = MemoryDocument(
            id=str(uuid.uuid4()),
            text=text,
            embedding=embedding,
            timestamp=timestamp,
            session_id=session_id,
            speaker=speaker,
        )
        self._store.upsert([doc])
        return doc

    def search(
        self,
        query: str,
        top_k: int,
        session_id: Optional[str] = None,
    ) -> List[SearchResult]:
        """Embed the query and return top-K semantically similar stored messages."""
        embedding = self._embedder.embed([query])[0]
        filters = {"session_id": session_id} if session_id else None
        return self._store.search(
            query_embedding=embedding, top_k=top_k, filters=filters
        )

    def count(self) -> int:
        return self._store.count()

    def delete(self, ids: List[str]) -> None:
        """Delete documents by ID, delegating to the underlying VectorStore."""
        self._store.delete(ids)
