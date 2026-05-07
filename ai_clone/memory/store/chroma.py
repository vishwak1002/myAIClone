from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional
import chromadb
from .base import VectorStore
from ..models import MemoryDocument, SearchResult

_COLLECTION_NAME = "conversations"


class ChromaVectorStore(VectorStore):
    """VectorStore backed by ChromaDB (persistent, local).

    To swap to a different backend, implement VectorStore and pass the new
    implementation to ConversationStore — no other code needs to change.
    """

    def __init__(self, path: Path = Path("./data/chroma")) -> None:
        self._client = chromadb.PersistentClient(path=str(path))
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, documents: List[MemoryDocument]) -> None:
        if not documents:
            return
        self._collection.upsert(
            ids=[d.id for d in documents],
            documents=[d.text for d in documents],
            embeddings=[d.embedding for d in documents],
            metadatas=[d.to_metadata() for d in documents],
        )

    def search(
        self,
        query_embedding: List[float],
        top_k: int,
        filters: Optional[Dict] = None,
    ) -> List[SearchResult]:
        actual_n = min(top_k, self._collection.count())
        if actual_n == 0:
            return []

        kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": actual_n,
            "include": ["documents", "embeddings", "metadatas", "distances"],
        }
        if filters:
            kwargs["where"] = filters

        raw = self._collection.query(**kwargs)
        results = []
        ids = raw["ids"][0]
        texts = raw["documents"][0]
        embeddings = raw["embeddings"][0]
        metadatas = raw["metadatas"][0]
        distances = raw["distances"][0]

        for doc_id, text, emb, meta, dist in zip(ids, texts, embeddings, metadatas, distances):
            doc = MemoryDocument.from_metadata(
                id=doc_id, text=text, embedding=list(emb), metadata=meta
            )
            # ChromaDB returns cosine distance (0=identical); convert to similarity score
            results.append(SearchResult(document=doc, score=1.0 - dist))
        return results

    def delete(self, ids: List[str]) -> None:
        self._collection.delete(ids=ids)

    def count(self) -> int:
        return self._collection.count()
