from __future__ import annotations
import math
from datetime import datetime, timezone
from typing import List
from .conversation_store import ConversationStore
from .models import SearchResult

_DEFAULT_HALF_LIFE_DAYS = 30


class MemoryRetriever:
    """Returns semantically similar memories, re-ranked by recency.

    Score formula: final_score = similarity_score * recency_weight
    where recency_weight = 2^(-age_days / half_life_days)

    This means memories lose half their recency weight every `recency_half_life_days`.
    A memory from 30 days ago (default) has 0.5x the recency weight of a fresh memory.
    """

    def __init__(
        self,
        conversation_store: ConversationStore,
        recency_half_life_days: float = _DEFAULT_HALF_LIFE_DAYS,
    ) -> None:
        if recency_half_life_days <= 0:
            raise ValueError("recency_half_life_days must be > 0")
        self._store = conversation_store
        self._half_life_days = recency_half_life_days

    def retrieve(self, query: str, top_k: int) -> List[SearchResult]:
        """Return top-K memories re-ranked by similarity × recency weight."""
        results = self._store.search(query, top_k=top_k)
        now = datetime.now(tz=timezone.utc)
        reranked = []
        for result in results:
            ts = result.document.timestamp
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            age_days = (now - ts).total_seconds() / 86400
            recency_weight = math.pow(2.0, -age_days / self._half_life_days)
            reranked.append(
                SearchResult(
                    document=result.document,
                    score=result.score * recency_weight,
                )
            )
        reranked.sort(key=lambda r: r.score, reverse=True)
        return reranked
