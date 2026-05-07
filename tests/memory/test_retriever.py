from __future__ import annotations
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock
from ai_clone.memory.retriever import MemoryRetriever
from ai_clone.memory.models import MemoryDocument, SearchResult


def _make_result(id: str, score: float, days_ago: int) -> SearchResult:
    doc = MemoryDocument(
        id=id,
        text=f"message {id}",
        embedding=[],
        timestamp=datetime.now() - timedelta(days=days_ago),
        session_id="s1",
        speaker="user",
    )
    return SearchResult(document=doc, score=score)


@pytest.fixture
def fake_store():
    store = MagicMock()
    return store


def test_retrieve_returns_top_k_results(fake_store):
    fake_store.search.return_value = [
        _make_result("a", 0.9, 1),
        _make_result("b", 0.8, 2),
        _make_result("c", 0.7, 3),
    ]
    retriever = MemoryRetriever(conversation_store=fake_store)
    results = retriever.retrieve("what time is it", top_k=3)
    assert len(results) == 3


def test_retrieve_applies_recency_weighting(fake_store):
    """A recent low-score result can outrank an old high-score result."""
    old_high = _make_result("old", score=0.95, days_ago=365)
    new_low = _make_result("new", score=0.6, days_ago=1)
    fake_store.search.return_value = [old_high, new_low]
    retriever = MemoryRetriever(conversation_store=fake_store, recency_half_life_days=30)
    results = retriever.retrieve("query", top_k=2)
    # new_low should outscore old_high after recency weighting
    assert results[0].document.id == "new"


def test_retrieve_results_sorted_by_final_score(fake_store):
    results_raw = [
        _make_result("a", 0.9, 100),
        _make_result("b", 0.5, 1),
        _make_result("c", 0.7, 10),
    ]
    fake_store.search.return_value = results_raw
    retriever = MemoryRetriever(conversation_store=fake_store, recency_half_life_days=30)
    results = retriever.retrieve("q", top_k=3)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_delegates_to_conversation_store(fake_store):
    fake_store.search.return_value = []
    retriever = MemoryRetriever(conversation_store=fake_store)
    retriever.retrieve("hello", top_k=5)
    fake_store.search.assert_called_once_with("hello", top_k=5)
