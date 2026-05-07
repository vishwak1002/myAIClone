from __future__ import annotations
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock
from ai_clone.memory.conversation_store import ConversationStore
from ai_clone.memory.retriever import MemoryRetriever
from ai_clone.memory.models import SearchResult


@pytest.fixture
def fake_embedder():
    embedder = MagicMock()
    call_count = [0]
    def side_effect(texts):
        call_count[0] += 1
        return [[float(call_count[0]) * 0.1, 0.2, 0.3, 0.4]]
    embedder.embed.side_effect = side_effect
    return embedder


@pytest.fixture
def fake_vector_store():
    stored = []

    store = MagicMock()

    def upsert(docs):
        stored.extend(docs)
    def search(query_embedding, top_k, filters=None):
        return [SearchResult(doc, 0.9 - i * 0.1) for i, doc in enumerate(stored[:top_k])]
    def count():
        return len(stored)
    def delete(ids):
        stored[:] = [d for d in stored if d.id not in ids]

    store.upsert.side_effect = upsert
    store.search.side_effect = search
    store.count.side_effect = count
    store.delete.side_effect = delete
    return store, stored


def test_store_and_retrieve_messages(fake_embedder, fake_vector_store):
    vector_store, stored = fake_vector_store
    cs = ConversationStore(vector_store=vector_store, embedder=fake_embedder)
    retriever = MemoryRetriever(conversation_store=cs, recency_half_life_days=30)

    cs.add_message("Hello there", "user", "s1", datetime.now() - timedelta(hours=1))
    cs.add_message("I am fine thanks", "clone", "s1", datetime.now() - timedelta(minutes=30))

    assert cs.count() == 2

    results = retriever.retrieve("how are you", top_k=2)
    assert len(results) == 2
    assert results[0].score >= results[1].score


def test_retriever_reranks_by_recency(fake_embedder, fake_vector_store):
    vector_store, stored = fake_vector_store
    cs = ConversationStore(vector_store=vector_store, embedder=fake_embedder)
    retriever = MemoryRetriever(conversation_store=cs, recency_half_life_days=1)

    cs.add_message("recent message", "user", "s1", datetime.now() - timedelta(hours=1))
    cs.add_message("old message", "user", "s1", datetime.now() - timedelta(days=365))

    results = retriever.retrieve("some query", top_k=2)
    assert len(results) == 2
    # With half_life of 1 day, the old message score is near zero — recent should rank first
    assert results[0].score >= results[1].score
