from __future__ import annotations
import pytest
from datetime import datetime
from unittest.mock import MagicMock
from ai_clone.memory.conversation_store import ConversationStore
from ai_clone.memory.models import MemoryDocument, SearchResult


@pytest.fixture
def fake_store():
    store = MagicMock()
    store.count.return_value = 0
    return store


@pytest.fixture
def fake_embedder():
    embedder = MagicMock()
    embedder.embed.return_value = [[0.1, 0.2, 0.3, 0.4]]
    return embedder


def test_add_message_upserts_to_store(fake_store, fake_embedder):
    cs = ConversationStore(vector_store=fake_store, embedder=fake_embedder)
    cs.add_message(
        text="Hello there",
        speaker="user",
        session_id="s1",
        timestamp=datetime(2024, 3, 1, 10, 0, 0),
    )
    fake_embedder.embed.assert_called_once_with(["Hello there"])
    fake_store.upsert.assert_called_once()
    upserted = fake_store.upsert.call_args[0][0]
    assert len(upserted) == 1
    assert upserted[0].text == "Hello there"
    assert upserted[0].speaker == "user"
    assert upserted[0].session_id == "s1"
    assert upserted[0].embedding == [0.1, 0.2, 0.3, 0.4]


def test_add_message_generates_unique_ids(fake_store, fake_embedder):
    cs = ConversationStore(vector_store=fake_store, embedder=fake_embedder)
    cs.add_message("msg 1", "user", "s1", datetime(2024, 1, 1))
    cs.add_message("msg 2", "clone", "s1", datetime(2024, 1, 2))
    ids = [call[0][0][0].id for call in fake_store.upsert.call_args_list]
    assert ids[0] != ids[1]


def test_search_embeds_query_and_calls_store_search(fake_store, fake_embedder):
    doc = MemoryDocument(
        id="d1", text="hi", embedding=[0.1], timestamp=datetime(2024, 1, 1), session_id="s1", speaker="user"
    )
    fake_store.search.return_value = [SearchResult(document=doc, score=0.9)]
    cs = ConversationStore(vector_store=fake_store, embedder=fake_embedder)
    results = cs.search("hello", top_k=3)
    fake_embedder.embed.assert_called_once_with(["hello"])
    fake_store.search.assert_called_once_with(
        query_embedding=[0.1, 0.2, 0.3, 0.4], top_k=3, filters=None
    )
    assert len(results) == 1
    assert results[0].document.id == "d1"


def test_search_with_session_filter(fake_store, fake_embedder):
    fake_store.search.return_value = []
    cs = ConversationStore(vector_store=fake_store, embedder=fake_embedder)
    cs.search("query", top_k=5, session_id="s1")
    call_kwargs = fake_store.search.call_args.kwargs
    assert call_kwargs["filters"] == {"session_id": "s1"}


def test_count_delegates_to_store(fake_store, fake_embedder):
    fake_store.count.return_value = 17
    cs = ConversationStore(vector_store=fake_store, embedder=fake_embedder)
    assert cs.count() == 17


def test_delete_delegates_to_store(fake_store, fake_embedder):
    cs = ConversationStore(vector_store=fake_store, embedder=fake_embedder)
    cs.delete(["id-1", "id-2"])
    fake_store.delete.assert_called_once_with(["id-1", "id-2"])
