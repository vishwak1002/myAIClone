from __future__ import annotations
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock
from ai_clone.memory.consolidator import MemoryConsolidator
from ai_clone.memory.models import MemoryDocument


def _make_old_doc(id: str, days_ago: int = 100) -> MemoryDocument:
    return MemoryDocument(
        id=id,
        text=f"old message {id}",
        embedding=[0.1, 0.2],
        timestamp=datetime.now(tz=timezone.utc) - timedelta(days=days_ago),
        session_id="s1",
        speaker="user",
    )


@pytest.fixture
def fake_store():
    store = MagicMock()
    store.count.return_value = 200
    return store


@pytest.fixture
def fake_openrouter():
    client = MagicMock()
    response = MagicMock()
    response.choices[0].message.content = "Summary of old conversations."
    client.chat.completions.create.return_value = response
    return client


def test_consolidate_skips_when_count_below_threshold(fake_store, fake_openrouter):
    fake_store.count.return_value = 50
    consolidator = MemoryConsolidator(
        conversation_store=fake_store,
        openrouter_client=fake_openrouter,
        threshold=100,
    )
    deleted = consolidator.consolidate(older_than_days=60)
    assert deleted == 0
    fake_openrouter.chat.completions.create.assert_not_called()
    fake_store.get_older_than.assert_not_called()


def test_consolidate_fetches_old_messages_and_summarises(fake_store, fake_openrouter):
    old_docs = [
        MemoryDocument(
            id=f"doc-{i}",
            text=f"old message {i}",
            embedding=[0.1, 0.2],
            timestamp=datetime.now(tz=timezone.utc) - timedelta(days=90),
            session_id="s1",
            speaker="user",
        )
        for i in range(5)
    ]
    fake_store.get_older_than.return_value = old_docs
    fake_store.count.return_value = 200

    consolidator = MemoryConsolidator(
        conversation_store=fake_store,
        openrouter_client=fake_openrouter,
        threshold=100,
    )
    deleted = consolidator.consolidate(older_than_days=60)

    fake_openrouter.chat.completions.create.assert_called_once()
    fake_store.add_message.assert_called_once()
    fake_store.delete.assert_called_once()
    deleted_ids = fake_store.delete.call_args[0][0]
    assert set(deleted_ids) == {f"doc-{i}" for i in range(5)}
    assert deleted == 5


def test_consolidate_returns_zero_when_no_old_messages(fake_store, fake_openrouter):
    fake_store.get_older_than.return_value = []
    consolidator = MemoryConsolidator(
        conversation_store=fake_store,
        openrouter_client=fake_openrouter,
        threshold=100,
    )
    deleted = consolidator.consolidate(older_than_days=60)
    assert deleted == 0
    fake_openrouter.chat.completions.create.assert_not_called()
