from __future__ import annotations
from datetime import datetime
from ai_clone.memory.models import MemoryDocument, SearchResult


def test_memory_document_stores_fields():
    doc = MemoryDocument(
        id="doc-1",
        text="Hello, how are you?",
        embedding=[0.1, 0.2, 0.3],
        timestamp=datetime(2024, 1, 1, 12, 0, 0),
        session_id="session-abc",
        speaker="user",
    )
    assert doc.id == "doc-1"
    assert doc.text == "Hello, how are you?"
    assert doc.embedding == [0.1, 0.2, 0.3]
    assert doc.session_id == "session-abc"
    assert doc.speaker == "user"


def test_search_result_stores_fields():
    doc = MemoryDocument(
        id="doc-2",
        text="I am fine.",
        embedding=[0.4, 0.5],
        timestamp=datetime(2024, 1, 2, 10, 0, 0),
        session_id="session-xyz",
        speaker="clone",
    )
    result = SearchResult(document=doc, score=0.87)
    assert result.document is doc
    assert result.score == 0.87


def test_memory_document_metadata_roundtrip():
    ts = datetime(2024, 6, 15, 9, 30, 0)
    doc = MemoryDocument(
        id="doc-3",
        text="test",
        embedding=[],
        timestamp=ts,
        session_id="s1",
        speaker="user",
    )
    meta = doc.to_metadata()
    assert meta["timestamp"] == ts.isoformat()
    assert meta["session_id"] == "s1"
    assert meta["speaker"] == "user"

    restored = MemoryDocument.from_metadata(id="doc-3", text="test", embedding=[], metadata=meta)
    assert restored.timestamp == ts
    assert restored.session_id == "s1"
    assert restored.speaker == "user"
