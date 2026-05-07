from __future__ import annotations
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch
from ai_clone.memory.store.chroma import ChromaVectorStore
from ai_clone.memory.models import MemoryDocument


@pytest.fixture
def mock_chroma(tmp_path):
    with patch("ai_clone.memory.store.chroma.chromadb") as mock_chromadb:
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_chromadb.PersistentClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection
        yield mock_chromadb, mock_client, mock_collection, tmp_path


def _make_doc(id: str = "doc-1") -> MemoryDocument:
    return MemoryDocument(
        id=id,
        text="Hello world",
        embedding=[0.1, 0.2, 0.3],
        timestamp=datetime(2024, 1, 1, 12, 0, 0),
        session_id="session-1",
        speaker="user",
    )


def test_upsert_calls_collection_upsert(mock_chroma):
    _, _, mock_collection, tmp_path = mock_chroma
    store = ChromaVectorStore(path=tmp_path)
    doc = _make_doc()
    store.upsert([doc])
    mock_collection.upsert.assert_called_once_with(
        ids=["doc-1"],
        documents=["Hello world"],
        embeddings=[[0.1, 0.2, 0.3]],
        metadatas=[doc.to_metadata()],
    )


def test_search_returns_search_results(mock_chroma):
    _, _, mock_collection, tmp_path = mock_chroma
    mock_collection.count.return_value = 10
    mock_collection.query.return_value = {
        "ids": [["doc-1"]],
        "documents": [["Hello world"]],
        "embeddings": [[[0.1, 0.2, 0.3]]],
        "metadatas": [[{
            "timestamp": "2024-01-01T12:00:00",
            "session_id": "session-1",
            "speaker": "user",
        }]],
        "distances": [[0.12]],
    }
    store = ChromaVectorStore(path=tmp_path)
    results = store.search(query_embedding=[0.1, 0.2, 0.3], top_k=5)
    assert len(results) == 1
    assert results[0].document.id == "doc-1"
    assert results[0].document.text == "Hello world"
    # score = 1 - distance
    assert abs(results[0].score - (1 - 0.12)) < 1e-6


def test_search_with_filter_passes_where_clause(mock_chroma):
    _, _, mock_collection, tmp_path = mock_chroma
    mock_collection.count.return_value = 10
    mock_collection.query.return_value = {
        "ids": [[]], "documents": [[]], "embeddings": [[]], "metadatas": [[]], "distances": [[]]
    }
    store = ChromaVectorStore(path=tmp_path)
    store.search([0.1], top_k=3, filters={"speaker": "user"})
    call_kwargs = mock_collection.query.call_args.kwargs
    assert call_kwargs["where"] == {"speaker": "user"}


def test_search_clamps_n_results_when_collection_smaller(mock_chroma):
    _, _, mock_collection, tmp_path = mock_chroma
    mock_collection.count.return_value = 2  # only 2 docs in collection
    mock_collection.query.return_value = {
        "ids": [["doc-1", "doc-2"]],
        "documents": [["msg1", "msg2"]],
        "embeddings": [[[0.1], [0.2]]],
        "metadatas": [[
            {"timestamp": "2024-01-01T00:00:00", "session_id": "s1", "speaker": "user"},
            {"timestamp": "2024-01-02T00:00:00", "session_id": "s1", "speaker": "clone"},
        ]],
        "distances": [[0.1, 0.2]],
    }
    store = ChromaVectorStore(path=tmp_path)
    results = store.search(query_embedding=[0.1], top_k=10)  # ask for 10 but only 2 exist
    call_kwargs = mock_collection.query.call_args.kwargs
    assert call_kwargs["n_results"] == 2  # clamped to collection size
    assert len(results) == 2


def test_search_returns_empty_when_collection_empty(mock_chroma):
    _, _, mock_collection, tmp_path = mock_chroma
    mock_collection.count.return_value = 0
    store = ChromaVectorStore(path=tmp_path)
    results = store.search(query_embedding=[0.1], top_k=5)
    assert results == []
    mock_collection.query.assert_not_called()


def test_delete_calls_collection_delete(mock_chroma):
    _, _, mock_collection, tmp_path = mock_chroma
    store = ChromaVectorStore(path=tmp_path)
    store.delete(["doc-1", "doc-2"])
    mock_collection.delete.assert_called_once_with(ids=["doc-1", "doc-2"])


def test_count_returns_collection_count(mock_chroma):
    _, _, mock_collection, tmp_path = mock_chroma
    mock_collection.count.return_value = 42
    store = ChromaVectorStore(path=tmp_path)
    assert store.count() == 42
