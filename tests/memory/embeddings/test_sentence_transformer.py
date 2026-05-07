from __future__ import annotations
import pytest
from unittest.mock import MagicMock, patch
from ai_clone.memory.embeddings.sentence_transformer import SentenceTransformerEmbedder


@pytest.fixture
def mock_model():
    with patch("ai_clone.memory.embeddings.sentence_transformer.SentenceTransformer") as mock_cls:
        instance = MagicMock()
        instance.encode.return_value = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
        mock_cls.return_value = instance
        yield mock_cls, instance


def test_embed_returns_list_of_vectors(mock_model):
    _, model_instance = mock_model
    embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")
    result = embedder.embed(["hello world", "how are you"])
    assert len(result) == 2
    assert len(result[0]) == 4


def test_embed_calls_model_encode(mock_model):
    _, model_instance = mock_model
    embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")
    embedder.embed(["test"])
    model_instance.encode.assert_called_once()


def test_embed_empty_list_returns_empty(mock_model):
    _, model_instance = mock_model
    model_instance.encode.return_value = []
    embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")
    result = embedder.embed([])
    assert result == []


def test_default_model_name(mock_model):
    mock_cls, _ = mock_model
    SentenceTransformerEmbedder()
    mock_cls.assert_called_once_with("all-MiniLM-L6-v2")
