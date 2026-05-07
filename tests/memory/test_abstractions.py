from __future__ import annotations
import pytest
from ai_clone.memory.store.base import VectorStore
from ai_clone.memory.embeddings.base import EmbeddingProvider


def test_vector_store_is_abstract():
    with pytest.raises(TypeError):
        VectorStore()  # type: ignore


def test_embedding_provider_is_abstract():
    with pytest.raises(TypeError):
        EmbeddingProvider()  # type: ignore


def test_vector_store_concrete_subclass_requires_all_methods():
    class PartialStore(VectorStore):
        def upsert(self, documents):
            pass
        # Missing: search, delete, count

    with pytest.raises(TypeError):
        PartialStore()


def test_embedding_provider_concrete_subclass_works():
    class FakeEmbedder(EmbeddingProvider):
        def embed(self, texts):
            return [[0.1] * 4 for _ in texts]

    embedder = FakeEmbedder()
    result = embedder.embed(["hello"])
    assert len(result) == 1
    assert len(result[0]) == 4


def test_vector_store_concrete_subclass_works():
    class FakeStore(VectorStore):
        def upsert(self, documents):
            pass
        def search(self, query_embedding, top_k, filters=None):
            return []
        def delete(self, ids):
            pass
        def count(self):
            return 0
        def get_older_than(self, cutoff, limit=100):
            return []

    store = FakeStore()
    assert store.count() == 0
    assert store.search([0.1], top_k=5) == []
