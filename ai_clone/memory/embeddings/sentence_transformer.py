from __future__ import annotations
from typing import List
from sentence_transformers import SentenceTransformer
from .base import EmbeddingProvider

_DEFAULT_MODEL = "all-MiniLM-L6-v2"


class SentenceTransformerEmbedder(EmbeddingProvider):
    """EmbeddingProvider backed by a local sentence-transformers model."""

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vectors = self._model.encode(texts, convert_to_numpy=True)
        return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vectors]
