from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List


class EmbeddingProvider(ABC):
    """Provider-agnostic interface for producing text embeddings.

    Implement this to add a new embedding backend (OpenAI, Cohere, local models, etc.).
    SentenceTransformerEmbedder is the default implementation.
    """

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts. Returns one embedding vector per input text."""
        ...
