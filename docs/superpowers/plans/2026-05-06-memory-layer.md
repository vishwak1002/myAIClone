# Memory Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a provider-agnostic semantic memory layer — abstract interfaces for vector store and embedder, ChromaDB + sentence-transformers as the default implementations, with ConversationStore, MemoryRetriever, and MemoryConsolidator on top.

**Architecture:** `VectorStore` and `EmbeddingProvider` ABCs decouple storage and embedding from any specific provider. ChromaDB and SentenceTransformer are pluggable defaults. `ConversationStore` orchestrates embed + upsert. `MemoryRetriever` does similarity search with time-weighted recency scoring. `MemoryConsolidator` summarizes old clusters to keep the store lean.

**Tech Stack:** Python 3.11+, chromadb, sentence-transformers, pytest, pytest-mock

---

## File Structure

```
ai_clone/memory/
├── __init__.py
├── models.py                        # MemoryDocument, SearchResult dataclasses
├── store/
│   ├── __init__.py
│   ├── base.py                      # VectorStore ABC
│   └── chroma.py                    # ChromaVectorStore(VectorStore)
├── embeddings/
│   ├── __init__.py
│   ├── base.py                      # EmbeddingProvider ABC
│   └── sentence_transformer.py      # SentenceTransformerEmbedder(EmbeddingProvider)
├── conversation_store.py            # ConversationStore — embed + upsert + fetch
├── retriever.py                     # MemoryRetriever — similarity + recency scoring
└── consolidator.py                  # MemoryConsolidator — cluster + summarize old memories

tests/memory/
├── __init__.py
├── store/
│   ├── __init__.py
│   └── test_chroma.py
├── embeddings/
│   ├── __init__.py
│   └── test_sentence_transformer.py
├── test_conversation_store.py
├── test_retriever.py
├── test_consolidator.py
└── test_integration_memory.py
```

---

### Task 1: Memory Models

**Files:**
- Create: `ai_clone/memory/__init__.py`
- Create: `ai_clone/memory/models.py`
- Create: `tests/memory/__init__.py`

- [ ] **Step 1: Write the failing test**

Create `tests/memory/test_models.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/vishwasaikarnati/myAIClone && pytest tests/memory/test_models.py -v
```
Expected: ImportError or ModuleNotFoundError

- [ ] **Step 3: Write the implementation**

Create `ai_clone/memory/__init__.py` (empty).

Create `ai_clone/memory/models.py`:

```python
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import List


@dataclass
class MemoryDocument:
    id: str
    text: str
    embedding: List[float]
    timestamp: datetime
    session_id: str
    speaker: str  # "user" | "clone"

    def to_metadata(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
            "speaker": self.speaker,
        }

    @classmethod
    def from_metadata(
        cls,
        id: str,
        text: str,
        embedding: List[float],
        metadata: dict,
    ) -> "MemoryDocument":
        return cls(
            id=id,
            text=text,
            embedding=embedding,
            timestamp=datetime.fromisoformat(metadata["timestamp"]),
            session_id=metadata["session_id"],
            speaker=metadata["speaker"],
        )


@dataclass
class SearchResult:
    document: MemoryDocument
    score: float
```

Create `tests/memory/__init__.py` (empty).

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /Users/vishwasaikarnati/myAIClone && pytest tests/memory/test_models.py -v
```
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
cd /Users/vishwasaikarnati/myAIClone && git add ai_clone/memory/__init__.py ai_clone/memory/models.py tests/memory/__init__.py tests/memory/test_models.py && git commit -m "feat: memory models — MemoryDocument and SearchResult"
```

---

### Task 2: VectorStore ABC + EmbeddingProvider ABC

**Files:**
- Create: `ai_clone/memory/store/__init__.py`
- Create: `ai_clone/memory/store/base.py`
- Create: `ai_clone/memory/embeddings/__init__.py`
- Create: `ai_clone/memory/embeddings/base.py`
- Create: `tests/memory/store/__init__.py`
- Create: `tests/memory/embeddings/__init__.py`
- Create: `tests/memory/test_abstractions.py`

- [ ] **Step 1: Write the failing test**

Create `tests/memory/test_abstractions.py`:

```python
from __future__ import annotations
import pytest
from datetime import datetime
from ai_clone.memory.store.base import VectorStore
from ai_clone.memory.embeddings.base import EmbeddingProvider
from ai_clone.memory.models import MemoryDocument, SearchResult


def test_vector_store_is_abstract():
    with pytest.raises(TypeError):
        VectorStore()  # type: ignore


def test_embedding_provider_is_abstract():
    with pytest.raises(TypeError):
        EmbeddingProvider()  # type: ignore


def test_vector_store_concrete_subclass_requires_all_methods():
    """A partial implementation missing required methods cannot be instantiated."""
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

    store = FakeStore()
    assert store.count() == 0
    assert store.search([0.1], top_k=5) == []
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/vishwasaikarnati/myAIClone && pytest tests/memory/test_abstractions.py -v
```
Expected: ImportError

- [ ] **Step 3: Write the implementation**

Create `ai_clone/memory/store/__init__.py` (empty).

Create `ai_clone/memory/store/base.py`:

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from ..models import MemoryDocument, SearchResult


class VectorStore(ABC):
    """Provider-agnostic interface for a vector store.

    Implement this to add a new backend (Pinecone, Weaviate, pgvector, etc.).
    ChromaVectorStore is the default implementation.
    """

    @abstractmethod
    def upsert(self, documents: List[MemoryDocument]) -> None:
        """Insert or update documents in the store."""
        ...

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int,
        filters: Optional[Dict] = None,
    ) -> List[SearchResult]:
        """Return top-K most similar documents, optionally filtered by metadata."""
        ...

    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """Delete documents by ID."""
        ...

    @abstractmethod
    def count(self) -> int:
        """Return total number of stored documents."""
        ...
```

Create `ai_clone/memory/embeddings/__init__.py` (empty).

Create `ai_clone/memory/embeddings/base.py`:

```python
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
```

Create `tests/memory/store/__init__.py` (empty).
Create `tests/memory/embeddings/__init__.py` (empty).

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /Users/vishwasaikarnati/myAIClone && pytest tests/memory/test_abstractions.py -v
```
Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
cd /Users/vishwasaikarnati/myAIClone && git add ai_clone/memory/store/ ai_clone/memory/embeddings/ tests/memory/store/__init__.py tests/memory/embeddings/__init__.py tests/memory/test_abstractions.py && git commit -m "feat: VectorStore and EmbeddingProvider abstract interfaces"
```

---

### Task 3: SentenceTransformerEmbedder

**Files:**
- Create: `ai_clone/memory/embeddings/sentence_transformer.py`
- Create: `tests/memory/embeddings/test_sentence_transformer.py`

- [ ] **Step 1: Write the failing test**

Create `tests/memory/embeddings/test_sentence_transformer.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/vishwasaikarnati/myAIClone && pytest tests/memory/embeddings/test_sentence_transformer.py -v
```
Expected: ImportError

- [ ] **Step 3: Write the implementation**

Create `ai_clone/memory/embeddings/sentence_transformer.py`:

```python
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
        return [v.tolist() for v in vectors]
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /Users/vishwasaikarnati/myAIClone && pytest tests/memory/embeddings/test_sentence_transformer.py -v
```
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
cd /Users/vishwasaikarnati/myAIClone && git add ai_clone/memory/embeddings/sentence_transformer.py tests/memory/embeddings/test_sentence_transformer.py && git commit -m "feat: SentenceTransformerEmbedder — default embedding provider"
```

---

### Task 4: ChromaVectorStore

**Files:**
- Create: `ai_clone/memory/store/chroma.py`
- Create: `tests/memory/store/test_chroma.py`

- [ ] **Step 1: Write the failing test**

Create `tests/memory/store/test_chroma.py`:

```python
from __future__ import annotations
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch, call
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
    mock_collection.query.return_value = {
        "ids": [[]], "documents": [[]], "embeddings": [[]], "metadatas": [[]], "distances": [[]]
    }
    store = ChromaVectorStore(path=tmp_path)
    store.search([0.1], top_k=3, filters={"speaker": "user"})
    call_kwargs = mock_collection.query.call_args.kwargs
    assert call_kwargs["where"] == {"speaker": "user"}


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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/vishwasaikarnati/myAIClone && pytest tests/memory/store/test_chroma.py -v
```
Expected: ImportError

- [ ] **Step 3: Write the implementation**

Create `ai_clone/memory/store/chroma.py`:

```python
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional
import chromadb
from .base import VectorStore
from ..models import MemoryDocument, SearchResult

_COLLECTION_NAME = "conversations"


class ChromaVectorStore(VectorStore):
    """VectorStore backed by ChromaDB (persistent, local).

    To swap to a different backend, implement VectorStore and pass the new
    implementation to ConversationStore — no other code needs to change.
    """

    def __init__(self, path: Path = Path("./data/chroma")) -> None:
        self._client = chromadb.PersistentClient(path=str(path))
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, documents: List[MemoryDocument]) -> None:
        if not documents:
            return
        self._collection.upsert(
            ids=[d.id for d in documents],
            documents=[d.text for d in documents],
            embeddings=[d.embedding for d in documents],
            metadatas=[d.to_metadata() for d in documents],
        )

    def search(
        self,
        query_embedding: List[float],
        top_k: int,
        filters: Optional[Dict] = None,
    ) -> List[SearchResult]:
        kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "embeddings", "metadatas", "distances"],
        }
        if filters:
            kwargs["where"] = filters

        raw = self._collection.query(**kwargs)
        results = []
        ids = raw["ids"][0]
        texts = raw["documents"][0]
        embeddings = raw["embeddings"][0]
        metadatas = raw["metadatas"][0]
        distances = raw["distances"][0]

        for doc_id, text, emb, meta, dist in zip(ids, texts, embeddings, metadatas, distances):
            doc = MemoryDocument.from_metadata(
                id=doc_id, text=text, embedding=list(emb), metadata=meta
            )
            # ChromaDB returns cosine distance (0=identical); convert to similarity score
            results.append(SearchResult(document=doc, score=1.0 - dist))
        return results

    def delete(self, ids: List[str]) -> None:
        self._collection.delete(ids=ids)

    def count(self) -> int:
        return self._collection.count()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /Users/vishwasaikarnati/myAIClone && pytest tests/memory/store/test_chroma.py -v
```
Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
cd /Users/vishwasaikarnati/myAIClone && git add ai_clone/memory/store/chroma.py tests/memory/store/test_chroma.py && git commit -m "feat: ChromaVectorStore — ChromaDB implementation of VectorStore"
```

---

### Task 5: ConversationStore

**Files:**
- Create: `ai_clone/memory/conversation_store.py`
- Create: `tests/memory/test_conversation_store.py`

- [ ] **Step 1: Write the failing test**

Create `tests/memory/test_conversation_store.py`:

```python
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
    upserted: list[MemoryDocument] = fake_store.upsert.call_args[0][0]
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/vishwasaikarnati/myAIClone && pytest tests/memory/test_conversation_store.py -v
```
Expected: ImportError

- [ ] **Step 3: Write the implementation**

Create `ai_clone/memory/conversation_store.py`:

```python
from __future__ import annotations
import uuid
from datetime import datetime
from typing import List, Optional
from .embeddings.base import EmbeddingProvider
from .models import MemoryDocument, SearchResult
from .store.base import VectorStore


class ConversationStore:
    """Stores and retrieves conversation messages using a pluggable VectorStore + EmbeddingProvider.

    Usage:
        store = ConversationStore(
            vector_store=ChromaVectorStore(path=Path("./data/chroma")),
            embedder=SentenceTransformerEmbedder(),
        )
        store.add_message(text="Hello", speaker="user", session_id="s1", timestamp=datetime.utcnow())
        results = store.search("how are you", top_k=5)
    """

    def __init__(self, vector_store: VectorStore, embedder: EmbeddingProvider) -> None:
        self._store = vector_store
        self._embedder = embedder

    def add_message(
        self,
        text: str,
        speaker: str,
        session_id: str,
        timestamp: datetime,
    ) -> MemoryDocument:
        """Embed and persist a single message. Returns the stored document."""
        embedding = self._embedder.embed([text])[0]
        doc = MemoryDocument(
            id=str(uuid.uuid4()),
            text=text,
            embedding=embedding,
            timestamp=timestamp,
            session_id=session_id,
            speaker=speaker,
        )
        self._store.upsert([doc])
        return doc

    def search(
        self,
        query: str,
        top_k: int,
        session_id: Optional[str] = None,
    ) -> List[SearchResult]:
        """Embed the query and return top-K semantically similar stored messages."""
        embedding = self._embedder.embed([query])[0]
        filters = {"session_id": session_id} if session_id else None
        return self._store.search(
            query_embedding=embedding, top_k=top_k, filters=filters
        )

    def count(self) -> int:
        return self._store.count()

    def delete(self, ids: List[str]) -> None:
        self._store.delete(ids)

    def get_older_than(self, cutoff: datetime, limit: int = 100) -> List[MemoryDocument]:
        return self._store.get_older_than(cutoff, limit=limit)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /Users/vishwasaikarnati/myAIClone && pytest tests/memory/test_conversation_store.py -v
```
Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
cd /Users/vishwasaikarnati/myAIClone && git add ai_clone/memory/conversation_store.py tests/memory/test_conversation_store.py && git commit -m "feat: ConversationStore — embed and persist messages via pluggable providers"
```

---

### Task 6: MemoryRetriever

**Files:**
- Create: `ai_clone/memory/retriever.py`
- Create: `tests/memory/test_retriever.py`

- [ ] **Step 1: Write the failing test**

Create `tests/memory/test_retriever.py`:

```python
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
        timestamp=datetime(2024, 6, 1) - timedelta(days=days_ago),
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/vishwasaikarnati/myAIClone && pytest tests/memory/test_retriever.py -v
```
Expected: ImportError

- [ ] **Step 3: Write the implementation**

Create `ai_clone/memory/retriever.py`:

```python
from __future__ import annotations
import math
from datetime import datetime, timezone
from typing import List
from .conversation_store import ConversationStore
from .models import SearchResult

_DEFAULT_HALF_LIFE_DAYS = 30


class MemoryRetriever:
    """Returns semantically similar memories, re-ranked by recency.

    Score formula: final_score = similarity_score * recency_weight
    where recency_weight = 2^(-age_days / half_life_days)

    This means memories lose half their recency weight every `recency_half_life_days`.
    A memory from 30 days ago (default) has 0.5x the recency weight of a fresh memory.
    """

    def __init__(
        self,
        conversation_store: ConversationStore,
        recency_half_life_days: float = _DEFAULT_HALF_LIFE_DAYS,
    ) -> None:
        self._store = conversation_store
        self._half_life_days = recency_half_life_days

    def retrieve(self, query: str, top_k: int) -> List[SearchResult]:
        """Return top-K memories re-ranked by similarity × recency weight."""
        results = self._store.search(query, top_k=top_k)
        now = datetime.now(tz=timezone.utc)
        reranked = []
        for result in results:
            ts = result.document.timestamp
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            age_days = (now - ts).total_seconds() / 86400
            recency_weight = math.pow(2.0, -age_days / self._half_life_days)
            reranked.append(
                SearchResult(
                    document=result.document,
                    score=result.score * recency_weight,
                )
            )
        reranked.sort(key=lambda r: r.score, reverse=True)
        return reranked
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /Users/vishwasaikarnati/myAIClone && pytest tests/memory/test_retriever.py -v
```
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
cd /Users/vishwasaikarnati/myAIClone && git add ai_clone/memory/retriever.py tests/memory/test_retriever.py && git commit -m "feat: MemoryRetriever — similarity search with exponential recency weighting"
```

---

### Task 7: MemoryConsolidator

**Files:**
- Create: `ai_clone/memory/consolidator.py`
- Create: `tests/memory/test_consolidator.py`

- [ ] **Step 1: Write the failing test**

Create `tests/memory/test_consolidator.py`:

```python
from __future__ import annotations
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from ai_clone.memory.consolidator import MemoryConsolidator
from ai_clone.memory.models import MemoryDocument, SearchResult


def _make_old_result(id: str, days_ago: int = 100) -> SearchResult:
    doc = MemoryDocument(
        id=id,
        text=f"old message {id}",
        embedding=[0.1, 0.2],
        timestamp=datetime(2024, 1, 1) - timedelta(days=days_ago),
        session_id="s1",
        speaker="user",
    )
    return SearchResult(document=doc, score=0.5)


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
    fake_store.count.return_value = 50  # below default threshold of 100
    consolidator = MemoryConsolidator(
        conversation_store=fake_store,
        openrouter_client=fake_openrouter,
        threshold=100,
    )
    deleted = consolidator.consolidate(older_than_days=60)
    assert deleted == 0
    fake_openrouter.chat.completions.create.assert_not_called()


def test_consolidate_fetches_old_messages_and_summarises(fake_store, fake_openrouter):
    old_results = [_make_old_result(f"doc-{i}", days_ago=90) for i in range(5)]
    fake_store.search.return_value = old_results
    fake_store.count.return_value = 200

    consolidator = MemoryConsolidator(
        conversation_store=fake_store,
        openrouter_client=fake_openrouter,
        threshold=100,
    )
    deleted = consolidator.consolidate(older_than_days=60)

    fake_openrouter.chat.completions.create.assert_called_once()
    # Should delete the old documents and store the summary
    fake_store.delete.assert_called_once()
    deleted_ids = fake_store.delete.call_args[0][0]
    assert set(deleted_ids) == {f"doc-{i}" for i in range(5)}
    fake_store.add_message.assert_called_once()
    assert deleted == 5


def test_consolidate_returns_zero_when_no_old_messages(fake_store, fake_openrouter):
    fake_store.search.return_value = []
    consolidator = MemoryConsolidator(
        conversation_store=fake_store,
        openrouter_client=fake_openrouter,
        threshold=100,
    )
    deleted = consolidator.consolidate(older_than_days=60)
    assert deleted == 0
    fake_openrouter.chat.completions.create.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/vishwasaikarnati/myAIClone && pytest tests/memory/test_consolidator.py -v
```
Expected: ImportError

- [ ] **Step 3: Write the implementation**

Create `ai_clone/memory/consolidator.py`:

```python
from __future__ import annotations
from datetime import datetime, timedelta, timezone
from typing import Any, List
from .conversation_store import ConversationStore
from .models import SearchResult

_DEFAULT_THRESHOLD = 100
_SUMMARY_SPEAKER = "summary"
_SUMMARY_SESSION = "consolidated"


class MemoryConsolidator:
    """Summarises old conversation clusters to keep the vector store lean.

    How it works:
    1. Skip consolidation if the store has fewer than `threshold` documents.
    2. Search for a broad sample of documents older than `older_than_days`.
    3. If any found, send their text to OpenRouter and ask for a summary.
    4. Delete the originals and store the summary as a single new document.

    This reduces ChromaDB size while preserving semantic content.
    """

    def __init__(
        self,
        conversation_store: ConversationStore,
        openrouter_client: Any,
        threshold: int = _DEFAULT_THRESHOLD,
        model: str = "tencent/hy3-preview:free",
    ) -> None:
        self._store = conversation_store
        self._client = openrouter_client
        self._threshold = threshold
        self._model = model

    def consolidate(self, older_than_days: int = 60) -> int:
        """Consolidate old memories. Returns number of documents deleted."""
        if self._store.count() < self._threshold:
            return 0

        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=older_than_days)
        # Use a broad semantic query to surface old messages (sentinel phrase)
        old_results: List[SearchResult] = self._store.search(
            "conversation history", top_k=50
        )
        old_results = [
            r for r in old_results
            if _as_utc(r.document.timestamp) < cutoff
        ]
        if not old_results:
            return 0

        texts = [r.document.text for r in old_results]
        summary = self._summarise(texts)
        ids = [r.document.id for r in old_results]
        self._store.delete(ids)

        self._store.add_message(
            text=summary,
            speaker=_SUMMARY_SPEAKER,
            session_id=_SUMMARY_SESSION,
            timestamp=datetime.now(tz=timezone.utc),
        )
        return len(ids)

    def _summarise(self, texts: List[str]) -> str:
        joined = "\n".join(f"- {t}" for t in texts)
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": "You are summarising past conversation history to compact it. Preserve key facts, topics, and personality cues.",
                },
                {
                    "role": "user",
                    "content": f"Summarise these conversation excerpts concisely:\n{joined}",
                },
            ],
        )
        return response.choices[0].message.content


def _as_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /Users/vishwasaikarnati/myAIClone && pytest tests/memory/test_consolidator.py -v
```
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
cd /Users/vishwasaikarnati/myAIClone && git add ai_clone/memory/consolidator.py tests/memory/test_consolidator.py && git commit -m "feat: MemoryConsolidator — summarise and compact old conversation clusters"
```

---

### Task 8: Integration Test + PR

**Files:**
- Create: `tests/memory/test_integration_memory.py`
- Create: `tests/memory/__init__.py` (if not yet created)

- [ ] **Step 1: Write the failing test**

Create `tests/memory/test_integration_memory.py`:

```python
from __future__ import annotations
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch
from ai_clone.memory.models import MemoryDocument, SearchResult
from ai_clone.memory.conversation_store import ConversationStore
from ai_clone.memory.retriever import MemoryRetriever


@pytest.fixture
def fake_embedder():
    embedder = MagicMock()
    # Return slightly different embeddings so search returns varied results
    call_count = [0]
    def side_effect(texts):
        call_count[0] += 1
        return [[float(call_count[0]) * 0.1, 0.2, 0.3, 0.4]]
    embedder.embed.side_effect = side_effect
    return embedder


@pytest.fixture
def fake_vector_store():
    store = MagicMock()
    stored = []

    def upsert(docs):
        stored.extend(docs)
    def search(query_embedding, top_k, filters=None):
        return [SearchResult(doc, 0.9 - i * 0.1) for i, doc in enumerate(stored[:top_k])]
    def count():
        return len(stored)

    store.upsert.side_effect = upsert
    store.search.side_effect = search
    store.count.side_effect = count
    return store, stored


def test_store_and_retrieve_messages(fake_embedder, fake_vector_store):
    vector_store, stored = fake_vector_store
    cs = ConversationStore(vector_store=vector_store, embedder=fake_embedder)
    retriever = MemoryRetriever(conversation_store=cs, recency_half_life_days=30)

    cs.add_message("Hello there", "user", "s1", datetime(2024, 6, 1, 10, 0))
    cs.add_message("I am fine thanks", "clone", "s1", datetime(2024, 6, 1, 10, 1))

    assert cs.count() == 2

    results = retriever.retrieve("how are you", top_k=2)
    assert len(results) == 2
    # Results should be sorted by final score (descending)
    assert results[0].score >= results[1].score


def test_retriever_reranks_by_recency(fake_embedder, fake_vector_store):
    from datetime import timedelta
    vector_store, stored = fake_vector_store
    cs = ConversationStore(vector_store=vector_store, embedder=fake_embedder)
    retriever = MemoryRetriever(conversation_store=cs, recency_half_life_days=1)

    # Add one recent and one old message
    recent_ts = datetime(2024, 6, 1)
    old_ts = datetime(2023, 1, 1)

    cs.add_message("recent message", "user", "s1", recent_ts)
    cs.add_message("old message", "user", "s1", old_ts)

    results = retriever.retrieve("some query", top_k=2)
    # With half_life of 1 day, the old message's score should be dramatically reduced
    # Both results should still be returned
    assert len(results) == 2
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/vishwasaikarnati/myAIClone && pytest tests/memory/test_integration_memory.py -v
```
Expected: 2 tests fail or error

- [ ] **Step 3: Run full suite to check baseline**

```bash
cd /Users/vishwasaikarnati/myAIClone && pytest -v
```

- [ ] **Step 4: Implement any missing `__init__.py` files**

Ensure the following exist (create empty if missing):
- `tests/memory/__init__.py`
- `tests/memory/store/__init__.py`
- `tests/memory/embeddings/__init__.py`

- [ ] **Step 5: Run integration test to verify it passes**

```bash
cd /Users/vishwasaikarnati/myAIClone && pytest tests/memory/test_integration_memory.py -v
```
Expected: 2 PASSED

- [ ] **Step 6: Run full suite**

```bash
cd /Users/vishwasaikarnati/myAIClone && pytest -v
```
Expected: all tests pass (48 + new memory tests)

- [ ] **Step 7: Commit**

```bash
cd /Users/vishwasaikarnati/myAIClone && git add tests/memory/ && git commit -m "test: memory layer integration test — store, retrieve, recency reranking"
```

- [ ] **Step 8: Push feature branch**

```bash
cd /Users/vishwasaikarnati/myAIClone && git push -u origin feature/memory-layer
```

- [ ] **Step 9: Open PR**

```bash
gh pr create \
  --title "feat: memory layer — provider-agnostic vector store + ChromaDB + retriever" \
  --body "$(cat <<'EOF'
## Summary

- Adds `VectorStore` and `EmbeddingProvider` abstract interfaces so any backend (Pinecone, Weaviate, pgvector, OpenAI embeddings, etc.) can be swapped in without changing calling code
- Implements `ChromaVectorStore` (default backend) and `SentenceTransformerEmbedder` (default embedder)
- `ConversationStore` orchestrates embed + upsert + search via injected providers
- `MemoryRetriever` re-ranks similarity results with exponential recency decay (`score = similarity × 2^(-age/half_life)`)
- `MemoryConsolidator` compacts old conversation clusters via OpenRouter summary

## Architecture

The key extensibility point is the two ABCs:

\`\`\`python
class VectorStore(ABC):         # swap ChromaDB → Pinecone, pgvector, etc.
class EmbeddingProvider(ABC):   # swap sentence-transformers → OpenAI, Cohere, etc.
\`\`\`

Both are injected into `ConversationStore`, so no code outside `store/` and `embeddings/` needs to change when providers are swapped.

## Test plan

- [ ] Unit tests for all components (mocked dependencies)
- [ ] Integration test: store → retrieve → recency rerank pipeline
- [ ] All 48 pre-existing tests still pass
- [ ] Run: `pytest -v`

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Report the PR URL and wait for CodeRabbit review.
