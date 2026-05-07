from __future__ import annotations
from datetime import datetime, timedelta, timezone
from typing import Any, List
from .conversation_store import ConversationStore

_DEFAULT_THRESHOLD = 100
_SUMMARY_SPEAKER = "summary"
_SUMMARY_SESSION = "consolidated"


class MemoryConsolidator:
    """Summarises old conversation clusters to keep the vector store lean.

    How it works:
    1. Skip if store has fewer than `threshold` documents.
    2. Search for documents older than `older_than_days`.
    3. Summarise their text via OpenRouter.
    4. Delete the originals and store the summary as a single new document.
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
        old_docs = self._store.get_older_than(cutoff, limit=50)
        if not old_docs:
            return 0

        texts = [d.text for d in old_docs]
        summary = self._summarise(texts)
        ids = [d.id for d in old_docs]

        self._store.add_message(
            text=summary,
            speaker=_SUMMARY_SPEAKER,
            session_id=_SUMMARY_SESSION,
            timestamp=datetime.now(tz=timezone.utc),
        )
        self._store.delete(ids)
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


