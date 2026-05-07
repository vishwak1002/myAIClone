from __future__ import annotations
from dataclasses import dataclass
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
            "timestamp_unix": self.timestamp.timestamp(),
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
