from __future__ import annotations
import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class PersonalityProfile:
    avg_sentence_length: float = 0.0
    emoji_frequency: float = 0.0
    top_phrases: List[str] = field(default_factory=list)
    formality_score: float = 0.5
    punctuation_style: Dict[str, float] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(dataclasses.asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> PersonalityProfile:
        with open(path) as f:
            data = json.load(f)
        return cls(**data)
