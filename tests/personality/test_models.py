import json
import pytest
from pathlib import Path

from ai_clone.personality.models import PersonalityProfile


def test_default_profile_has_zero_values():
    profile = PersonalityProfile()
    assert profile.avg_sentence_length == 0.0
    assert profile.emoji_frequency == 0.0
    assert profile.top_phrases == []


def test_save_and_load_roundtrip(tmp_path):
    profile = PersonalityProfile(
        avg_sentence_length=8.5,
        emoji_frequency=0.3,
        top_phrases=["how are", "sounds good"],
        formality_score=0.4,
        punctuation_style={"!": 0.2, "?": 0.1},
        tone_traits={"humor": 0.7},
        top_topics=["tech", "travel"],
    )
    path = tmp_path / "profile.json"
    profile.save(path)
    loaded = PersonalityProfile.load(path)
    assert loaded.avg_sentence_length == 8.5
    assert loaded.top_phrases == ["how are", "sounds good"]
    assert loaded.tone_traits == {"humor": 0.7}


def test_save_creates_parent_dirs(tmp_path):
    profile = PersonalityProfile()
    path = tmp_path / "data" / "nested" / "profile.json"
    profile.save(path)
    assert path.exists()
