from datetime import datetime
from pathlib import Path
import pytest

from ai_clone.ingestion.models import ParsedMessage, Platform
from ai_clone.personality.builder import PersonalityBuilder
from ai_clone.personality.updater import ProfileUpdater


def _msg(text: str) -> ParsedMessage:
    return ParsedMessage(
        sender="Vishwa", text=text,
        timestamp=datetime(2023, 1, 1), platform=Platform.WHATSAPP
    )


def test_update_saves_new_profile(tmp_path):
    path = tmp_path / "profile.json"
    updater = ProfileUpdater(profile_path=path, builder=PersonalityBuilder())
    profile = updater.update(
        new_messages=[_msg("hello world")],
        existing_messages=[],
    )
    assert path.exists()
    assert profile.avg_sentence_length == 2.0


def test_update_uses_rolling_window(tmp_path):
    from ai_clone.personality.updater import ROLLING_WINDOW
    path = tmp_path / "profile.json"
    updater = ProfileUpdater(profile_path=path, builder=PersonalityBuilder())
    existing = [_msg("old message")] * (ROLLING_WINDOW + 10)
    new_msgs = [_msg("new one two three")]
    profile = updater.update(new_messages=new_msgs, existing_messages=existing)
    assert profile.avg_sentence_length > 0


def test_update_combines_new_and_existing(tmp_path):
    path = tmp_path / "profile.json"
    updater = ProfileUpdater(profile_path=path, builder=PersonalityBuilder())
    existing = [_msg("one")]
    new_msgs = [_msg("two three")]
    profile = updater.update(new_messages=new_msgs, existing_messages=existing)
    assert profile.avg_sentence_length == 1.5
