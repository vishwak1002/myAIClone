import json
import pytest
from pathlib import Path

from ai_clone.ingestion.pipeline import IngestionPipeline
from ai_clone.personality.builder import PersonalityBuilder
from ai_clone.personality.updater import ProfileUpdater


@pytest.fixture
def telegram_export(tmp_path):
    data = {
        "messages": [
            {"id": 1, "type": "message", "date": "2023-05-23T10:00:00",
             "from": "Vishwa", "text": "Hey, how are you doing today?"},
            {"id": 2, "type": "message", "date": "2023-05-23T10:01:00",
             "from": "John", "text": "I'm good, thanks!"},
            {"id": 3, "type": "message", "date": "2023-05-23T10:02:00",
             "from": "Vishwa", "text": "That's great to hear. How are things going at work?"},
            {"id": 4, "type": "message", "date": "2023-05-23T10:03:00",
             "from": "John", "text": "Pretty busy lately."},
            {"id": 5, "type": "message", "date": "2023-05-23T10:04:00",
             "from": "Vishwa", "text": "I know the feeling. Let me know if you need help."},
            {"id": 6, "type": "message", "date": "2023-05-23T10:05:00",
             "from": "Vishwa", "text": "We should catch up soon over coffee."},
            {"id": 7, "type": "message", "date": "2023-05-23T10:06:00",
             "from": "John", "text": "Sounds good!"},
            {"id": 8, "type": "message", "date": "2023-05-23T10:07:00",
             "from": "Vishwa", "text": "How about this weekend?"},
            {"id": 9, "type": "message", "date": "2023-05-23T10:08:00",
             "from": "Vishwa", "text": "I'm free on Saturday afternoon."},
            {"id": 10, "type": "message", "date": "2023-05-23T10:09:00",
             "from": "Vishwa", "text": "We could go to that new cafe downtown."},
            {"id": 11, "type": "message", "date": "2023-05-23T10:10:00",
             "from": "Vishwa", "text": "I heard they have really good espresso."},
            {"id": 12, "type": "message", "date": "2023-05-23T10:11:00",
             "from": "John", "text": "That sounds amazing!"},
            {"id": 13, "type": "message", "date": "2023-05-23T10:12:00",
             "from": "Vishwa", "text": "How are you getting there? Do you need a ride?"},
            {"id": 14, "type": "message", "date": "2023-05-23T10:13:00",
             "from": "Vishwa", "text": "I can pick you up if needed."},
            {"id": 15, "type": "service", "date": "2023-05-23T10:14:00",
             "from": "Vishwa", "text": "should be skipped"},
            {"id": 15, "type": "message", "date": "2023-05-23T10:14:00",
             "from": "Vishwa", "text": ""},
        ]
    }
    p = tmp_path / "result.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_full_ingest_to_profile_pipeline(tmp_path, telegram_export):
    pipeline = IngestionPipeline(user_name="Vishwa")
    user_messages = pipeline.ingest_user_messages(telegram_export)
    assert len(user_messages) == 10
    assert all(m.sender == "Vishwa" for m in user_messages)
    builder = PersonalityBuilder()
    updater = ProfileUpdater(profile_path=tmp_path / "profile.json", builder=builder)
    profile = updater.update(new_messages=user_messages, existing_messages=[])
    assert (tmp_path / "profile.json").exists()
    assert profile.avg_sentence_length > 0
    assert "how are" in profile.top_phrases or len(profile.top_phrases) > 0
