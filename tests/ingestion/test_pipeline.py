import json
import pytest
from pathlib import Path

from ai_clone.ingestion.pipeline import IngestionPipeline


@pytest.fixture
def telegram_file(tmp_path):
    data = {
        "messages": [
            {"id": 1, "type": "message", "date": "2023-05-23T14:32:10",
             "from": "Vishwa", "text": "Hello"},
            {"id": 2, "type": "message", "date": "2023-05-23T14:32:15",
             "from": "John", "text": "Hi!"},
        ]
    }
    p = tmp_path / "result.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_ingest_returns_all_messages(telegram_file):
    pipeline = IngestionPipeline(user_name="Vishwa")
    messages = pipeline.ingest(telegram_file)
    assert len(messages) == 2


def test_ingest_user_messages_filters_to_user(telegram_file):
    pipeline = IngestionPipeline(user_name="Vishwa")
    messages = pipeline.ingest_user_messages(telegram_file)
    assert len(messages) == 1
    assert messages[0].sender == "Vishwa"


def test_ingest_raises_for_unknown_format(tmp_path):
    pipeline = IngestionPipeline(user_name="Vishwa")
    unknown = tmp_path / "data.xml"
    unknown.touch()
    with pytest.raises(ValueError, match="No parser found"):
        pipeline.ingest(unknown)
