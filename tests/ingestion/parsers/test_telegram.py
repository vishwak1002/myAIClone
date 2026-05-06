import json
import pytest
from datetime import datetime
from pathlib import Path

from ai_clone.ingestion.parsers.telegram import TelegramParser
from ai_clone.ingestion.models import Platform


@pytest.fixture
def telegram_file(tmp_path):
    data = {
        "messages": [
            {"id": 1, "type": "message", "date": "2023-05-23T14:32:10",
             "from": "Vishwa", "text": "Hello"},
            {"id": 2, "type": "message", "date": "2023-05-23T14:32:15",
             "from": "John", "text": "Hi!"},
            {"id": 3, "type": "service", "date": "2023-05-23T14:33:00",
             "from": "Vishwa", "text": "should be skipped"},
            {"id": 4, "type": "message", "date": "2023-05-23T14:34:00",
             "from": "Vishwa", "text": ""},
        ]
    }
    p = tmp_path / "result.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


@pytest.fixture
def rich_text_file(tmp_path):
    data = {
        "messages": [
            {"id": 1, "type": "message", "date": "2023-05-23T14:32:10",
             "from": "Vishwa",
             "text": [{"type": "bold", "text": "Hello"}, " world"]},
        ]
    }
    p = tmp_path / "result.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_parses_plain_messages(telegram_file):
    parser = TelegramParser()
    messages = parser.parse(telegram_file)
    assert len(messages) == 2
    assert messages[0].sender == "Vishwa"
    assert messages[0].text == "Hello"
    assert messages[0].platform == Platform.TELEGRAM
    assert messages[0].timestamp == datetime(2023, 5, 23, 14, 32, 10)


def test_skips_service_messages(telegram_file):
    parser = TelegramParser()
    messages = parser.parse(telegram_file)
    senders = [m.sender for m in messages]
    assert len([s for s in senders if s == "Vishwa"]) == 1


def test_skips_empty_text(telegram_file):
    parser = TelegramParser()
    messages = parser.parse(telegram_file)
    assert all(m.text for m in messages)


def test_parses_rich_text(rich_text_file):
    parser = TelegramParser()
    messages = parser.parse(rich_text_file)
    assert messages[0].text == "Hello world"


def test_can_parse_json(tmp_path):
    parser = TelegramParser()
    f = tmp_path / "result.json"
    f.touch()
    assert parser.can_parse(f) is True


def test_cannot_parse_txt(tmp_path):
    parser = TelegramParser()
    f = tmp_path / "chat.txt"
    f.touch()
    assert parser.can_parse(f) is False
