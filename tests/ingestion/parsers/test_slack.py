import json
import zipfile
import pytest
from datetime import datetime
from pathlib import Path

from ai_clone.ingestion.parsers.slack import SlackParser
from ai_clone.ingestion.models import Platform


@pytest.fixture
def slack_zip(tmp_path):
    zip_path = tmp_path / "slack_export.zip"
    users = [
        {"id": "U001", "real_name": "Vishwa"},
        {"id": "U002", "real_name": "John"},
    ]
    messages = [
        {"type": "message", "user": "U001", "text": "Hello team", "ts": "1684849930.000100"},
        {"type": "message", "user": "U002", "text": "Hey!", "ts": "1684849935.000200"},
        {"type": "message", "subtype": "channel_join", "user": "U001",
         "text": "joined", "ts": "1684849940.000300"},
        {"type": "message", "user": "U001", "text": "", "ts": "1684849945.000400"},
    ]
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("users.json", json.dumps(users))
        zf.writestr("general/2023-05-23.json", json.dumps(messages))
    return zip_path


def test_parses_messages(slack_zip):
    parser = SlackParser()
    messages = parser.parse(slack_zip)
    assert len(messages) == 2


def test_resolves_user_names(slack_zip):
    parser = SlackParser()
    messages = parser.parse(slack_zip)
    assert messages[0].sender == "Vishwa"
    assert messages[1].sender == "John"


def test_skips_subtype_messages(slack_zip):
    parser = SlackParser()
    messages = parser.parse(slack_zip)
    assert all(m.text != "joined" for m in messages)


def test_skips_empty_text(slack_zip):
    parser = SlackParser()
    messages = parser.parse(slack_zip)
    assert all(m.text for m in messages)


def test_platform_is_slack(slack_zip):
    parser = SlackParser()
    messages = parser.parse(slack_zip)
    assert all(m.platform == Platform.SLACK for m in messages)


def test_can_parse_zip(tmp_path):
    parser = SlackParser()
    f = tmp_path / "export.zip"
    f.touch()
    assert parser.can_parse(f) is True


def test_cannot_parse_json(tmp_path):
    parser = SlackParser()
    f = tmp_path / "data.json"
    f.touch()
    assert parser.can_parse(f) is False
