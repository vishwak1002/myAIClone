import pytest
from datetime import datetime
from pathlib import Path

from ai_clone.ingestion.parsers.whatsapp import WhatsAppParser
from ai_clone.ingestion.models import Platform


@pytest.fixture
def bracket_format_file(tmp_path):
    content = (
        "[23/05/2023, 14:32:10] Vishwa: Hello there\n"
        "[23/05/2023, 14:32:15] John: Hi how are you?\n"
        "[23/05/2023, 14:33:00] Vishwa: Doing great!\n"
        "This line has no pattern and should be skipped\n"
    )
    p = tmp_path / "WhatsApp Chat with John.txt"
    p.write_text(content, encoding="utf-8")
    return p


@pytest.fixture
def dash_format_file(tmp_path):
    content = (
        "5/23/23, 2:32 PM - Vishwa: Hello\n"
        "5/23/23, 2:33 PM - John: Hey!\n"
    )
    p = tmp_path / "WhatsApp Chat.txt"
    p.write_text(content, encoding="utf-8")
    return p


def test_parses_bracket_format(bracket_format_file):
    parser = WhatsAppParser()
    messages = parser.parse(bracket_format_file)
    assert len(messages) == 3
    assert messages[0].sender == "Vishwa"
    assert messages[0].text == "Hello there"
    assert messages[0].platform == Platform.WHATSAPP
    assert messages[0].timestamp == datetime(2023, 5, 23, 14, 32, 10)


def test_parses_dash_format(dash_format_file):
    parser = WhatsAppParser()
    messages = parser.parse(dash_format_file)
    assert len(messages) == 2
    assert messages[0].sender == "Vishwa"
    assert messages[0].text == "Hello"


def test_skips_unparseable_lines(bracket_format_file):
    parser = WhatsAppParser()
    messages = parser.parse(bracket_format_file)
    assert len(messages) == 3  # 4 lines, 1 skipped


def test_can_parse_txt_file(tmp_path):
    parser = WhatsAppParser()
    f = tmp_path / "WhatsApp Chat.txt"
    f.touch()
    assert parser.can_parse(f) is True


def test_cannot_parse_json_file(tmp_path):
    parser = WhatsAppParser()
    f = tmp_path / "result.json"
    f.touch()
    assert parser.can_parse(f) is False
