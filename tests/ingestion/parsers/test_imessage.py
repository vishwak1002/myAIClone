import sqlite3
import pytest
from datetime import datetime
from pathlib import Path

from ai_clone.ingestion.parsers.imessage import iMessageParser
from ai_clone.ingestion.models import Platform

SAMPLE_DATE_NS = 706710730 * 1_000_000_000


@pytest.fixture
def imessage_db(tmp_path):
    db_path = tmp_path / "chat.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE handle (rowid INTEGER PRIMARY KEY, id TEXT)")
    conn.execute("INSERT INTO handle (rowid, id) VALUES (1, 'john@example.com')")
    conn.execute("""
        CREATE TABLE message (
            rowid INTEGER PRIMARY KEY,
            date INTEGER,
            text TEXT,
            is_from_me INTEGER,
            handle_id INTEGER
        )
    """)
    conn.execute("INSERT INTO message VALUES (1, ?, 'Hello there', 1, 0)", (SAMPLE_DATE_NS,))
    conn.execute("INSERT INTO message VALUES (2, ?, 'Hi!', 0, 1)", (SAMPLE_DATE_NS + 5_000_000_000,))
    conn.execute("INSERT INTO message VALUES (3, ?, NULL, 1, 0)", (SAMPLE_DATE_NS + 10_000_000_000,))
    conn.commit()
    conn.close()
    return db_path


def test_parses_messages(imessage_db):
    parser = iMessageParser()
    messages = parser.parse(imessage_db)
    assert len(messages) == 2


def test_me_sender_for_own_messages(imessage_db):
    parser = iMessageParser()
    messages = parser.parse(imessage_db)
    assert messages[0].sender == "Me"


def test_handle_sender_for_received(imessage_db):
    parser = iMessageParser()
    messages = parser.parse(imessage_db)
    assert messages[1].sender == "john@example.com"


def test_skips_null_text(imessage_db):
    parser = iMessageParser()
    messages = parser.parse(imessage_db)
    assert all(m.text for m in messages)


def test_platform_is_imessage(imessage_db):
    parser = iMessageParser()
    messages = parser.parse(imessage_db)
    assert all(m.platform == Platform.IMESSAGE for m in messages)


def test_can_parse_db_file(tmp_path):
    parser = iMessageParser()
    f = tmp_path / "chat.db"
    f.touch()
    assert parser.can_parse(f) is True


def test_cannot_parse_txt(tmp_path):
    parser = iMessageParser()
    f = tmp_path / "chat.txt"
    f.touch()
    assert parser.can_parse(f) is False
