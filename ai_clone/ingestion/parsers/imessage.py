from __future__ import annotations
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

from .base import BaseParser
from ..models import ParsedMessage, Platform

_IMESSAGE_EPOCH = datetime(2001, 1, 1)


class iMessageParser(BaseParser):
    def can_parse(self, path: Path) -> bool:
        return path.suffix == ".db"

    def parse(self, path: Path) -> List[ParsedMessage]:
        conn = sqlite3.connect(path)
        messages = []
        try:
            rows = conn.execute("""
                SELECT
                    m.date,
                    m.text,
                    m.is_from_me,
                    COALESCE(h.id, 'Me') AS handle
                FROM message m
                LEFT JOIN handle h ON m.handle_id = h.rowid
                WHERE m.text IS NOT NULL AND m.text != ''
                ORDER BY m.date ASC
            """).fetchall()
            for date_ns, text, is_from_me, handle in rows:
                timestamp = _IMESSAGE_EPOCH + timedelta(seconds=date_ns / 1e9)
                sender = "Me" if is_from_me else handle
                messages.append(ParsedMessage(
                    sender=sender,
                    text=text.strip(),
                    timestamp=timestamp,
                    platform=Platform.IMESSAGE,
                ))
        finally:
            conn.close()
        return messages
