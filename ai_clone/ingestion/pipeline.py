from __future__ import annotations
from pathlib import Path
from typing import List, Optional

from .models import ParsedMessage
from .parsers.base import BaseParser
from .parsers.whatsapp import WhatsAppParser
from .parsers.telegram import TelegramParser
from .parsers.imessage import iMessageParser
from .parsers.slack import SlackParser


class IngestionPipeline:
    def __init__(self, user_name: str) -> None:
        self.user_name = user_name
        self._parsers: List[BaseParser] = [
            SlackParser(),
            TelegramParser(),
            iMessageParser(),
            WhatsAppParser(),
        ]

    def ingest(self, path: Path) -> List[ParsedMessage]:
        parser = self._select_parser(path)
        if parser is None:
            raise ValueError(f"No parser found for {path}")
        return parser.parse(path)

    def ingest_user_messages(self, path: Path) -> List[ParsedMessage]:
        return [m for m in self.ingest(path) if m.sender == self.user_name]

    def _select_parser(self, path: Path) -> Optional[BaseParser]:
        for parser in self._parsers:
            if parser.can_parse(path):
                return parser
        return None
