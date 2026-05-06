from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import List, Union

from .base import BaseParser
from ..models import ParsedMessage, Platform


class TelegramParser(BaseParser):
    def can_parse(self, path: Path) -> bool:
        return path.suffix == ".json"

    def parse(self, path: Path) -> List[ParsedMessage]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        messages = []
        for msg in data.get("messages", []):
            if msg.get("type") != "message":
                continue
            text = self._extract_text(msg.get("text", ""))
            if not text:
                continue
            messages.append(ParsedMessage(
                sender=msg.get("from", "Unknown"),
                text=text,
                timestamp=datetime.fromisoformat(msg["date"]),
                platform=Platform.TELEGRAM,
            ))
        return messages

    def _extract_text(self, raw: Union[str, list]) -> str:
        if isinstance(raw, str):
            return raw.strip()
        return "".join(
            part if isinstance(part, str) else part.get("text", "")
            for part in raw
        ).strip()
