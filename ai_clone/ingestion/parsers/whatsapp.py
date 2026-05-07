from __future__ import annotations
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .base import BaseParser
from ..models import ParsedMessage, Platform


class WhatsAppParser(BaseParser):
    _BRACKET = re.compile(
        r"\[(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}:\d{2})\]\s(.+?):\s(.+)"
    )
    _DASH = re.compile(
        r"(\d{1,2}/\d{1,2}/\d{2}),\s(\d{1,2}:\d{2}\s(?:AM|PM))\s-\s(.+?):\s(.+)"
    )

    def can_parse(self, path: Path) -> bool:
        return path.suffix == ".txt"

    def parse(self, path: Path) -> List[ParsedMessage]:
        messages = []
        # NOTE: messages are read line-by-line; multiline messages (those with
        # embedded newlines in the export) will be silently truncated to their
        # first line.
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                msg = self._parse_line(line.strip())
                if msg:
                    messages.append(msg)
        return messages

    def _parse_line(self, line: str) -> Optional[ParsedMessage]:
        m = self._BRACKET.match(line)
        if m:
            date_s, time_s, sender, text = m.groups()
            try:
                ts = datetime.strptime(f"{date_s} {time_s}", "%d/%m/%Y %H:%M:%S")
                return ParsedMessage(sender=sender.strip(), text=text.strip(),
                                     timestamp=ts, platform=Platform.WHATSAPP)
            except ValueError:
                pass
            try:
                ts = datetime.strptime(f"{date_s} {time_s}", "%d/%m/%y %H:%M:%S")
                return ParsedMessage(sender=sender.strip(), text=text.strip(),
                                     timestamp=ts, platform=Platform.WHATSAPP)
            except ValueError:
                pass

        m = self._DASH.match(line)
        if m:
            date_s, time_s, sender, text = m.groups()
            try:
                ts = datetime.strptime(f"{date_s} {time_s}", "%m/%d/%y %I:%M %p")
                return ParsedMessage(sender=sender.strip(), text=text.strip(),
                                     timestamp=ts, platform=Platform.WHATSAPP)
            except ValueError:
                pass

        return None
