from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class Platform(str, Enum):
    WHATSAPP = "whatsapp"
    TELEGRAM = "telegram"
    IMESSAGE = "imessage"
    SLACK = "slack"


@dataclass
class ParsedMessage:
    sender: str
    text: str
    timestamp: datetime
    platform: Platform
