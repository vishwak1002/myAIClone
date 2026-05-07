from __future__ import annotations
import json
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from .base import BaseParser
from ..models import ParsedMessage, Platform


class SlackParser(BaseParser):
    def can_parse(self, path: Path) -> bool:
        return path.suffix == ".zip"

    def parse(self, path: Path) -> List[ParsedMessage]:
        messages: List[ParsedMessage] = []
        with zipfile.ZipFile(path, "r") as zf:
            user_map = self._load_user_map(zf)
            for name in zf.namelist():
                if not name.endswith(".json") or "/" not in name:
                    continue
                with zf.open(name) as f:
                    day_msgs = json.load(f)
                for msg in day_msgs:
                    if msg.get("type") != "message" or "subtype" in msg:
                        continue
                    text = msg.get("text", "").strip()
                    if not text:
                        continue
                    user_id = msg.get("user", "unknown")
                    timestamp = datetime.fromtimestamp(float(msg["ts"]))
                    messages.append(ParsedMessage(
                        sender=user_map.get(user_id, user_id),
                        text=text,
                        timestamp=timestamp,
                        platform=Platform.SLACK,
                    ))
        return sorted(messages, key=lambda m: m.timestamp)

    def _load_user_map(self, zf: zipfile.ZipFile) -> Dict[str, str]:
        if "users.json" not in zf.namelist():
            return {}
        with zf.open("users.json") as f:
            users = json.load(f)
        return {
            u["id"]: u.get("real_name", u.get("name", u["id"]))
            for u in users
        }
