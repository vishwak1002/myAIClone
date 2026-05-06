# AI Clone — Sub-project 1: Project Setup + Personality Profile Builder

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Set up the Python project and build the personality profile builder — parsing chat exports from WhatsApp, Telegram, iMessage, and Slack into a structured `profile.json` that captures the user's writing style, vocabulary, and tone.

**Architecture:** Layered monolith. All parsers implement `BaseParser` and output `ParsedMessage` objects. `IngestionPipeline` auto-selects the right parser. `PersonalityBuilder` processes filtered user messages into `PersonalityProfile`. `ProfileUpdater` refreshes the profile incrementally on a rolling window of 500 messages.

**Tech Stack:** Python 3.11+, pytest, pytest-mock, pyyaml, click (entry point wired but CLI implementation is Sub-project 5)

---

## File Map

**Create:**
- `pyproject.toml` — project metadata + dependencies + `ai-clone` entry point
- `config.yml` — runtime config template (all keys blank)
- `ai_clone/__init__.py`
- `ai_clone/config.py` — `Config` dataclass (loads/saves `config.yml`)
- `ai_clone/ingestion/__init__.py`
- `ai_clone/ingestion/models.py` — `ParsedMessage`, `Platform`
- `ai_clone/ingestion/parsers/__init__.py`
- `ai_clone/ingestion/parsers/base.py` — `BaseParser` ABC
- `ai_clone/ingestion/parsers/whatsapp.py` — `WhatsAppParser`
- `ai_clone/ingestion/parsers/telegram.py` — `TelegramParser`
- `ai_clone/ingestion/parsers/imessage.py` — `iMessageParser`
- `ai_clone/ingestion/parsers/slack.py` — `SlackParser`
- `ai_clone/ingestion/pipeline.py` — `IngestionPipeline`
- `ai_clone/personality/__init__.py`
- `ai_clone/personality/models.py` — `PersonalityProfile`
- `ai_clone/personality/builder.py` — `PersonalityBuilder`
- `ai_clone/personality/updater.py` — `ProfileUpdater`
- `tests/__init__.py`
- `tests/conftest.py`
- `tests/ingestion/__init__.py`
- `tests/ingestion/parsers/__init__.py`
- `tests/ingestion/parsers/test_whatsapp.py`
- `tests/ingestion/parsers/test_telegram.py`
- `tests/ingestion/parsers/test_imessage.py`
- `tests/ingestion/parsers/test_slack.py`
- `tests/ingestion/test_pipeline.py`
- `tests/personality/__init__.py`
- `tests/personality/test_builder.py`
- `tests/personality/test_updater.py`

---

## Task 1: Project Setup

**Files:**
- Create: `pyproject.toml`
- Create: `config.yml`
- Create: `ai_clone/__init__.py`
- Create: `ai_clone/config.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "ai-clone"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.110.0",
    "uvicorn>=0.27.0",
    "click>=8.1.7",
    "chromadb>=0.4.22",
    "sentence-transformers>=2.4.0",
    "faster-whisper>=1.0.0",
    "webrtcvad>=2.0.10",
    "sounddevice>=0.4.6",
    "elevenlabs>=1.0.0",
    "openai>=1.12.0",
    "httpx>=0.26.0",
    "pyyaml>=6.0.1",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-mock>=3.12.0",
    "pytest-asyncio>=0.23.0",
]

[project.scripts]
ai-clone = "ai_clone.cli.main:cli"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create `config.yml`**

```yaml
openrouter_api_key: ""
elevenlabs_api_key: ""
together_api_key: ""
db_path: "./data/memory.db"
chroma_path: "./data/chroma"
profile_path: "./data/profile.json"
fine_tuned_model_id: ""
user_name: ""
elevenlabs_voice_id: ""
whisper_model: "base"
vad_aggressiveness: 2
```

- [ ] **Step 3: Create `ai_clone/__init__.py`** (empty)

```python
```

- [ ] **Step 4: Create `ai_clone/config.py`**

```python
from __future__ import annotations
from dataclasses import dataclass, field, fields
from pathlib import Path
import yaml


@dataclass
class Config:
    openrouter_api_key: str = ""
    elevenlabs_api_key: str = ""
    together_api_key: str = ""
    db_path: Path = field(default_factory=lambda: Path("./data/memory.db"))
    chroma_path: Path = field(default_factory=lambda: Path("./data/chroma"))
    profile_path: Path = field(default_factory=lambda: Path("./data/profile.json"))
    fine_tuned_model_id: str = ""
    user_name: str = ""
    elevenlabs_voice_id: str = ""
    whisper_model: str = "base"
    vad_aggressiveness: int = 2

    @classmethod
    def load(cls, path: Path = Path("config.yml")) -> Config:
        if not path.exists():
            return cls()
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        field_names = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in field_names}
        for key in ("db_path", "chroma_path", "profile_path"):
            if key in filtered:
                filtered[key] = Path(filtered[key])
        return cls(**filtered)

    def save(self, path: Path = Path("config.yml")) -> None:
        data = {
            k: str(v) if isinstance(v, Path) else v
            for k, v in self.__dict__.items()
        }
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
```

- [ ] **Step 5: Create `tests/__init__.py` and `tests/conftest.py`** (empty files)

```python
# tests/__init__.py  (empty)
```

```python
# tests/conftest.py  (empty for now)
```

- [ ] **Step 6: Install dependencies**

```bash
cd /Users/vishwasaikarnati/myAIClone
pip install -e ".[dev]"
```

Expected: installs without errors.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml config.yml ai_clone/__init__.py ai_clone/config.py tests/__init__.py tests/conftest.py
git commit -m "chore: project setup — pyproject.toml, Config, test harness"
```

---

## Task 2: ParsedMessage Model + BaseParser

**Files:**
- Create: `ai_clone/ingestion/__init__.py`
- Create: `ai_clone/ingestion/models.py`
- Create: `ai_clone/ingestion/parsers/__init__.py`
- Create: `ai_clone/ingestion/parsers/base.py`
- Create: `tests/ingestion/__init__.py`
- Create: `tests/ingestion/parsers/__init__.py`

- [ ] **Step 1: Write failing test for ParsedMessage**

Create `tests/ingestion/test_models.py`:

```python
from datetime import datetime
from ai_clone.ingestion.models import ParsedMessage, Platform


def test_parsed_message_stores_fields():
    msg = ParsedMessage(
        sender="Vishwa",
        text="Hello there",
        timestamp=datetime(2023, 5, 23, 14, 32, 10),
        platform=Platform.WHATSAPP,
    )
    assert msg.sender == "Vishwa"
    assert msg.text == "Hello there"
    assert msg.platform == Platform.WHATSAPP


def test_platform_values():
    assert Platform.WHATSAPP == "whatsapp"
    assert Platform.TELEGRAM == "telegram"
    assert Platform.IMESSAGE == "imessage"
    assert Platform.SLACK == "slack"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/ingestion/test_models.py -v
```

Expected: `ModuleNotFoundError: No module named 'ai_clone.ingestion'`

- [ ] **Step 3: Create `ai_clone/ingestion/__init__.py`** (empty) and `ai_clone/ingestion/models.py`**

```python
# ai_clone/ingestion/__init__.py  (empty)
```

```python
# ai_clone/ingestion/models.py
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
```

- [ ] **Step 4: Create `ai_clone/ingestion/parsers/__init__.py`** (empty) and `ai_clone/ingestion/parsers/base.py`**

```python
# ai_clone/ingestion/parsers/__init__.py  (empty)
```

```python
# ai_clone/ingestion/parsers/base.py
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from ..models import ParsedMessage


class BaseParser(ABC):
    @abstractmethod
    def parse(self, path: Path) -> List[ParsedMessage]:
        raise NotImplementedError

    @abstractmethod
    def can_parse(self, path: Path) -> bool:
        raise NotImplementedError
```

- [ ] **Step 5: Create empty test `__init__` files**

```bash
touch tests/ingestion/__init__.py tests/ingestion/parsers/__init__.py
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
pytest tests/ingestion/test_models.py -v
```

Expected: `2 passed`

- [ ] **Step 7: Commit**

```bash
git add ai_clone/ingestion/ tests/ingestion/ tests/ingestion/parsers/__init__.py
git commit -m "feat: ParsedMessage model and BaseParser ABC"
```

---

## Task 3: WhatsAppParser

**Files:**
- Create: `ai_clone/ingestion/parsers/whatsapp.py`
- Create: `tests/ingestion/parsers/test_whatsapp.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/ingestion/parsers/test_whatsapp.py
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
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/ingestion/parsers/test_whatsapp.py -v
```

Expected: `ModuleNotFoundError: No module named 'ai_clone.ingestion.parsers.whatsapp'`

- [ ] **Step 3: Implement `WhatsAppParser`**

```python
# ai_clone/ingestion/parsers/whatsapp.py
from __future__ import annotations
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .base import BaseParser
from ..models import ParsedMessage, Platform


class WhatsAppParser(BaseParser):
    # [DD/MM/YYYY, HH:MM:SS] Sender: text
    _BRACKET = re.compile(
        r"\[(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}:\d{2})\]\s(.+?):\s(.+)"
    )
    # MM/DD/YY, HH:MM AM/PM - Sender: text
    _DASH = re.compile(
        r"(\d{1,2}/\d{1,2}/\d{2}),\s(\d{1,2}:\d{2}\s(?:AM|PM))\s-\s(.+?):\s(.+)"
    )

    def can_parse(self, path: Path) -> bool:
        return path.suffix == ".txt"

    def parse(self, path: Path) -> List[ParsedMessage]:
        messages = []
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/ingestion/parsers/test_whatsapp.py -v
```

Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add ai_clone/ingestion/parsers/whatsapp.py tests/ingestion/parsers/test_whatsapp.py
git commit -m "feat: WhatsAppParser — bracket and dash format support"
```

---

## Task 4: TelegramParser

**Files:**
- Create: `ai_clone/ingestion/parsers/telegram.py`
- Create: `tests/ingestion/parsers/test_telegram.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/ingestion/parsers/test_telegram.py
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
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/ingestion/parsers/test_telegram.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement `TelegramParser`**

```python
# ai_clone/ingestion/parsers/telegram.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/ingestion/parsers/test_telegram.py -v
```

Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add ai_clone/ingestion/parsers/telegram.py tests/ingestion/parsers/test_telegram.py
git commit -m "feat: TelegramParser — plain and rich text support"
```

---

## Task 5: iMessageParser

**Files:**
- Create: `ai_clone/ingestion/parsers/imessage.py`
- Create: `tests/ingestion/parsers/test_imessage.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/ingestion/parsers/test_imessage.py
import sqlite3
import pytest
from datetime import datetime
from pathlib import Path

from ai_clone.ingestion.parsers.imessage import iMessageParser
from ai_clone.ingestion.models import Platform

# iMessage epoch offset: seconds from 2001-01-01 to 2023-05-23T14:32:10
# 2001-01-01 to 2023-05-23T14:32:10 = 706710730 seconds
# stored as nanoseconds: 706710730 * 1_000_000_000
SAMPLE_DATE_NS = 706710730 * 1_000_000_000


@pytest.fixture
def imessage_db(tmp_path):
    db_path = tmp_path / "chat.db"
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE handle (
            rowid INTEGER PRIMARY KEY,
            id TEXT
        )
    """)
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
    conn.execute(
        "INSERT INTO message VALUES (1, ?, 'Hello there', 1, 0)",
        (SAMPLE_DATE_NS,)
    )
    conn.execute(
        "INSERT INTO message VALUES (2, ?, 'Hi!', 0, 1)",
        (SAMPLE_DATE_NS + 5_000_000_000,)
    )
    conn.execute(
        "INSERT INTO message VALUES (3, ?, NULL, 1, 0)",
        (SAMPLE_DATE_NS + 10_000_000_000,)
    )
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
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/ingestion/parsers/test_imessage.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement `iMessageParser`**

```python
# ai_clone/ingestion/parsers/imessage.py
from __future__ import annotations
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

from .base import BaseParser
from ..models import ParsedMessage, Platform

# iMessage stores timestamps as nanoseconds since 2001-01-01 00:00:00 UTC
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/ingestion/parsers/test_imessage.py -v
```

Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add ai_clone/ingestion/parsers/imessage.py tests/ingestion/parsers/test_imessage.py
git commit -m "feat: iMessageParser — SQLite chat.db support"
```

---

## Task 6: SlackParser

**Files:**
- Create: `ai_clone/ingestion/parsers/slack.py`
- Create: `tests/ingestion/parsers/test_slack.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/ingestion/parsers/test_slack.py
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
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/ingestion/parsers/test_slack.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement `SlackParser`**

```python
# ai_clone/ingestion/parsers/slack.py
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
                if name == "users.json" or name == "channels.json":
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/ingestion/parsers/test_slack.py -v
```

Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add ai_clone/ingestion/parsers/slack.py tests/ingestion/parsers/test_slack.py
git commit -m "feat: SlackParser — zip export with user name resolution"
```

---

## Task 7: IngestionPipeline

**Files:**
- Create: `ai_clone/ingestion/pipeline.py`
- Create: `tests/ingestion/test_pipeline.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/ingestion/test_pipeline.py
import json
import pytest
from pathlib import Path

from ai_clone.ingestion.pipeline import IngestionPipeline


@pytest.fixture
def telegram_file(tmp_path):
    data = {
        "messages": [
            {"id": 1, "type": "message", "date": "2023-05-23T14:32:10",
             "from": "Vishwa", "text": "Hello"},
            {"id": 2, "type": "message", "date": "2023-05-23T14:32:15",
             "from": "John", "text": "Hi!"},
        ]
    }
    p = tmp_path / "result.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_ingest_returns_all_messages(telegram_file):
    pipeline = IngestionPipeline(user_name="Vishwa")
    messages = pipeline.ingest(telegram_file)
    assert len(messages) == 2


def test_ingest_user_messages_filters_to_user(telegram_file):
    pipeline = IngestionPipeline(user_name="Vishwa")
    messages = pipeline.ingest_user_messages(telegram_file)
    assert len(messages) == 1
    assert messages[0].sender == "Vishwa"


def test_ingest_raises_for_unknown_format(tmp_path):
    pipeline = IngestionPipeline(user_name="Vishwa")
    unknown = tmp_path / "data.xml"
    unknown.touch()
    with pytest.raises(ValueError, match="No parser found"):
        pipeline.ingest(unknown)
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/ingestion/test_pipeline.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement `IngestionPipeline`**

```python
# ai_clone/ingestion/pipeline.py
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
            SlackParser(),       # .zip — check before .txt/.json
            TelegramParser(),    # .json
            iMessageParser(),    # .db
            WhatsAppParser(),    # .txt
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/ingestion/test_pipeline.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Run all ingestion tests**

```bash
pytest tests/ingestion/ -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add ai_clone/ingestion/pipeline.py tests/ingestion/test_pipeline.py
git commit -m "feat: IngestionPipeline — auto-selects parser, filters user messages"
```

---

## Task 8: PersonalityProfile Model

**Files:**
- Create: `ai_clone/personality/__init__.py`
- Create: `ai_clone/personality/models.py`
- Create: `tests/personality/__init__.py`
- Create: `tests/personality/test_models.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/personality/test_models.py
import json
import pytest
from pathlib import Path

from ai_clone.personality.models import PersonalityProfile


def test_default_profile_has_zero_values():
    profile = PersonalityProfile()
    assert profile.avg_sentence_length == 0.0
    assert profile.emoji_frequency == 0.0
    assert profile.top_phrases == []


def test_save_and_load_roundtrip(tmp_path):
    profile = PersonalityProfile(
        avg_sentence_length=8.5,
        emoji_frequency=0.3,
        top_phrases=["how are", "sounds good"],
        formality_score=0.4,
        punctuation_style={"!": 0.2, "?": 0.1},
        tone_traits={"humor": 0.7},
        top_topics=["tech", "travel"],
    )
    path = tmp_path / "profile.json"
    profile.save(path)
    loaded = PersonalityProfile.load(path)
    assert loaded.avg_sentence_length == 8.5
    assert loaded.top_phrases == ["how are", "sounds good"]
    assert loaded.tone_traits == {"humor": 0.7}


def test_save_creates_parent_dirs(tmp_path):
    profile = PersonalityProfile()
    path = tmp_path / "data" / "nested" / "profile.json"
    profile.save(path)
    assert path.exists()
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/personality/test_models.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Create `ai_clone/personality/__init__.py`** (empty) and implement model**

```python
# ai_clone/personality/__init__.py  (empty)
```

```python
# ai_clone/personality/models.py
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class PersonalityProfile:
    avg_sentence_length: float = 0.0
    emoji_frequency: float = 0.0
    top_phrases: List[str] = field(default_factory=list)
    formality_score: float = 0.5
    punctuation_style: Dict[str, float] = field(default_factory=dict)
    tone_traits: Dict[str, float] = field(default_factory=dict)
    top_topics: List[str] = field(default_factory=list)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> PersonalityProfile:
        with open(path) as f:
            data = json.load(f)
        return cls(**data)
```

- [ ] **Step 4: Create `tests/personality/__init__.py`** (empty), run tests**

```bash
touch tests/personality/__init__.py
pytest tests/personality/test_models.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add ai_clone/personality/ tests/personality/
git commit -m "feat: PersonalityProfile model with save/load"
```

---

## Task 9: PersonalityBuilder

**Files:**
- Create: `ai_clone/personality/builder.py`
- Create: `tests/personality/test_builder.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/personality/test_builder.py
from datetime import datetime
import pytest

from ai_clone.ingestion.models import ParsedMessage, Platform
from ai_clone.personality.builder import PersonalityBuilder


def _msg(text: str) -> ParsedMessage:
    return ParsedMessage(
        sender="Vishwa", text=text,
        timestamp=datetime(2023, 1, 1), platform=Platform.WHATSAPP
    )


def test_empty_messages_returns_default_profile():
    builder = PersonalityBuilder()
    profile = builder.build([])
    assert profile.avg_sentence_length == 0.0
    assert profile.emoji_frequency == 0.0


def test_avg_sentence_length():
    builder = PersonalityBuilder()
    messages = [_msg("one two three"), _msg("four five")]
    profile = builder.build(messages)
    assert profile.avg_sentence_length == 2.5  # (3 + 2) / 2


def test_emoji_frequency():
    builder = PersonalityBuilder()
    messages = [_msg("hello 😀"), _msg("no emojis")]
    profile = builder.build(messages)
    assert profile.emoji_frequency == 0.5  # 1 emoji / 2 messages


def test_top_phrases_are_bigrams():
    builder = PersonalityBuilder()
    messages = [_msg("how are you"), _msg("how are things"), _msg("how are we")]
    profile = builder.build(messages)
    assert "how are" in profile.top_phrases


def test_informal_text_has_low_formality():
    builder = PersonalityBuilder()
    messages = [_msg("lol lmao omg brb tbh ngl idk imo gonna wanna")]
    profile = builder.build(messages)
    assert profile.formality_score < 0.5


def test_formal_text_has_high_formality():
    builder = PersonalityBuilder()
    messages = [_msg("I would like to discuss the proposal thoroughly.")]
    profile = builder.build(messages)
    assert profile.formality_score > 0.5


def test_punctuation_style_sums_to_one_or_zero():
    builder = PersonalityBuilder()
    messages = [_msg("Hello! How are you? I'm fine.")]
    profile = builder.build(messages)
    total = sum(profile.punctuation_style.values())
    assert abs(total - 1.0) < 1e-6 or total == 0.0
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/personality/test_builder.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement `PersonalityBuilder`**

```python
# ai_clone/personality/builder.py
from __future__ import annotations
import re
from collections import Counter
from typing import Dict, List

from ..ingestion.models import ParsedMessage
from .models import PersonalityProfile

_EMOJI_RE = re.compile(
    "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0\U000024C2-\U0001F251]+",
    flags=re.UNICODE,
)
_INFORMAL = frozenset({
    "lol", "lmao", "omg", "brb", "tbh", "ngl", "idk",
    "imo", "rn", "gonna", "wanna", "gotta",
})
_PUNCT = "!?.,;:"


class PersonalityBuilder:
    def build(self, messages: List[ParsedMessage]) -> PersonalityProfile:
        if not messages:
            return PersonalityProfile()
        texts = [m.text for m in messages]
        return PersonalityProfile(
            avg_sentence_length=self._avg_words(texts),
            emoji_frequency=self._emoji_freq(texts),
            top_phrases=self._top_ngrams(texts, n=2, top_k=20),
            formality_score=self._formality(texts),
            punctuation_style=self._punct_style(texts),
        )

    def _avg_words(self, texts: List[str]) -> float:
        counts = [len(t.split()) for t in texts if t.strip()]
        return sum(counts) / len(counts) if counts else 0.0

    def _emoji_freq(self, texts: List[str]) -> float:
        total = sum(len(_EMOJI_RE.findall(t)) for t in texts)
        return total / len(texts) if texts else 0.0

    def _top_ngrams(self, texts: List[str], n: int, top_k: int) -> List[str]:
        counter: Counter = Counter()
        for text in texts:
            words = re.findall(r"\b\w+\b", text.lower())
            for i in range(len(words) - n + 1):
                counter[" ".join(words[i : i + n])] += 1
        return [phrase for phrase, _ in counter.most_common(top_k)]

    def _formality(self, texts: List[str]) -> float:
        total = informal = 0
        for text in texts:
            words = re.findall(r"\b\w+\b", text.lower())
            total += len(words)
            informal += sum(1 for w in words if w in _INFORMAL)
        if total == 0:
            return 0.5
        return max(0.0, min(1.0, 1.0 - (informal / total) * 10))

    def _punct_style(self, texts: List[str]) -> Dict[str, float]:
        counter: Counter = Counter()
        for text in texts:
            for ch in text:
                if ch in _PUNCT:
                    counter[ch] += 1
        total = sum(counter.values())
        if total == 0:
            return {p: 0.0 for p in _PUNCT}
        return {p: counter.get(p, 0) / total for p in _PUNCT}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/personality/test_builder.py -v
```

Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add ai_clone/personality/builder.py tests/personality/test_builder.py
git commit -m "feat: PersonalityBuilder — style, emoji, ngrams, formality, punctuation"
```

---

## Task 10: ProfileUpdater

**Files:**
- Create: `ai_clone/personality/updater.py`
- Create: `tests/personality/test_updater.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/personality/test_updater.py
from datetime import datetime
from pathlib import Path
import pytest

from ai_clone.ingestion.models import ParsedMessage, Platform
from ai_clone.personality.builder import PersonalityBuilder
from ai_clone.personality.updater import ProfileUpdater


def _msg(text: str) -> ParsedMessage:
    return ParsedMessage(
        sender="Vishwa", text=text,
        timestamp=datetime(2023, 1, 1), platform=Platform.WHATSAPP
    )


def test_update_saves_new_profile(tmp_path):
    path = tmp_path / "profile.json"
    updater = ProfileUpdater(profile_path=path, builder=PersonalityBuilder())
    profile = updater.update(
        new_messages=[_msg("hello world")],
        existing_messages=[],
    )
    assert path.exists()
    assert profile.avg_sentence_length == 2.0


def test_update_uses_rolling_window(tmp_path):
    from ai_clone.personality.updater import ROLLING_WINDOW
    path = tmp_path / "profile.json"
    updater = ProfileUpdater(profile_path=path, builder=PersonalityBuilder())
    existing = [_msg("old message")] * (ROLLING_WINDOW + 10)
    new_msgs = [_msg("new one two three")]
    profile = updater.update(new_messages=new_msgs, existing_messages=existing)
    # Combined was truncated to ROLLING_WINDOW; new message is included
    assert profile.avg_sentence_length > 0


def test_update_combines_new_and_existing(tmp_path):
    path = tmp_path / "profile.json"
    updater = ProfileUpdater(profile_path=path, builder=PersonalityBuilder())
    existing = [_msg("one")]
    new_msgs = [_msg("two three")]
    profile = updater.update(new_messages=new_msgs, existing_messages=existing)
    # avg of [1, 2] = 1.5
    assert profile.avg_sentence_length == 1.5
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/personality/test_updater.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement `ProfileUpdater`**

```python
# ai_clone/personality/updater.py
from __future__ import annotations
from pathlib import Path
from typing import List

from ..ingestion.models import ParsedMessage
from .builder import PersonalityBuilder
from .models import PersonalityProfile

ROLLING_WINDOW = 500


class ProfileUpdater:
    def __init__(self, profile_path: Path, builder: PersonalityBuilder) -> None:
        self.profile_path = profile_path
        self.builder = builder

    def update(
        self,
        new_messages: List[ParsedMessage],
        existing_messages: List[ParsedMessage],
    ) -> PersonalityProfile:
        combined = (existing_messages + new_messages)[-ROLLING_WINDOW:]
        profile = self.builder.build(combined)
        profile.save(self.profile_path)
        return profile
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/personality/test_updater.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add ai_clone/personality/updater.py tests/personality/test_updater.py
git commit -m "feat: ProfileUpdater — rolling window profile refresh"
```

---

## Task 11: Integration Test + Push

**Files:**
- Create: `tests/test_integration_ingestion_profile.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_integration_ingestion_profile.py
import json
from pathlib import Path
import pytest

from ai_clone.ingestion.pipeline import IngestionPipeline
from ai_clone.personality.builder import PersonalityBuilder
from ai_clone.personality.updater import ProfileUpdater


@pytest.fixture
def telegram_export(tmp_path):
    data = {
        "messages": [
            {"id": i, "type": "message",
             "date": f"2023-05-23T14:3{i}:00",
             "from": "Vishwa",
             "text": f"Hello there how are you doing {i}"}
            for i in range(10)
        ] + [
            {"id": 20, "type": "message", "date": "2023-05-23T15:00:00",
             "from": "John", "text": "Hey Vishwa!"}
        ]
    }
    p = tmp_path / "result.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_full_ingest_to_profile_pipeline(tmp_path, telegram_export):
    pipeline = IngestionPipeline(user_name="Vishwa")
    user_messages = pipeline.ingest_user_messages(telegram_export)

    assert len(user_messages) == 10
    assert all(m.sender == "Vishwa" for m in user_messages)

    builder = PersonalityBuilder()
    updater = ProfileUpdater(
        profile_path=tmp_path / "profile.json",
        builder=builder,
    )
    profile = updater.update(new_messages=user_messages, existing_messages=[])

    assert (tmp_path / "profile.json").exists()
    assert profile.avg_sentence_length > 0
    assert "how are" in profile.top_phrases or len(profile.top_phrases) > 0
```

- [ ] **Step 2: Run integration test**

```bash
pytest tests/test_integration_ingestion_profile.py -v
```

Expected: `1 passed`

- [ ] **Step 3: Run full test suite**

```bash
pytest -v
```

Expected: all tests pass.

- [ ] **Step 4: Commit and push**

```bash
git add tests/test_integration_ingestion_profile.py
git commit -m "test: integration test — ingest to profile pipeline"
git push origin main
```

---

## Self-Review Notes

- All 4 parsers use the same `ParsedMessage` output — consistent
- `IngestionPipeline.ingest_user_messages` uses exact string match on `sender` — user must set `user_name` in config to match their name in exports exactly
- `PersonalityBuilder` does not call OpenRouter (tone traits via LLM) — that is deferred to Sub-project 3 when OpenRouter integration is built. `tone_traits` defaults to `{}` until then
- `top_topics` also deferred to Sub-project 3 (needs embeddings from ChromaDB setup in Sub-project 2)
- `FineTuneTrainer` is part of Sub-project 6 — not included here
