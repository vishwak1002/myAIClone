from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from ..models import ParsedMessage


class BaseParser(ABC):
    @abstractmethod
    def parse(self, path: Path) -> List[ParsedMessage]:
        ...

    @abstractmethod
    def can_parse(self, path: Path) -> bool:
        ...
