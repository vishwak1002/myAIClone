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
