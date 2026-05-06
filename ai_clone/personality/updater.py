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
