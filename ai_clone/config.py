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
