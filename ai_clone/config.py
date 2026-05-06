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
    def load(cls, path: Path | None = None) -> "Config":
        if path is None:
            path = Path("config.yml")
        if not path.exists():
            return cls()
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        field_names = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in field_names}
        path_fields = {f.name for f in fields(cls) if "Path" in str(f.type)}
        for key in path_fields:
            if key in filtered and filtered[key] is not None:
                filtered[key] = Path(filtered[key])
        return cls(**filtered)

    def save(self, path: Path | None = None) -> None:
        if path is None:
            path = Path("config.yml")
        data = {
            f.name: str(getattr(self, f.name)) if isinstance(getattr(self, f.name), Path) else getattr(self, f.name)
            for f in fields(self)
        }
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
