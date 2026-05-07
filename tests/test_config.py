from __future__ import annotations
from pathlib import Path

import pytest

from ai_clone.config import Config


def test_load_from_missing_file_returns_defaults(tmp_path):
    cfg = Config.load(tmp_path / "missing.yml")
    assert cfg.openrouter_api_key == ""
    assert isinstance(cfg.db_path, Path)


def test_load_ignores_unknown_keys(tmp_path):
    config_file = tmp_path / "config.yml"
    config_file.write_text("unknown_key: value\n")
    cfg = Config.load(config_file)
    assert cfg.openrouter_api_key == ""


def test_save_then_load_round_trips_path_fields(tmp_path):
    custom_db = tmp_path / "custom" / "memory.db"
    cfg = Config(db_path=custom_db)
    save_path = tmp_path / "config.yml"
    cfg.save(save_path)
    loaded = Config.load(save_path)
    assert isinstance(loaded.db_path, Path)
    assert loaded.db_path == custom_db


def test_save_then_load_round_trips_string_fields(tmp_path):
    cfg = Config(openrouter_api_key="test-key")
    save_path = tmp_path / "config.yml"
    cfg.save(save_path)
    loaded = Config.load(save_path)
    assert loaded.openrouter_api_key == "test-key"
