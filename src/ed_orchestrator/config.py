"""Central configuration loader."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml


@lru_cache(maxsize=1)
def get_config_path() -> Path:
    candidates = [
        Path(__file__).resolve().parents[2] / "simulation" / "config" / "ed_params.yaml",
        Path(__file__).resolve().parents[1] / "config" / "ed_params.yaml",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("ed_params.yaml not found")


@lru_cache(maxsize=1)
def load_config() -> dict:
    with open(get_config_path(), encoding="utf-8") as f:
        return yaml.safe_load(f)


CONFIG = load_config()
