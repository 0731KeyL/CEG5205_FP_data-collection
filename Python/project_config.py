"""Shared collection and dataset configuration."""

from __future__ import annotations

import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = BASE_DIR / "collection_config.json"


def load_project_config(path: str | Path | None = None) -> dict:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    gestures = config.get("gestures", [])
    if not gestures or len(gestures) != len(set(gestures)):
        raise ValueError("gestures must be a non-empty unique list")
    required_defaults = {
        "duration_s",
        "trials",
        "countdown_s",
        "rest_s",
    }
    if not required_defaults.issubset(config.get("defaults", {})):
        raise ValueError("defaults is missing one or more required fields")
    layout = config.get("sensor_layout", {})
    if not layout.get("positions") or not layout.get("display"):
        raise ValueError("sensor_layout.positions and display are required")
    return config
