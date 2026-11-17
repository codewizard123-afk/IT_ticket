"""P3 config helpers: ``config/base.yaml`` (shared) merged with ``config/backend.yaml`` (P3)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"


def _load(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


@lru_cache(maxsize=1)
def base_config() -> dict[str, Any]:
    return _load("base.yaml")


@lru_cache(maxsize=1)
def backend_config() -> dict[str, Any]:
    return _load("backend.yaml")
