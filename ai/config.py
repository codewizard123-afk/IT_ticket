"""Config loading for the AI engine. Thresholds live in `config/*.yaml`, never in code."""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"


def load_yaml(name: str, config_dir: Path | None = None) -> dict[str, Any]:
    """Load `config/<name>` (returns `{}` if the file does not exist)."""
    path = (config_dir or CONFIG_DIR) / name
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a mapping at the top level")
    return data


@cache
def ai_config() -> dict[str, Any]:
    return load_yaml("ai.yaml")


@cache
def signals_config() -> dict[str, Any]:
    return load_yaml("signals.yaml")


@cache
def rules_config() -> dict[str, Any]:
    return load_yaml("rules.yaml")


@cache
def taxonomy_map() -> dict[str, Any]:
    """P1-owned `config/taxonomy_map.yaml` (architecture §3.3); `{}` until P1 lands it."""
    return load_yaml("taxonomy_map.yaml")
