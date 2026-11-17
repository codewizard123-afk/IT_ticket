"""UI settings: ``config/ui.yaml`` plus environment overrides (README §13)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
UI_CONFIG = ROOT / "config" / "ui.yaml"


def _flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def load_ui_config() -> dict[str, Any]:
    """The raw ``config/ui.yaml`` (cached; it never changes at runtime)."""
    return yaml.safe_load(UI_CONFIG.read_text(encoding="utf-8")) or {}


@dataclass(frozen=True)
class Settings:
    api_base_url: str
    mock_api: bool
    demo_auth: bool
    timeout_s: float
    config: dict[str, Any] = field(default_factory=dict)


def get_settings() -> Settings:
    """Read on every call so tests can flip env vars with ``monkeypatch``."""
    cfg = load_ui_config()
    return Settings(
        api_base_url=os.environ.get("API_BASE_URL") or cfg["api"]["base_url"],
        mock_api=_flag("MOCK_API"),
        demo_auth=_flag("DEMO_AUTH", default=True),
        timeout_s=float(cfg["api"].get("timeout_s", 30)),
        config=cfg,
    )
