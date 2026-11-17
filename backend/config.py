"""P3 config helpers.

``config/base.yaml`` is shared; ``backend.yaml``, ``escalation.yaml``,
``kb_escalation_clauses.yaml``, ``clarify_templates.yaml`` and ``users.yaml`` are owned by P3.
Every loader is cached; tests pass explicit dicts instead of mutating these.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _load(name: str) -> dict[str, Any]:
    return load_yaml(CONFIG_DIR / name)


@lru_cache(maxsize=1)
def base_config() -> dict[str, Any]:
    return _load("base.yaml")


@lru_cache(maxsize=1)
def backend_config() -> dict[str, Any]:
    return _load("backend.yaml")


@lru_cache(maxsize=1)
def escalation_config() -> dict[str, Any]:
    return _load("escalation.yaml")


@lru_cache(maxsize=1)
def clarify_config() -> dict[str, Any]:
    return _load("clarify_templates.yaml")


@lru_cache(maxsize=1)
def kb_clauses() -> dict[str, list[dict[str, Any]]]:
    """``{kb_id: [{text, when}]}``; YAML anchor helpers (keys not starting with ``KB-``) dropped."""
    raw = _load("kb_escalation_clauses.yaml")
    return {k: v or [] for k, v in raw.items() if k.startswith("KB-")}


@lru_cache(maxsize=1)
def users_config() -> dict[str, Any]:
    return _load("users.yaml")


def repo_path(key: str) -> Path:
    """``paths.<key>`` from base.yaml, resolved under the repo root."""
    return ROOT / base_config()["paths"][key]
