"""P1 config + path helpers. Reads ``config/base.yaml`` (shared) and ``config/data.yaml`` (P1).

Everything P1 builds (ingest, store, eval) resolves paths through here, so a test can point the
whole pipeline at a temporary directory by passing ``root=``.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@lru_cache(maxsize=1)
def base_config() -> dict[str, Any]:
    return load_yaml(CONFIG_DIR / "base.yaml")


@lru_cache(maxsize=1)
def data_config() -> dict[str, Any]:
    return load_yaml(CONFIG_DIR / "data.yaml")


def repo_path(key: str, root: Path | None = None) -> Path:
    """Resolve a ``paths.<key>`` entry from base.yaml (or data.yaml ``paths``) under ``root``."""
    paths = {**base_config().get("paths", {}), **data_config().get("paths", {})}
    if key not in paths:
        raise KeyError(f"unknown path key {key!r}")
    return (root or ROOT) / paths[key]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
