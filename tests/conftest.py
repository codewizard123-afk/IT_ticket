"""Repo-wide pytest fixtures. Owner folders may add their own conftest.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "shared" / "contract" / "analysis_output.schema.json"
GOLDEN_DIR = ROOT / "shared" / "fixtures" / "golden"


@pytest.fixture(autouse=True)
def _no_groq_key(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    """Tests run offline unless marked ``groq`` (CLAUDE.md rule 6)."""
    if request.node.get_closest_marker("groq") is None:
        monkeypatch.delenv("GROQ_API_KEY", raising=False)


@pytest.fixture(scope="session")
def contract_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def golden() -> dict[str, dict]:
    return {
        p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(GOLDEN_DIR.glob("*.json"))
    }
