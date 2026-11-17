"""Shared pytest setup for P2 tests: repo root on sys.path, offline by default."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests must pass with GROQ_API_KEY unset (CLAUDE.md rule 6)."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
