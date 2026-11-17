"""P3 test helpers: demo tickets, golden engine results and gate inputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from backend.decision import GateInput
from shared.schemas import EngineResult
from shared.stubs import FakeKBStore, engine_result_from_golden, load_golden

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = ROOT / "data" / "samples" / "sample_tickets.json"
DEMO_IDS = [f"DEMO-00{i}" for i in range(1, 8)]

# architecture §16: per-issue decision, escalation group/tier, and the two contract booleans
EXPECTED: dict[str, dict[str, Any]] = {
    "DEMO-001": {"issues": ["recommend"], "esc": False, "approval": True},
    "DEMO-002": {
        "issues": ["escalate"],
        "esc": True,
        "approval": True,
        "group": "DBA",
        "tier": "L3",
    },
    "DEMO-003": {"issues": ["recommend"], "esc": False, "approval": False},
    "DEMO-004": {"issues": ["recommend"], "esc": False, "approval": True},
    "DEMO-005": {
        "issues": ["recommend", "escalate"],
        "esc": True,
        "approval": True,
        "group": "EUC-HW",
        "tier": "L2",
    },
    "DEMO-006": {
        "issues": ["escalate"],
        "esc": True,
        "approval": True,
        "group": "NET-OPS",
        "tier": "L2",
    },
    "DEMO-007": {"issues": ["recommend"], "esc": False, "approval": True},
}


def sample_tickets() -> dict[str, dict[str, Any]]:
    return {t["id"]: t for t in json.loads(SAMPLES.read_text(encoding="utf-8"))}


@pytest.fixture(scope="session")
def demo_tickets() -> dict[str, dict[str, Any]]:
    return sample_tickets()


@pytest.fixture(scope="session")
def fake_kb() -> FakeKBStore:
    return FakeKBStore()


def golden_engine_result(ticket_id: str, kb: FakeKBStore | None = None) -> EngineResult:
    golden = load_golden(ticket_id)
    assert golden is not None
    return engine_result_from_golden(golden, kb or FakeKBStore())


def gate_for(ticket: dict[str, Any], er: EngineResult, clarification_round: int = 0) -> GateInput:
    return GateInput(
        signals=er.signals,
        text=f"{ticket['title']}\n{ticket['description']}",
        clarification_round=clarification_round,
    )


# --- API fixtures -------------------------------------------------------------------------------
def make_services(tmp_path: Path, **overrides: Any):  # noqa: ANN201 - backend.services.Services
    """Stub stores/engine/redact, a temp app.db and demo auth."""
    from backend.services import build_services
    from backend.settings import Settings

    settings = Settings(
        demo_auth=True,
        app_secret="test-secret-not-for-production-0123456789",
        use_stub_stores=True,
        use_stub_engine=True,
        use_stub_redact=True,
        db_path=tmp_path / "app.db",
        kb_governed_path=tmp_path / "articles_governed.json",
    )
    services = build_services(settings)
    for key, value in overrides.items():
        setattr(services, key, value)
    return services


def make_client(services: Any):  # noqa: ANN201 - fastapi.testclient.TestClient
    from fastapi.testclient import TestClient

    from backend.main import create_app

    return TestClient(create_app(services))


def role(r: str, group: str | None = None) -> dict[str, str]:
    h = {"X-Demo-Role": r}
    if group:
        h["X-Demo-Group"] = group
    return h


@pytest.fixture
def services(tmp_path: Path):  # noqa: ANN201
    return make_services(tmp_path)


@pytest.fixture
def client(services: Any):  # noqa: ANN201
    with make_client(services) as c:
        yield c
