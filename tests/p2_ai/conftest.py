"""P2 test fixtures: demo tickets (P2 reconstructions) and the shared stub stores."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shared.schemas import SanitizedTicket
from shared.stubs import FakeKBStore, FakeTicketStore

FIXTURES = Path(__file__).parent / "fixtures"


def _demos() -> dict[str, SanitizedTicket]:
    raw = json.loads((FIXTURES / "demo_tickets.json").read_text(encoding="utf-8"))
    return {
        t["id"]: SanitizedTicket(ticket_id=t["id"], title=t["title"], description=t["description"])
        for t in raw
    }


DEMOS = _demos()


@pytest.fixture(scope="session")
def demos() -> dict[str, SanitizedTicket]:
    return DEMOS


@pytest.fixture(scope="session")
def kb_store() -> FakeKBStore:
    return FakeKBStore()


@pytest.fixture
def ticket_store() -> FakeTicketStore:
    return FakeTicketStore()
