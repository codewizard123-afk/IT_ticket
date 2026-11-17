"""P4 fixtures: a mock backend with a controllable clock and clients for each role."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from ui.client import ApiClient, mock_transport
from ui.mock_backend import MockBackend

ROOT = Path(__file__).resolve().parents[2]
DEMO_TICKETS = json.loads((ROOT / "ui" / "mock_data" / "demo_tickets.json").read_text("utf-8"))


class FakeClock:
    def __init__(self, start: float = 1_790_000_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def tick(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def backend(clock: FakeClock) -> MockBackend:
    return MockBackend(clock=clock)


@pytest.fixture
def make_client(backend: MockBackend) -> Callable[..., ApiClient]:
    def make(role: str | None = "l2", group: str | None = None) -> ApiClient:
        return ApiClient(mock_transport(backend), demo_role=role, demo_group=group, mock=True)

    return make


@pytest.fixture
def demo_tickets() -> dict[str, dict]:
    return {t["id"]: t for t in DEMO_TICKETS}
