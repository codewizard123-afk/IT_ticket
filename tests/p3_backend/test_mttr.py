"""P3-T7: events and MTTR metrics (architecture §8.10).

Done when: TTR, TTD and MTTR can be grouped by category / priority / mode / decision.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from tests.p3_backend.conftest import ROOT, make_client, make_services, role


class ManualClock:
    def __init__(self) -> None:
        self.t = datetime(2026, 11, 17, 9, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.t

    def at(self, seconds: float) -> None:
        self.t = datetime(2026, 11, 17, 9, 0, tzinfo=UTC) + timedelta(seconds=seconds)


@pytest.fixture
def clock() -> ManualClock:
    return ManualClock()


@pytest.fixture
def timed(tmp_path: Path, clock: ManualClock):  # noqa: ANN201
    services = make_services(tmp_path, clock=clock)
    with make_client(services) as c:
        yield c


def _event(c: Any, ticket_id: str, aid: str | None, event: str, mode: str = "assisted", h=None):  # noqa: ANN001, ANN202
    body = {"ticket_id": ticket_id, "analysis_id": aid, "event": event, "mode": mode}
    return c.post("/api/v1/events", json=body, headers=h or role("l1"))


def _assisted(
    c: Any,
    clock: ManualClock,
    ticket: dict[str, Any],
    t0: float,
    view: float,
    decide: float,
    final: str = "resolved",
) -> str:  # noqa: E501
    clock.at(t0)
    aid = c.post("/api/v1/analyze", json=ticket, headers=role("l1")).json()["details"][
        "analysis_id"
    ]
    clock.at(view)
    assert _event(c, ticket["id"], aid, "ui.viewed").json() == {"ok": True}
    clock.at(decide)
    c.post(
        "/api/v1/decisions",
        json={"analysis_id": aid, "final_decision": final, "mode": "assisted"},
        headers=role("l1"),
    )
    return aid


def _manual(c: Any, clock: ManualClock, ticket_id: str, sid: str, start: float, end: float) -> None:
    clock.at(start)
    assert _event(c, ticket_id, sid, "manual.started", "manual").status_code == 200
    clock.at(end)
    _event(c, ticket_id, sid, "manual.decision", "manual")
    c.post(
        "/api/v1/decisions",
        json={
            "analysis_id": sid,
            "final_decision": "resolved",
            "final_category": "Network",
            "final_priority": "High",
            "mode": "manual",
        },
        headers=role("l1"),
    )


def _groups(c: Any, **params: str) -> dict[str, dict[str, Any]]:
    r = c.get("/api/v1/metrics/mttr", params=params, headers=role("l1"))
    assert r.status_code == 200, r.text
    return {g["key"]: g for g in r.json()["groups"]}


def test_assisted_and_manual_timings(
    timed: Any, clock: ManualClock, demo_tickets: dict[str, Any]
) -> None:
    _assisted(timed, clock, demo_tickets["DEMO-001"], 0, 30, 90)  # TTD 60, MTTR 90
    _assisted(timed, clock, demo_tickets["DEMO-003"], 100, 110, 130)  # TTD 20, MTTR 30
    _manual(timed, clock, "T01", "manual-1", 200, 380)  # TTD 180, MTTR 180

    ttd = _groups(timed, group_by="mode", metric="ttd")
    assert ttd["assisted"]["n"] == 2 and ttd["assisted"]["mean_s"] == 40.0
    assert ttd["assisted"]["median_s"] == 40.0 and ttd["assisted"]["p90_s"] == 60.0
    assert ttd["manual"] == {
        "key": "manual",
        "n": 1,
        "mean_s": 180.0,
        "median_s": 180.0,
        "p90_s": 180.0,
    }

    mttr = _groups(timed, group_by="category", metric="mttr")
    assert mttr["Network"]["n"] == 2 and mttr["Network"]["mean_s"] == 135.0  # DEMO-001 + manual
    assert mttr["Hardware"]["mean_s"] == 30.0

    by_priority = _groups(timed, group_by="priority", metric="mttr")
    assert set(by_priority) == {"High", "Medium"}
    by_decision = _groups(timed, group_by="decision")
    assert set(by_decision) == {"recommend", "manual"}
    ttr = _groups(timed, group_by="mode", metric="ttr")
    assert ttr["assisted"]["mean_s"] == 0.0  # same clock instant in this test

    assert _groups(timed, group_by="mode", metric="ttd", mode="manual").keys() == {"manual"}
    assert _groups(timed, group_by="mode", **{"from": "2026-11-18"}) == {}


def test_unresolved_tickets_have_no_mttr(
    timed: Any, clock: ManualClock, demo_tickets: dict[str, Any]
) -> None:
    _assisted(timed, clock, demo_tickets["DEMO-002"], 0, 5, 50, final="escalated")
    assert _groups(timed, group_by="category", metric="mttr") == {}
    assert _groups(timed, group_by="category", metric="ttd")["Database"]["mean_s"] == 45.0


@pytest.mark.parametrize("params", [{"group_by": "user"}, {"metric": "speed"}, {"mode": "x"}])
def test_bad_metric_params_are_400(client: Any, params: dict[str, str]) -> None:
    assert client.get("/api/v1/metrics/mttr", params=params, headers=role("l1")).status_code == 400


def test_ui_may_only_post_ui_events(client: Any) -> None:
    assert _event(client, "T", "a", "ticket.resolved").status_code == 400
    assert _event(client, "T", "a", "manual.started").status_code == 400  # needs mode=manual
    assert _event(client, "T", None, "manual.started", "manual").status_code == 400
    r = _event(client, "T", "s", "manual.started", "manual", h=role("kb_owner"))
    assert r.status_code == 403


def test_summary(timed: Any, clock: ManualClock, demo_tickets: dict[str, Any]) -> None:
    aid = _assisted(timed, clock, demo_tickets["DEMO-001"], 0, 30, 90)
    _assisted(timed, clock, demo_tickets["DEMO-002"], 100, 110, 200, final="escalated")
    timed.post(
        "/api/v1/feedback",
        json={
            "analysis_id": aid,
            "target_type": "kedb_match",
            "target_ref": "KB-0018",
            "verdict": "useful",
        },
        headers=role("l1"),
    )
    s = timed.get("/api/v1/metrics/summary", headers=role("l1")).json()
    assert s["analyses"] == 2
    assert s["decisions"] == {"recommend": 1, "escalate": 1}
    assert s["final_decisions"] == {"resolved": 1, "escalated": 1}
    assert s["escalation_rate"] == 0.5
    assert s["feedback"]["useful"] == 1 and s["feedback"]["rates"]["useful"] == 1.0
    assert s["ttd_by_mode"]["assisted"]["n"] == 2
    base = s["historical_baseline"]
    assert base["synthetic"] is True and "synthetic" in base["label"] and base["n"] > 0


def test_trial_tickets_have_no_gold(client: Any) -> None:
    items = client.get("/api/v1/trial/tickets", headers=role("l1")).json()["items"]
    gold = json.loads((ROOT / "data/test/trial_gold.json").read_text(encoding="utf-8"))
    assert {t["id"] for t in items} == set(gold)
    dump = json.dumps(items)
    assert "expected_" not in dump and "gold" not in dump
    assert all(t["category"] is None and t["priority"] is None for t in items)
    assert client.get("/api/v1/trial/tickets", headers=role("kb_owner")).status_code == 403
