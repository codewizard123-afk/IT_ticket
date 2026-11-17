"""P4-T8: seeded crossover schedule and raw_events.csv export (dry run on the mock API)."""

from __future__ import annotations

import csv
from collections import Counter

from trial.export import build_rows, export, parse_confidence
from trial.schedule import assign_groups, make_schedule, write_csv

PEOPLE = ["ana", "ben", "cai", "dev"]
FIRST = [f"T{i:02d}" for i in range(1, 7)]
SECOND = [f"T{i:02d}" for i in range(7, 13)]


def test_groups_are_balanced_and_seeded():
    g = assign_groups(PEOPLE, 42)
    assert Counter(g.values()) == {"A": 2, "B": 2}
    assert g == assign_groups(list(reversed(PEOPLE)), 42)


def test_crossover_design():
    slots = make_schedule(PEOPLE, seed=42)
    assert len(slots) == len(PEOPLE) * 13
    for p in PEOPLE:
        mine = [s for s in slots if s.participant == p]
        assert mine[0].practice and mine[0].ticket_id == "DEMO-001"
        timed = [s for s in mine if not s.practice]
        assert sorted(s.ticket_id for s in timed) == FIRST + SECOND
        modes = {s.ticket_id: s.mode for s in timed}
        if mine[0].group == "A":
            assert {modes[t] for t in FIRST} == {"manual"}
            assert {modes[t] for t in SECOND} == {"assisted"}
        else:
            assert {modes[t] for t in FIRST} == {"assisted"}
            assert {modes[t] for t in SECOND} == {"manual"}
        assert [s.seq for s in timed] == list(range(1, 13))


def test_schedule_is_reproducible_and_shuffled():
    assert make_schedule(PEOPLE, seed=42) == make_schedule(PEOPLE, seed=42)
    assert make_schedule(PEOPLE, seed=42) != make_schedule(PEOPLE, seed=7)
    orders = {
        tuple(s.ticket_id for s in make_schedule(PEOPLE, seed=42) if s.participant == p)
        for p in PEOPLE
    }
    assert len(orders) > 1  # block order differs between participants


def test_confidence_parsing():
    assert parse_confidence("did x\nSelf-reported confidence: 4/5") == 4
    assert parse_confidence("no rating") is None


def test_dry_run_export(tmp_path, backend, make_client, clock):
    """4 participants × 12 tickets through the client, then export: ≥ 48 decisions recorded."""
    for p in PEOPLE:
        backend.cfg["users"][p] = {"role": "l1", "assignment_group": "SERVICE-DESK"}
    slots = make_schedule(PEOPLE, seed=42)
    write_csv(slots, tmp_path / "schedule.csv")
    trial = {t["id"]: t for t in backend.trial_tickets}
    for s in (s for s in slots if not s.practice):
        client = make_client(None)
        client.login(s.participant, "pw")
        if s.mode == "manual":
            sid = f"manual-{s.ticket_id}-{s.participant}"
            client.event(s.ticket_id, "manual.started", sid, mode="manual")
            clock.tick(200)
            client.decide(
                {
                    "analysis_id": sid,
                    "final_decision": "resolved",
                    "final_category": "Network",
                    "resolution_notes": "Self-reported confidence: 3/5",
                    "mode": "manual",
                }
            )
            client.event(s.ticket_id, "manual.decision", sid, mode="manual")
        else:
            resp = client.analyze(trial[s.ticket_id])
            aid = resp.details.analysis_id
            client.event(s.ticket_id, "ui.viewed", aid)
            clock.tick(120)
            client.decide(
                {
                    "analysis_id": aid,
                    "final_decision": "escalated",
                    "resolution_notes": "Self-reported confidence: 5/5",
                    "mode": "assisted",
                }
            )
        clock.tick(30)
    admin = make_client("admin")
    out = tmp_path / "raw_events.csv"
    n = export(admin, out, tmp_path / "schedule.csv")
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert n == len(rows) > 0
    decisions = [r for r in rows if r["event"] in ("manual.decision", "decision.submitted")]
    assert len(decisions) == len(PEOPLE) * 12
    assert {r["participant"] for r in decisions} == set(PEOPLE)
    assert Counter(r["mode"] for r in decisions) == {"manual": 24, "assisted": 24}
    assert {r["confidence"] for r in decisions} == {"3", "5"}
    assert all(r["group"] in ("A", "B") and r["seq"] for r in decisions)
    starts = [r for r in rows if r["event"] in ("manual.started", "ui.viewed")]
    assert len(starts) == len(PEOPLE) * 12


def test_non_trial_events_are_excluded():
    rows = [
        {
            "action": "event",
            "payload_json": '{"ticket_id": "DEMO-001", "analysis_id": "a1", '
            '"event": "ui.viewed", "actor": "x", "mode": "assisted", "ts_utc": "t"}',
        },
        {
            "action": "event",
            "payload_json": '{"ticket_id": "T01", "analysis_id": "a2", '
            '"event": "ui.viewed", "actor": "x", "mode": "assisted", "ts_utc": "t"}',
        },
    ]
    out = build_rows(rows, {"T01"})
    assert [r["ticket_id"] for r in out] == ["T01"]
