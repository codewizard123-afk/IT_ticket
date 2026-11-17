"""MTTR report (P1-T8) on a small hand-built trial export (test fixture only)."""

from __future__ import annotations

import csv
import json
import sqlite3

import pytest

from eval.mttr_report import (
    build_report,
    live_metrics,
    observations,
    parse_ts,
    score,
)

FIELDS = [
    "participant",
    "group",
    "seq",
    "practice",
    "ticket_id",
    "analysis_id",
    "mode",
    "event",
    "ts_utc",
    "final_decision",
    "final_category",
    "final_subcategory",
    "final_priority",
    "kb_or_incident_used",
    "confidence",
]


def _row(p, ticket, mode, event, ts, **dec):
    r = dict.fromkeys(FIELDS, "")
    r.update(participant=p, ticket_id=ticket, mode=mode, event=event, ts_utc=ts, **dec)
    return r


GOOD_T01 = dict(  # gold T01: Network/VPN, High, KB-0018, recommend
    final_decision="resolved",
    final_category="Network",
    final_subcategory="VPN",
    final_priority="High",
    kb_or_incident_used="KB-0018",
    confidence="4",
)
GOOD_T03 = dict(  # gold T03: Database/Oracle, Critical, escalate
    final_decision="escalated",
    final_category="Database",
    final_subcategory="Oracle",
    final_priority="Critical",
    kb_or_incident_used="",
    confidence="3",
)


@pytest.fixture()
def rows():
    return [
        # practice row is ignored
        _row("pA", "DEMO-001", "assisted", "ui.viewed", "2026-11-20T10:00:00Z", practice="True"),
        # pA: T01 manual 300 s, T03 assisted 100 s
        _row("pA", "T01", "manual", "manual.started", "2026-11-20T10:05:00Z"),
        _row("pA", "T01", "manual", "manual.decision", "2026-11-20T10:10:00Z", **GOOD_T01),
        _row("pA", "T03", "assisted", "ui.viewed", "2026-11-20T10:20:00Z"),
        _row("pA", "T03", "assisted", "decision.submitted", "2026-11-20T10:21:40Z", **GOOD_T03),
        # pB: T01 assisted 120 s with a curated incident linked to KB-0018; T03 manual 400 s wrong
        _row("pB", "T01", "assisted", "ui.viewed", "2026-11-20T11:00:00Z"),
        _row(
            "pB",
            "T01",
            "assisted",
            "decision.submitted",
            "2026-11-20T11:02:00Z",
            **{**GOOD_T01, "kb_or_incident_used": "INC-H0001", "final_priority": "Medium"},
        ),
        _row("pB", "T03", "manual", "manual.started", "2026-11-20T11:10:00Z"),
        _row(
            "pB",
            "T03",
            "manual",
            "manual.decision",
            "2026-11-20T11:16:40Z",
            **{**GOOD_T03, "final_decision": "resolved", "final_subcategory": "DB2"},
        ),
        # incomplete pair is skipped
        _row("pB", "T05", "manual", "manual.started", "2026-11-20T11:30:00Z"),
    ]


def test_parse_ts():
    assert parse_ts("2026-11-20T10:00:00Z") == parse_ts("2026-11-20T10:00:00+00:00")
    assert parse_ts("2026-11-20T10:00:00") == parse_ts("2026-11-20T10:00:00Z")


def test_observations(rows):
    obs = observations(rows, {f"T{i:02d}" for i in range(1, 13)})
    got = {(o["participant"], o["ticket_id"], o["mode"]): o["ttd_s"] for o in obs}
    assert got == {
        ("pA", "T01", "manual"): 300,
        ("pA", "T03", "assisted"): 100,
        ("pB", "T01", "assisted"): 120,
        ("pB", "T03", "manual"): 400,
    }


def test_score_rules():
    gold_rec = {
        "category": "Network",
        "subcategory": "VPN",
        "priority": "High",
        "kb_ids": ["KB-0018"],
        "decision": "recommend",
    }
    obs = {**GOOD_T01, "final_priority": "P3"}  # internal code accepted, Medium is within 1
    assert score(obs, gold_rec, {})["all"]
    via_incident = {**GOOD_T01, "kb_or_incident_used": "INC-H0001"}
    assert score(via_incident, gold_rec, {"INC-H0001": {"KB-0018"}})["kb_or_escalation"]
    assert not score({**GOOD_T01, "kb_or_incident_used": "KB-0019"}, gold_rec, {})["all"]
    gold_esc = {
        "category": "Database",
        "subcategory": "Oracle",
        "priority": "Critical",
        "kb_ids": ["KB-0032"],
        "decision": "escalate",
    }
    assert score(GOOD_T03, gold_esc, {})["all"]
    assert not score({**GOOD_T03, "final_decision": "resolved"}, gold_esc, {})["kb_or_escalation"]
    assert not score({**GOOD_T03, "final_priority": "Low"}, gold_esc, {})["priority_within_one"]


def test_full_report(tmp_path, rows):
    events = tmp_path / "raw_events.csv"
    with events.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    db = tmp_path / "app.db"
    with sqlite3.connect(db) as con:
        con.execute(
            "CREATE TABLE ticket_events (id INTEGER PRIMARY KEY, ticket_id TEXT, analysis_id TEXT,"
            " event TEXT, ts_utc TEXT, actor TEXT, mode TEXT)"
        )
        con.executemany(
            "INSERT INTO ticket_events (ticket_id, analysis_id, event, ts_utc, actor, mode) "
            "VALUES (?, ?, ?, ?, 'u', 'assisted')",
            [
                ("X1", "a1", "ticket.submitted", "2026-11-20T09:00:00Z"),
                ("X1", "a1", "analysis.completed", "2026-11-20T09:00:02Z"),
                ("X1", "a1", "ticket.resolved", "2026-11-20T10:00:00Z"),
                ("X2", "a2", "ticket.submitted", "2026-11-20T09:00:00Z"),
                ("X2", "a2", "analysis.completed", "2026-11-20T09:00:04Z"),
            ],
        )
    out = tmp_path / "out"
    r = build_report(events_path=events, db_path=db, out_dir=out, plots=True)
    t = r["trial"]
    assert t["status"] == "ok" and t["participants"] == 2 and t["observations"] == 4
    assert t["ttd"]["manual"]["median_s"] == 350 and t["ttd"]["assisted"]["median_s"] == 110
    assert t["median_ttd_change_pct"] == pytest.approx((110 - 350) / 350 * 100)
    assert t["correctness"]["assisted"]["all"] == 1.0  # both assisted decisions correct
    assert t["correctness"]["manual"]["all"] == 0.5
    assert t["participants_faster_assisted"] == "2/2"
    assert r["live"]["ttr_s"]["n"] == 2 and r["live"]["ttr_s"]["p50"] == 3
    assert r["live"]["mttr_s"]["median_s"] == 3600
    assert "SYNTHETIC" in r["historical_synthetic"]["label"]
    saved = json.loads((out / "mttr.json").read_text())
    assert saved["trial"]["observations"] == 4 and saved["caveats"]
    if r.get("charts"):
        assert {"mttr_ttd.png", "mttr_correctness.png", "mttr_historical.png"} <= set(r["charts"])


def test_no_trial_data_is_reported_not_invented(tmp_path):
    r = build_report(
        events_path=tmp_path / "missing.csv",
        db_path=tmp_path / "no.db",
        out_dir=tmp_path,
        plots=False,
    )
    assert r["trial"]["status"] == "no_trial_data"
    assert r["live"]["status"].startswith("no app.db")
    assert r["historical_synthetic"]["category"]


def test_live_metrics_bad_db(tmp_path):
    db = tmp_path / "app.db"
    sqlite3.connect(db).close()
    assert live_metrics(db)["status"].startswith("ticket_events unavailable")
