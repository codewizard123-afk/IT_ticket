"""Export the trial's raw events to ``trial/results/raw_events.csv`` for P1's MTTR report.

Reads the admin (unmasked) audit export from the API (``GET /audit/export?format=jsonl``), keeps
the timer events of trial tickets and attaches each decision to its decision event:

* assisted: ``ui.viewed`` → ``decision.submitted`` (TTD)
* manual: ``manual.started`` → ``manual.decision`` (TTD)

Usage (log in as an admin; the export is masked for everyone else)::

    python -m trial.export --username admin --password ... [--schedule trial/results/schedule.csv]
    MOCK_API=1 python -m trial.export --demo-role admin      # dry run against the mock

Assumption to confirm with P3: audit rows look like ``{action, entity_id, payload_json, ts_utc}``,
with ``action == "event"`` for ticket events and ``action == "decision"`` for ``/decisions``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ui.client import ApiClient
from ui.components.feedback import CONFIDENCE_PREFIX
from ui.settings import ROOT

OUT = ROOT / "trial" / "results" / "raw_events.csv"
FIELDS = (
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
)
TRIAL_EVENTS = {
    "ticket.submitted",
    "analysis.completed",
    "ui.viewed",
    "clarify.requested",
    "clarify.answered",
    "decision.submitted",
    "ticket.resolved",
    "manual.started",
    "manual.decision",
}
DECISION_EVENTS = {"decision.submitted", "manual.decision"}
_CONF = re.compile(re.escape(CONFIDENCE_PREFIX) + r"(\d)/5")


def parse_confidence(notes: str | None) -> int | None:
    m = _CONF.search(notes or "")
    return int(m.group(1)) if m else None


def load_schedule(path: Path | None) -> dict[tuple[str, str], dict[str, str]]:
    if not path or not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return {(r["participant"], r["ticket_id"]): r for r in csv.DictReader(fh)}


def build_rows(
    audit_rows: Iterable[dict[str, Any]],
    trial_ids: set[str],
    schedule: dict[tuple[str, str], dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Audit rows → one CSV row per trial event, decisions attached to the decision event."""
    schedule = schedule or {}
    events: list[dict[str, Any]] = []
    decisions: dict[str, dict[str, Any]] = {}
    for row in audit_rows:
        payload = json.loads(row.get("payload_json") or "{}")
        if row.get("action") == "event" and payload.get("event") in TRIAL_EVENTS:
            events.append(payload)
        elif row.get("action") == "decision":
            decisions[payload.get("analysis_id") or row.get("entity_id")] = payload
    # assisted analyses are keyed by analysis id; a clarify round keeps the same ticket id
    trial_analyses = {e["analysis_id"] for e in events if e.get("ticket_id") in trial_ids}
    out = []
    for e in events:
        if e.get("ticket_id") not in trial_ids and e.get("analysis_id") not in trial_analyses:
            continue
        participant = e.get("actor") or ""
        slot = schedule.get((participant, e["ticket_id"]), {})
        row = {
            "participant": participant,
            "group": slot.get("group", ""),
            "seq": slot.get("seq", ""),
            "practice": slot.get("practice", ""),
            "ticket_id": e["ticket_id"],
            "analysis_id": e.get("analysis_id") or "",
            "mode": e.get("mode", ""),
            "event": e["event"],
            "ts_utc": e.get("ts_utc", ""),
        }
        if e["event"] in DECISION_EVENTS:
            d = decisions.get(e.get("analysis_id") or "", {})
            row.update(
                final_decision=d.get("final_decision", ""),
                final_category=d.get("final_category", ""),
                final_subcategory=d.get("final_subcategory", ""),
                final_priority=d.get("final_priority", ""),
                kb_or_incident_used=d.get("kb_or_incident_used", ""),
                confidence=parse_confidence(d.get("resolution_notes")) or "",
            )
        out.append({k: row.get(k, "") for k in FIELDS})
    out.sort(key=lambda r: (r["participant"], r["ts_utc"]))
    return out


def export(client: ApiClient, out: Path = OUT, schedule_path: Path | None = None) -> int:
    trial_ids = {t["id"] for t in client.trial_tickets()}
    text = client.audit_export("jsonl")
    rows = build_rows(
        (json.loads(line) for line in text.splitlines() if line.strip()),
        trial_ids,
        load_schedule(schedule_path),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export trial events to raw_events.csv")
    parser.add_argument("--username")
    parser.add_argument("--password")
    parser.add_argument("--demo-role", default=None, help="use X-Demo-Role instead of a login")
    parser.add_argument("--schedule", type=Path, default=ROOT / "trial/results/schedule.csv")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    client = ApiClient.from_env()
    if args.username:
        client.login(args.username, args.password or "")
    else:
        client.use_demo_role(args.demo_role or "admin")
    if client.role != "admin":
        print("warning: only an admin export is unmasked; participants will be pseudonyms")
    n = export(client, args.out, args.schedule)
    print(f"wrote {n} event rows to {args.out}")


if __name__ == "__main__":
    main()
