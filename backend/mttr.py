"""MTTR timing (FR-09, architecture §8.10).

| Metric | Definition |
|--------|------------|
| TTR    | ``analysis.completed − ticket.submitted`` |
| TTD    | ``decision.submitted − ui.viewed`` (assisted);                   |
|        | ``manual.decision − manual.started`` (manual)                    |
| MTTR   | ``ticket.resolved − first ticket.submitted`` of the ticket (or ``manual.started``) |

One timing row per analysis (or manual session); grouped by category, priority, mode or decision.
The historical baseline comes from the enriched corpus's synthetic timestamps and is labelled so.
"""

from __future__ import annotations

import json
import sqlite3
import statistics
from collections import Counter
from collections.abc import Iterable, Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

from backend import repo
from backend.intake import strip_gold

GROUP_BY = ("category", "priority", "mode", "decision")
METRICS = ("ttr", "ttd", "mttr")
VERDICTS = ("useful", "incorrect", "unresolved")


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    idx = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return ordered[idx]


def stats(key: str, values: list[float]) -> dict[str, Any]:
    return {
        "key": key,
        "n": len(values),
        "mean_s": round(statistics.fmean(values), 1),
        "median_s": round(statistics.median(values), 1),
        "p90_s": round(percentile(values, 90), 1),
    }


def _secs(a: str, b: str) -> float:
    return (repo.parse_iso(b) - repo.parse_iso(a)).total_seconds()


def timing_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """One row per analysis / manual session with TTR, TTD and MTTR in seconds."""
    by_aid: dict[str, list[dict[str, Any]]] = {}
    first_submit: dict[str, str] = {}
    for e in repo.events(conn):
        if e["event"] == "ticket.submitted":
            first_submit.setdefault(e["ticket_id"], e["ts_utc"])
        if e["analysis_id"]:
            by_aid.setdefault(e["analysis_id"], []).append(e)
    analyses = {a["analysis_id"]: a for a in repo.list_analyses(conn)}
    decisions = {r["analysis_id"]: dict(r) for r in conn.execute("SELECT * FROM decisions")}
    rows = []
    for aid, evs in by_aid.items():
        first: dict[str, str] = {}
        for e in evs:
            first.setdefault(e["event"], e["ts_utc"])
        mode = "manual" if "manual.started" in first else "assisted"
        a, dec = analyses.get(aid), decisions.get(aid, {})
        if a is not None:
            contract = a["contract"]
            category = contract["classification"]["category"]
            priority = contract["priority"]["level"]
            decision = a["decision"]
        else:
            category = dec.get("final_category") or "unknown"
            priority = dec.get("final_priority") or "unknown"
            decision = "manual" if mode == "manual" else "unknown"

        def span(a_ev: str, b_ev: str, first: dict[str, str] = first) -> float | None:
            return _secs(first[a_ev], first[b_ev]) if a_ev in first and b_ev in first else None

        ttd = (
            span("manual.started", "manual.decision")
            if mode == "manual"
            else span("ui.viewed", "decision.submitted")
        )
        ticket_id = evs[0]["ticket_id"]
        start = first_submit.get(ticket_id) if mode == "assisted" else first.get("manual.started")
        mttr = (
            _secs(start, first["ticket.resolved"]) if start and "ticket.resolved" in first else None
        )
        rows.append(
            {
                "analysis_id": aid,
                "ticket_id": ticket_id,
                "mode": mode,
                "category": category,
                "priority": priority,
                "decision": decision,
                "final_decision": dec.get("final_decision"),
                "started_at": start or evs[0]["ts_utc"],
                "ttr_s": span("ticket.submitted", "analysis.completed"),
                "ttd_s": ttd,
                "mttr_s": mttr,
            }
        )
    return rows


def _in_range(ts: str, lo: str | None, hi: str | None) -> bool:
    day = ts[:10]
    return (not lo or day >= lo[:10]) and (not hi or day <= hi[:10])


def metrics_mttr(
    rows: Iterable[Mapping[str, Any]],
    *,
    group_by: str = "category",
    metric: str = "mttr",
    mode: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict[str, Any]:
    groups: dict[str, list[float]] = {}
    for r in rows:
        if mode and r["mode"] != mode:
            continue
        if not _in_range(str(r["started_at"]), date_from, date_to):
            continue
        val = r[f"{metric}_s"]
        if val is not None:
            groups.setdefault(str(r[group_by]), []).append(float(val))
    return {
        "metric": metric,
        "group_by": group_by,
        "groups": [stats(k, v) for k, v in sorted(groups.items())],
    }


@lru_cache(maxsize=4)
def historical_baseline(path: Path) -> dict[str, Any]:
    """Synthetic context only: MTTR from the enriched corpus's generated timestamps."""
    label = "Historical MTTR (synthetic timestamps, context only)"
    if not path.exists():
        return {"synthetic": True, "label": label, "n": 0, "mean_s": None, "median_s": None}
    recs = json.loads(path.read_text(encoding="utf-8"))
    vals = [
        float(r["resolved_ts"] - r["opened_ts"])
        for r in recs
        if r.get("opened_ts") and r.get("resolved_ts") and r["resolved_ts"] >= r["opened_ts"]
    ]
    return {
        "synthetic": True,
        "label": label,
        "n": len(vals),
        "mean_s": round(statistics.fmean(vals), 1) if vals else None,
        "median_s": round(statistics.median(vals), 1) if vals else None,
    }


def summary(conn: sqlite3.Connection, enriched_path: Path) -> dict[str, Any]:
    analyses = repo.list_analyses(conn)
    n = len(analyses)
    verdicts = Counter(r["verdict"] for r in conn.execute("SELECT verdict FROM feedback"))
    total_fb = sum(verdicts.values())
    degraded = [a for a in analyses if a["degraded"]]
    gaps = Counter(
        a["contract"]["classification"]["subcategory"]
        for a in analyses
        if not a["contract"]["kedb_match"]["found"]
    )
    rows = timing_rows(conn)
    ttd_by_mode = {}
    for mode in ("manual", "assisted"):
        vals = [float(r["ttd_s"]) for r in rows if r["mode"] == mode and r["ttd_s"] is not None]
        if vals:
            ttd_by_mode[mode] = stats(mode, vals)
    llm = conn.execute(
        "SELECT COUNT(*) AS n, COALESCE(SUM(cached), 0) AS cached FROM llm_calls"
    ).fetchone()
    finals = Counter(
        r["final_decision"] for r in conn.execute("SELECT final_decision FROM decisions")
    )
    return {
        "analyses": n,
        "decisions": dict(Counter(a["decision"] for a in analyses)),
        "final_decisions": dict(finals),
        "escalation_rate": round(sum(a["escalation_required"] for a in analyses) / n, 3)
        if n
        else 0.0,
        "approval_rate": round(sum(a["human_approval_required"] for a in analyses) / n, 3)
        if n
        else 0.0,
        "feedback": {
            "total": total_fb,
            **{v: verdicts.get(v, 0) for v in VERDICTS},
            "rates": {
                v: round(verdicts.get(v, 0) / total_fb, 3) if total_fb else 0.0 for v in VERDICTS
            },
        },
        "degraded": {
            "count": len(degraded),
            "rate": round(len(degraded) / n, 3) if n else 0.0,
            "by_mode": dict(Counter(m for a in degraded for m in a["degraded"])),
        },
        "ttd_by_mode": ttd_by_mode,
        "kb_gaps": dict(gaps),
        "llm_calls": {"total": int(llm["n"]), "cached": int(llm["cached"])},
        "historical_baseline": historical_baseline(enriched_path),
    }


def trial_tickets(path: Path) -> list[dict[str, Any]]:
    """Trial tickets for the timed trial, with every gold field removed (§9.3)."""
    if not path.exists():
        return []
    out = []
    for t in json.loads(path.read_text(encoding="utf-8")):
        item = {k: v for k, v in t.items() if not str(k).startswith(("expected_", "gold"))}
        item["metadata"] = strip_gold(t.get("metadata"))
        for k in ("category", "subcategory", "priority", "resolution"):
            item[k] = None
        out.append(item)
    return out
