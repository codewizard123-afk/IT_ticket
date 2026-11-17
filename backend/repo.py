"""SQLite reads and writes for ``data/app.db`` (architecture §11.2).

Pure functions over a connection.

Only redacted text is ever written here (architecture §13).
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from backend.db.sqlite import transaction
from shared.schemas import AnalysisResponse

TICKET_COLS = (
    "ticket_id", "title", "description_redacted", "reporter_category", "reporter_priority",
    "application", "environment", "location", "assignment_group", "restricted", "status",
    "created_at", "updated_at", "raw_hash",
)  # fmt: skip
ANALYSIS_COLS = (
    "analysis_id", "ticket_id", "created_at", "user", "role", "clarification_round",
    "contract_json", "details_json", "signals_json", "decision", "escalation_required",
    "human_approval_required", "pipeline_version", "embed_model", "llm_models", "degraded",
    "timings_json",
)  # fmt: skip
EVIDENCE_COLS = (
    "analysis_id", "issue_idx", "kind", "ref_id", "rank", "score", "tier", "excerpt",
    "matched_field", "why", "origin",
)  # fmt: skip
LLM_COLS = (
    "call_id", "analysis_id", "purpose", "model", "prompt_hash", "tokens_in", "tokens_out",
    "latency_ms", "status", "cached", "created_at",
)  # fmt: skip


def iso(ts: datetime) -> str:
    return ts.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def parse_iso(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _insert(
    conn: sqlite3.Connection, table: str, cols: Sequence[str], row: Mapping[str, Any]
) -> None:
    placeholders = ", ".join(f":{c}" for c in cols)
    conn.execute(
        f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders})",
        {c: row.get(c) for c in cols},
    )


# --- tickets ---------------------------------------------------------------------------------
def count_ticket_ids(conn: sqlite3.Connection, prefix: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM tickets WHERE ticket_id LIKE ?", (prefix.replace("%", "") + "%",)
    ).fetchone()
    return int(row[0])


def upsert_ticket(conn: sqlite3.Connection, row: Mapping[str, Any]) -> None:
    cols = [c for c in TICKET_COLS if c != "created_at"]
    updates = ", ".join(f"{c}=excluded.{c}" for c in cols if c != "ticket_id")
    placeholders = ", ".join(f":{c}" for c in TICKET_COLS)
    conn.execute(
        f"INSERT INTO tickets ({', '.join(TICKET_COLS)}) VALUES ({placeholders}) "
        f"ON CONFLICT(ticket_id) DO UPDATE SET {updates}",
        {c: row.get(c) for c in TICKET_COLS},
    )


def get_ticket(conn: sqlite3.Connection, ticket_id: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,)).fetchone()
    return dict(row) if row else None


# --- analyses --------------------------------------------------------------------------------
def save_analysis(
    conn: sqlite3.Connection,
    row: Mapping[str, Any],
    evidence: Iterable[Mapping[str, Any]] = (),
    llm_calls: Iterable[Mapping[str, Any]] = (),
) -> None:
    with transaction(conn):
        _insert(conn, "analyses", ANALYSIS_COLS, row)
        for ev in evidence:
            _insert(conn, "evidence", EVIDENCE_COLS, ev)
        for call in llm_calls:
            _insert(conn, "llm_calls", LLM_COLS, call)


def _analysis_from_row(row: sqlite3.Row) -> dict[str, Any]:
    out = dict(row)
    for key in ("contract_json", "details_json", "signals_json", "timings_json"):
        out[key.removesuffix("_json")] = json.loads(out.pop(key) or "null")
    out["escalation_required"] = bool(out["escalation_required"])
    out["human_approval_required"] = bool(out["human_approval_required"])
    out["degraded"] = [d for d in (out.get("degraded") or "").split(",") if d]
    return out


def get_analysis(conn: sqlite3.Connection, analysis_id: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM analyses WHERE analysis_id = ?", (analysis_id,)).fetchone()
    return _analysis_from_row(row) if row else None


def list_analyses(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute("SELECT * FROM analyses ORDER BY created_at, analysis_id").fetchall()
    return [_analysis_from_row(r) for r in rows]


def response_of(analysis: Mapping[str, Any]) -> AnalysisResponse:
    """Rebuild the full (unfiltered) response stored for an analysis."""
    return AnalysisResponse.model_validate({**analysis["contract"], "details": analysis["details"]})


def evidence_for(conn: sqlite3.Connection, analysis_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM evidence WHERE analysis_id = ? ORDER BY issue_idx, kind, rank",
        (analysis_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def llm_calls_for(conn: sqlite3.Connection, analysis_id: str) -> list[dict[str, Any]]:
    rows = conn.execute("SELECT * FROM llm_calls WHERE analysis_id = ?", (analysis_id,)).fetchall()
    return [dict(r) for r in rows]


# --- events ----------------------------------------------------------------------------------
def record_event(
    conn: sqlite3.Connection,
    *,
    ticket_id: str,
    analysis_id: str | None,
    event: str,
    actor: str,
    mode: str,
    ts_utc: str,
) -> int:
    cur = conn.execute(
        "INSERT INTO ticket_events (ticket_id, analysis_id, event, ts_utc, actor, mode) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (ticket_id, analysis_id, event, ts_utc, actor, mode),
    )
    return int(cur.lastrowid or 0)


def events(
    conn: sqlite3.Connection,
    *,
    analysis_id: str | None = None,
    ticket_id: str | None = None,
) -> list[dict[str, Any]]:
    sql, clauses, params = "SELECT * FROM ticket_events", [], []
    if analysis_id is not None:
        clauses.append("analysis_id = ?")
        params.append(analysis_id)
    if ticket_id is not None:
        clauses.append("ticket_id = ?")
        params.append(ticket_id)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    return [dict(r) for r in conn.execute(sql + " ORDER BY id", params)]
