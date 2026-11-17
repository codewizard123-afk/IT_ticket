"""P3-T1: SQLite schema (architecture §11.2) and connection helpers."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from backend.db import sqlite as db

EXPECTED_TABLES = {
    "tickets": {
        "ticket_id",
        "title",
        "description_redacted",
        "reporter_category",
        "reporter_priority",
        "application",
        "environment",
        "location",
        "assignment_group",
        "restricted",
        "status",
        "created_at",
        "updated_at",
        "raw_hash",
    },
    "analyses": {
        "analysis_id",
        "ticket_id",
        "created_at",
        "user",
        "role",
        "clarification_round",
        "contract_json",
        "details_json",
        "signals_json",
        "decision",
        "escalation_required",
        "human_approval_required",
        "pipeline_version",
        "embed_model",
        "llm_models",
        "degraded",
        "timings_json",
    },
    "evidence": {
        "analysis_id",
        "issue_idx",
        "kind",
        "ref_id",
        "rank",
        "score",
        "tier",
        "excerpt",
        "matched_field",
        "why",
        "origin",
    },
    "feedback": {
        "feedback_id",
        "analysis_id",
        "issue_idx",
        "target_type",
        "target_ref",
        "verdict",
        "corrected_value",
        "comment",
        "user",
        "created_at",
    },
    "decisions": {
        "analysis_id",
        "mode",
        "final_decision",
        "final_category",
        "final_subcategory",
        "final_priority",
        "kb_or_incident_used",
        "resolution_notes",
        "user",
        "decided_at",
    },
    "ticket_events": {"id", "ticket_id", "analysis_id", "event", "ts_utc", "actor", "mode"},
    "audit_log": {
        "id",
        "ts_utc",
        "actor",
        "action",
        "entity",
        "entity_id",
        "payload_json",
        "prev_hash",
        "row_hash",
    },
    "llm_calls": {
        "call_id",
        "analysis_id",
        "purpose",
        "model",
        "prompt_hash",
        "tokens_in",
        "tokens_out",
        "latency_ms",
        "status",
        "cached",
        "created_at",
    },
    "kb_drafts": {
        "draft_id",
        "article_json",
        "status",
        "proposed_by",
        "source_analysis_id",
        "approved_by",
        "created_at",
        "approved_at",
    },
    "overrides": {"analysis_id", "field", "old_value", "new_value", "user", "ts_utc"},
}


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}


def test_schema_has_every_table_and_column_from_architecture() -> None:
    conn = db.open_db(":memory:")
    for table, cols in EXPECTED_TABLES.items():
        assert _columns(conn, table) == cols, table


def test_init_db_is_idempotent() -> None:
    conn = db.open_db(":memory:")
    db.init_db(conn)
    db.init_db(conn)
    assert _columns(conn, "audit_log") == EXPECTED_TABLES["audit_log"]


def test_open_db_creates_parent_dir_and_persists(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "app.db"
    conn = db.open_db(path)
    with db.transaction(conn):
        conn.execute("INSERT INTO tickets (ticket_id, title) VALUES ('T-1', 'x')")
    conn.close()
    conn2 = db.open_db(path)
    assert conn2.execute("SELECT title FROM tickets WHERE ticket_id='T-1'").fetchone()[0] == "x"
    conn2.close()


def test_rows_are_dict_like() -> None:
    conn = db.open_db(":memory:")
    conn.execute("INSERT INTO tickets (ticket_id, title) VALUES ('T-1', 'x')")
    row = conn.execute("SELECT * FROM tickets").fetchone()
    assert row["ticket_id"] == "T-1"


def test_transaction_rolls_back_on_error() -> None:
    conn = db.open_db(":memory:")
    with pytest.raises(RuntimeError), db.transaction(conn):
        conn.execute("INSERT INTO tickets (ticket_id) VALUES ('T-1')")
        raise RuntimeError("boom")
    assert conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0] == 0


def test_default_path_comes_from_config_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("APP_DB_PATH", raising=False)
    assert db.default_db_path().as_posix().endswith("data/app.db")
    monkeypatch.setenv("APP_DB_PATH", "/tmp/other.db")
    assert db.default_db_path() == Path("/tmp/other.db")
