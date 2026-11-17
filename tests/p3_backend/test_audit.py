"""P3-T1: hash-chained audit log (architecture §8.7). Done when tampering with a row is detected."""

from __future__ import annotations

import sqlite3

import pytest

from backend import audit
from backend.db import sqlite as db


@pytest.fixture
def conn() -> sqlite3.Connection:
    c = db.open_db(":memory:")
    for i in range(5):
        audit.append(
            c,
            actor="alice",
            action="analysis.created",
            entity="analysis",
            entity_id=f"A-{i}",
            payload={"decision": "recommend", "n": i, "nested": {"b": 2, "a": 1}},
        )
    return c


def _drop_guards(c: sqlite3.Connection) -> None:
    """Simulate an attacker with direct file access (the triggers only stop accidents)."""
    c.execute("DROP TRIGGER audit_log_no_update")
    c.execute("DROP TRIGGER audit_log_no_delete")


def test_append_links_rows_into_a_chain(conn: sqlite3.Connection) -> None:
    rows = conn.execute("SELECT * FROM audit_log ORDER BY id").fetchall()
    assert rows[0]["prev_hash"] == audit.genesis_hash()
    for prev, cur in zip(rows, rows[1:], strict=False):
        assert cur["prev_hash"] == prev["row_hash"]
    assert len({r["row_hash"] for r in rows}) == 5


def test_append_returns_entry_with_hashes(conn: sqlite3.Connection) -> None:
    entry = audit.append(
        conn,
        actor="bob",
        action="feedback.created",
        entity="feedback",
        entity_id="F-1",
        payload={"verdict": "useful"},
    )
    last = conn.execute("SELECT * FROM audit_log ORDER BY id DESC LIMIT 1").fetchone()
    assert entry.id == last["id"]
    assert entry.row_hash == last["row_hash"]
    assert entry.payload == {"verdict": "useful"}


def test_canonical_payload_is_key_order_independent() -> None:
    assert audit.canonical_json({"b": 1, "a": 2}) == audit.canonical_json({"a": 2, "b": 1})


def test_clean_chain_verifies(conn: sqlite3.Connection) -> None:
    result = audit.verify_chain(conn)
    assert result.ok is True
    assert result.broken_at is None
    assert result.checked == 5


def test_empty_chain_verifies() -> None:
    result = audit.verify_chain(db.open_db(":memory:"))
    assert result.ok and result.checked == 0


def test_update_and_delete_are_blocked_by_triggers(conn: sqlite3.Connection) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE audit_log SET actor='mallory' WHERE id=2")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("DELETE FROM audit_log WHERE id=2")


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("payload_json", '{"decision":"escalate","n":2,"nested":{"a":1,"b":2}}'),
        ("actor", "mallory"),
        ("action", "analysis.deleted"),
        ("ts_utc", "2000-01-01T00:00:00Z"),
        ("entity_id", "A-999"),
    ],
)
def test_tampering_with_a_field_is_detected(
    conn: sqlite3.Connection, column: str, value: str
) -> None:
    _drop_guards(conn)
    conn.execute(f"UPDATE audit_log SET {column}=? WHERE id=3", (value,))
    result = audit.verify_chain(conn)
    assert result.ok is False
    assert result.broken_at == 3


def test_rehashing_a_tampered_row_breaks_the_next_link(conn: sqlite3.Connection) -> None:
    _drop_guards(conn)
    row = conn.execute("SELECT * FROM audit_log WHERE id=3").fetchone()
    forged = dict(row) | {"actor": "mallory"}
    new_hash = audit.compute_row_hash(forged)
    conn.execute("UPDATE audit_log SET actor='mallory', row_hash=? WHERE id=3", (new_hash,))
    result = audit.verify_chain(conn)
    assert not result.ok and result.broken_at == 4


def test_deleting_a_middle_row_is_detected(conn: sqlite3.Connection) -> None:
    _drop_guards(conn)
    conn.execute("DELETE FROM audit_log WHERE id=3")
    result = audit.verify_chain(conn)
    assert not result.ok and result.broken_at == 4


def test_truncating_the_tail_is_detected(conn: sqlite3.Connection) -> None:
    _drop_guards(conn)
    conn.execute("DELETE FROM audit_log WHERE id=5")
    result = audit.verify_chain(conn)
    assert not result.ok and result.broken_at == 5


def test_inserting_a_forged_row_is_detected(conn: sqlite3.Connection) -> None:
    conn.execute(
        "INSERT INTO audit_log (ts_utc, actor, action, entity, entity_id, payload_json, "
        "prev_hash, row_hash) VALUES ('t', 'mallory', 'x', 'y', 'z', '{}', 'bad', 'bad')"
    )
    result = audit.verify_chain(conn)
    assert not result.ok and result.broken_at == 6


def test_list_entries_filters_by_entity(conn: sqlite3.Connection) -> None:
    entries = audit.list_entries(conn, entity="analysis", entity_id="A-2")
    assert [e.entity_id for e in entries] == ["A-2"]
    assert entries[0].payload["nested"] == {"a": 1, "b": 2}


def test_payload_with_raw_secret_text_is_not_logged(
    conn: sqlite3.Connection, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("DEBUG")
    audit.append(
        conn, actor="a", action="x", entity="e", entity_id="1", payload={"text": "pw=hunter2"}
    )
    assert "hunter2" not in caplog.text
