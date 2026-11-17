"""Hash-chained, append-only audit log (FR-08, architecture §8.7).

Each ``audit_log`` row stores ``prev_hash`` (the previous row's ``row_hash``, or the configured
genesis hash for the first row) and ``row_hash = sha256(prev_hash + canonical(record))``.

``record`` covers every stored field (``ts_utc, actor, action, entity, entity_id,
payload_json``), not only the payload, so rewriting who/when/what is detected too.
:func:`verify_chain` walks the table and reports the first row where the chain breaks:
an edited row (hash mismatch), a deleted or inserted row (link mismatch or id gap), or a
truncated tail (``sqlite_sequence`` is ahead of the last row).

Callers must pass already-redacted payloads; this module never logs payload contents.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel

from backend.config import backend_config
from backend.db.sqlite import transaction

log = logging.getLogger(__name__)

HASHED_FIELDS = ("ts_utc", "actor", "action", "entity", "entity_id", "payload_json")


class AuditEntry(BaseModel):
    id: int
    ts_utc: str
    actor: str
    action: str
    entity: str
    entity_id: str
    payload: Any
    prev_hash: str
    row_hash: str


class VerifyResult(BaseModel):
    ok: bool
    checked: int
    broken_at: int | None = None
    reason: str | None = None


def genesis_hash() -> str:
    return str(backend_config().get("audit", {}).get("genesis_hash", "0" * 64))


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace, UTF-8 kept, non-JSON types via ``str``."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def compute_row_hash(row: Mapping[str, Any]) -> str:
    """``sha256(prev_hash + canonical(record))`` for a row-like mapping."""
    record = canonical_json({k: row[k] for k in HASHED_FIELDS})
    return hashlib.sha256((str(row["prev_hash"]) + record).encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _to_entry(row: sqlite3.Row) -> AuditEntry:
    data = dict(row)
    data["payload"] = json.loads(data.pop("payload_json") or "null")
    return AuditEntry(**data)


def append(
    conn: sqlite3.Connection,
    *,
    actor: str,
    action: str,
    entity: str,
    entity_id: str,
    payload: Any,
    ts_utc: str | None = None,
) -> AuditEntry:
    """Append one row to the chain. ``payload`` must already be redacted."""
    row: dict[str, Any] = {
        "ts_utc": ts_utc or _utc_now(),
        "actor": actor,
        "action": action,
        "entity": entity,
        "entity_id": str(entity_id),
        "payload_json": canonical_json(payload),
    }
    with transaction(conn):
        last = conn.execute("SELECT row_hash FROM audit_log ORDER BY id DESC LIMIT 1").fetchone()
        row["prev_hash"] = last["row_hash"] if last else genesis_hash()
        row["row_hash"] = compute_row_hash(row)
        cur = conn.execute(
            "INSERT INTO audit_log (ts_utc, actor, action, entity, entity_id, payload_json, "
            "prev_hash, row_hash) VALUES (:ts_utc, :actor, :action, :entity, :entity_id, "
            ":payload_json, :prev_hash, :row_hash)",
            row,
        )
        row_id = int(cur.lastrowid or 0)
    log.debug("audit append id=%s action=%s entity=%s", row_id, action, entity)
    payload_out = json.loads(row.pop("payload_json"))
    return AuditEntry(id=row_id, payload=payload_out, **row)


def verify_chain(conn: sqlite3.Connection) -> VerifyResult:
    """Walk ``audit_log`` in id order; return the first broken row id, if any."""
    expected_prev = genesis_hash()
    expected_id: int | None = None
    checked = 0
    last_id = 0
    for row in conn.execute("SELECT * FROM audit_log ORDER BY id"):
        rid = int(row["id"])
        if expected_id is not None and rid != expected_id:
            return VerifyResult(ok=False, checked=checked, broken_at=rid, reason="id gap")
        if row["prev_hash"] != expected_prev:
            return VerifyResult(ok=False, checked=checked, broken_at=rid, reason="prev_hash link")
        if compute_row_hash(row) != row["row_hash"]:
            return VerifyResult(ok=False, checked=checked, broken_at=rid, reason="row_hash")
        expected_prev = row["row_hash"]
        expected_id = rid + 1
        last_id = rid
        checked += 1

    seq = conn.execute("SELECT seq FROM sqlite_sequence WHERE name = 'audit_log'").fetchone()
    if seq is not None and int(seq["seq"]) > last_id:
        return VerifyResult(
            ok=False, checked=checked, broken_at=last_id + 1, reason="tail truncated"
        )
    return VerifyResult(ok=True, checked=checked)


def list_entries(
    conn: sqlite3.Connection, *, entity: str | None = None, entity_id: str | None = None
) -> list[AuditEntry]:
    """Audit rows, oldest first, optionally filtered by entity and/or entity id."""
    sql = "SELECT * FROM audit_log"
    clauses, params = [], []
    if entity is not None:
        clauses.append("entity = ?")
        params.append(entity)
    if entity_id is not None:
        clauses.append("entity_id = ?")
        params.append(entity_id)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    return [_to_entry(r) for r in conn.execute(sql + " ORDER BY id", params)]
