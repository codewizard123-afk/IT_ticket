"""P3-T1: GET /audit/verify over HTTP."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import audit, deps
from backend.db import sqlite as db
from backend.routers import audit as audit_router


@pytest.fixture
def conn() -> sqlite3.Connection:
    c = db.open_db(":memory:")
    for i in range(3):
        audit.append(c, actor="a", action="x", entity="e", entity_id=str(i), payload={"i": i})
    return c


@pytest.fixture
def client(conn: sqlite3.Connection) -> TestClient:
    app = FastAPI()
    app.include_router(audit_router.router, prefix="/api/v1")

    def _db() -> Iterator[sqlite3.Connection]:
        yield conn

    app.dependency_overrides[deps.get_db] = _db
    return TestClient(app)


def test_verify_ok(client: TestClient) -> None:
    body = client.get("/api/v1/audit/verify").json()
    assert body == {"ok": True, "checked": 3, "broken_at": None, "reason": None}


def test_verify_reports_tampering(client: TestClient, conn: sqlite3.Connection) -> None:
    conn.execute("DROP TRIGGER audit_log_no_update")
    conn.execute("UPDATE audit_log SET payload_json='{\"i\":9}' WHERE id=2")
    body = client.get("/api/v1/audit/verify").json()
    assert body["ok"] is False
    assert body["broken_at"] == 2
    assert body["reason"]
