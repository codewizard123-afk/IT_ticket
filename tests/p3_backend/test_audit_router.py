"""P3-T1/T5: GET /audit/verify over HTTP (admin only), tampering is reported."""

from __future__ import annotations

from typing import Any

from backend import audit
from backend.db.sqlite import open_db
from tests.p3_backend.conftest import role


def _seed(services: Any, n: int = 3) -> None:
    with open_db(services.settings.db_path) as conn:
        for i in range(n):
            audit.append(
                conn, actor="a", action="x", entity="e", entity_id=str(i), payload={"i": i}
            )


def test_verify_ok(client: Any, services: Any) -> None:
    _seed(services)
    body = client.get("/api/v1/audit/verify", headers=role("admin")).json()
    assert body == {"ok": True, "checked": 3, "broken_at": None, "reason": None}


def test_verify_reports_tampering(client: Any, services: Any) -> None:
    _seed(services)
    with open_db(services.settings.db_path) as conn:
        conn.execute("DROP TRIGGER audit_log_no_update")
        conn.execute("UPDATE audit_log SET payload_json='{\"i\":9}' WHERE id=2")
    body = client.get("/api/v1/audit/verify", headers=role("admin")).json()
    assert body["ok"] is False and body["broken_at"] == 2 and body["reason"]


def test_verify_is_admin_only_and_denial_is_logged(client: Any, services: Any) -> None:
    r = client.get("/api/v1/audit/verify", headers=role("l2"))
    assert r.status_code == 403 and r.json()["error"]["code"] == "forbidden"
    with open_db(services.settings.db_path) as conn:
        assert [e.action for e in audit.list_entries(conn)] == ["access.denied"]
