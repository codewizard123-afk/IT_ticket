"""Audit endpoints (FR-08, architecture §8.6, §8.7, §10).

* ``GET /audit/verify`` (admin): walk the hash chain.
* ``GET /audit/export?format=csv|jsonl&from&to``: masked unless admin; kb_owner denied; each export
  writes an ``audit.exported`` watermark row.
* ``GET /audit/{analysis_id}``: the full record (l1 own, l2 own or group, admin all).
"""

from __future__ import annotations

import csv
import io
import json
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Query
from fastapi.responses import Response

from backend import access, audit, repo
from backend.deps import DbDep, UserDep
from backend.errors import not_found
from shared.schemas import User

router = APIRouter(prefix="/audit", tags=["audit"])


@router.get("/verify", response_model=audit.VerifyResult)
def verify(conn: DbDep, user: UserDep) -> audit.VerifyResult:
    """Walk the hash chain; ``ok=false`` with ``broken_at`` = first tampered row id."""
    access.require(conn, user, access.AUDIT_ALL, "audit verify")
    return audit.verify_chain(conn)


def _in_range(ts: str, lo: str | None, hi: str | None) -> bool:
    day = (ts or "")[:10]
    return (not lo or day >= lo[:10]) and (not hi or day <= hi[:10])


@router.get("/export", summary="Audit log export (role-masked, watermarked)")
def export(
    conn: DbDep,
    user: UserDep,
    format: Annotated[Literal["csv", "jsonl"], Query()] = "jsonl",  # noqa: A002 - API name
    date_from: Annotated[str | None, Query(alias="from")] = None,
    date_to: Annotated[str | None, Query(alias="to")] = None,
) -> Response:
    access.require(conn, user, access.EXPORT_ROLES, "audit export")
    rows = [
        access.mask_export_row(dict(r), user)
        for r in conn.execute("SELECT * FROM audit_log ORDER BY id")
        if _in_range(r["ts_utc"], date_from, date_to)
    ]
    masked = user.role != "admin"
    audit.append(
        conn, actor=user.username, action="audit.exported", entity="audit", entity_id=format,
        payload={"rows": len(rows), "masked": masked, "from": date_from, "to": date_to},
    )  # fmt: skip
    headers = {"X-Export-Masked": str(masked).lower(), "X-Export-Rows": str(len(rows))}
    if format == "jsonl":
        text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        return Response(text, media_type="application/x-ndjson", headers=headers)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0]) if rows else ["id"], lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return Response(buf.getvalue(), media_type="text/csv", headers=headers)


def _evidence_for_role(
    rows: list[dict[str, Any]], user: User, full: dict[str, Any]
) -> list[dict[str, Any]]:
    """``l1`` never sees L2 step text, including the step excerpts stored with citations."""
    if user.role != "l1":
        return rows
    out = []
    for r in rows:
        if r["kind"] == "citation" and r.get("tier") == "L2":
            r = {**r, "excerpt": f"Handled by {access.group_of(full, int(r['issue_idx']))}"}
        out.append(r)
    return out


def _entry_for_role(entry: audit.AuditEntry, user: User) -> dict[str, Any]:
    """The raw payload holds unmasked step text, so ``l1`` gets the chain metadata only."""
    data = entry.model_dump(mode="json")
    if user.role == "l1":
        data["payload"] = None
    return data


@router.get("/{analysis_id}", summary="Full audit record of one analysis")
def record(analysis_id: str, conn: DbDep, user: UserDep) -> dict[str, Any]:
    a = repo.get_analysis(conn, analysis_id)
    if a is None:
        raise not_found(f"analysis {analysis_id}")
    full = repo.response_of(a).model_dump(mode="json")
    if not access.can_view_analysis(user, a, access.group_of(full)):
        raise access.deny(conn, user, f"audit {analysis_id}")
    ticket = repo.get_ticket(conn, a["ticket_id"]) or {}
    timeline = [
        e
        for e in repo.events(conn, ticket_id=a["ticket_id"])
        if e["analysis_id"] in (analysis_id, None)
    ]
    feedback = [
        dict(r) for r in conn.execute("SELECT * FROM feedback WHERE analysis_id=?", (analysis_id,))
    ]
    decision = conn.execute(
        "SELECT * FROM decisions WHERE analysis_id=?", (analysis_id,)
    ).fetchone()
    overrides = [
        dict(r) for r in conn.execute("SELECT * FROM overrides WHERE analysis_id=?", (analysis_id,))
    ]
    return {
        "analysis_id": analysis_id,
        "ticket_id": a["ticket_id"],
        "user": a["user"],
        "role": a["role"],
        "created_at": a["created_at"],
        "clarification_round": a["clarification_round"],
        "ticket": {"title": ticket.get("title"), "description": ticket.get("description_redacted")},
        "response": access.filter_response(full, user),
        "evidence": _evidence_for_role(repo.evidence_for(conn, analysis_id), user, full),
        "llm_calls": repo.llm_calls_for(conn, analysis_id),
        "timeline": timeline,
        "feedback": feedback,
        "decision": dict(decision) if decision else None,
        "overrides": overrides,
        "audit_entries": [
            _entry_for_role(e, user)
            for e in audit.list_entries(conn, entity="analysis", entity_id=analysis_id)
        ],
        "chain": audit.verify_chain(conn).model_dump(),
        "filtered_for_role": user.role,
    }
