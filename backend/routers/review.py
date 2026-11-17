"""``GET /review-queue`` and ``GET /escalations/{bundle_id}`` (architecture §8.2, §10)."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import APIRouter, Query
from fastapi.responses import PlainTextResponse

from backend import access, escalate, repo
from backend.config import backend_config
from backend.deps import DbDep, ServicesDep, UserDep
from backend.errors import not_found

router = APIRouter(tags=["review"])
PRIORITY_RANK = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
Status = Literal["open", "all", "clarify", "escalate", "approval"]


@router.get("/review-queue", summary="Analyses that need a human: clarify, escalate or approval")
def review_queue(
    services: ServicesDep,
    conn: DbDep,
    user: UserDep,
    status: Annotated[Status, Query()] = "open",
    page: Annotated[int, Query(ge=1)] = 1,
) -> dict[str, Any]:
    now = services.clock()
    superseded = {e["analysis_id"] for e in repo.events(conn) if e["event"] == "clarify.answered"}
    decided = {r["analysis_id"] for r in conn.execute("SELECT analysis_id FROM decisions")}
    items = []
    for a in repo.list_analyses(conn):
        contract, details = a["contract"], a["details"]
        decision = a["decision"]
        needs = decision in ("clarify", "escalate") or a["human_approval_required"]
        if not needs or a["analysis_id"] in superseded:
            continue
        group = access.group_of({**contract, "details": details})
        if not access.can_view_analysis(user, a, group):
            continue
        is_decided = a["analysis_id"] in decided
        if status == "open" and is_decided:
            continue
        if status in ("clarify", "escalate") and decision != status:
            continue
        if status == "approval" and not a["human_approval_required"]:
            continue
        ticket = repo.get_ticket(conn, a["ticket_id"]) or {}
        items.append(
            {
                "analysis_id": a["analysis_id"],
                "ticket_id": a["ticket_id"],
                "title": ticket.get("title"),
                "decision": decision,
                "priority": contract["priority"]["level"],
                "category": contract["classification"]["category"],
                "subcategory": contract["classification"]["subcategory"],
                "confidence": contract["classification"]["confidence"],
                "human_approval_required": a["human_approval_required"],
                "approval_reasons": details.get("approval_reasons", []),
                "assignment_group": group,
                "bundle_id": (details.get("escalation") or {}).get("bundle_id"),
                "created_at": a["created_at"],
                "age_s": round((now - repo.parse_iso(a["created_at"])).total_seconds(), 1),
                "decided": is_decided,
            }
        )
    items.sort(key=lambda i: (PRIORITY_RANK.get(i["priority"], 9), -i["age_s"]))
    size = int(backend_config().get("review_queue", {}).get("page_size", 20))
    return {
        "items": items[(page - 1) * size : page * size],
        "total": len(items),
        "page": page,
        "size": size,
    }


def _find(conn: Any, bundle_id: str) -> dict[str, Any] | None:
    rows = conn.execute(
        "SELECT analysis_id FROM analyses WHERE details_json LIKE ?", (f'%"{bundle_id}"%',)
    ).fetchall()
    for r in rows:
        a = repo.get_analysis(conn, r["analysis_id"])
        if a and escalate.find_escalation(a["details"], bundle_id):
            return a
    return None


@router.get("/escalations/{bundle_id}", summary="Diagnostic bundle (JSON or Markdown)")
def bundle(
    bundle_id: str,
    conn: DbDep,
    user: UserDep,
    fmt: Annotated[Literal["json", "md"], Query()] = "json",
) -> Any:
    a = _find(conn, bundle_id)
    if a is None:
        raise not_found(f"bundle {bundle_id}")
    full = repo.response_of(a).model_dump(mode="json")
    group = access.group_of(full)
    allowed = user.role in ("admin", "kb_owner") or access.can_act_on_analysis(user, a, group)
    if not allowed:
        raise access.deny(conn, user, f"bundle {bundle_id}")
    shown = access.filter_response(full, user)
    ticket = repo.get_ticket(conn, a["ticket_id"]) or {}
    out = escalate.build_bundle(shown, ticket, bundle_id, a["created_at"], masked=user.role == "l1")
    if out is None:  # pragma: no cover - _find already matched it
        raise not_found(f"bundle {bundle_id}")
    if fmt == "md":
        return PlainTextResponse(escalate.bundle_markdown(out), media_type="text/markdown")
    return out
