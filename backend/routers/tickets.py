"""FR-07 ticket search and ticket detail (architecture §10).

``/tickets/search`` delegates ranking to P2's ``ai.ticket_search.search_tickets`` with the caller's
role filters (retrieval-time RBAC), then post-filters, pages and counts. A filter-only listing
(empty ``q``) uses the store's ``list(where=, limit=)`` directly.
"""

from __future__ import annotations

from datetime import UTC, datetime, time
from typing import Annotated, Any

from fastapi import APIRouter, Query

from backend import access, audit, repo
from backend.config import backend_config
from backend.deps import DbDep, ServicesDep, UserDep
from backend.errors import bad_request, not_found
from shared.constants import LEVEL_PRIORITY
from shared.schemas import TicketHit, TicketRecord, User

router = APIRouter(prefix="/tickets", tags=["tickets"])
EQ_PARAMS = (
    "category", "subcategory", "priority", "status", "assignment_group", "application",
    "environment", "source",
)  # fmt: skip


def _epoch(day: str, end: bool) -> int:
    try:
        d = datetime.strptime(day[:10], "%Y-%m-%d").date()
    except ValueError as exc:
        raise bad_request(f"dates must be YYYY-MM-DD, got {day!r}") from exc
    return int(datetime.combine(d, time.max if end else time.min, tzinfo=UTC).timestamp())


def build_filters(params: dict[str, Any], user: User) -> dict[str, Any]:
    filters: dict[str, Any] = {}
    for key in EQ_PARAMS:
        val = params.get(key)
        if val:
            if key == "priority":
                val = LEVEL_PRIORITY.get(val, val)
            filters[key] = {"$eq": val}
    if params.get("date_from"):
        filters["date_from"] = _epoch(params["date_from"], end=False)
    if params.get("date_to"):
        filters["date_to"] = _epoch(params["date_to"], end=True)
    filters.update(access.retrieval_filters(user))
    return filters


def _record_hit(rec: TicketRecord) -> TicketHit:
    return TicketHit(
        ticket_id=rec.ticket_id, score=0.0, title=rec.title, excerpt=rec.description[:200],
        category=rec.category, subcategory=rec.subcategory, priority=rec.priority,
        status=rec.status, resolution=rec.resolution, resolution_origin=rec.resolution_origin,
        collection=rec.collection,
        metadata={
            "restricted": rec.restricted, "assignment_group": rec.assignment_group,
            "application": rec.application, "environment": rec.environment,
            "source": rec.source, "opened_at": rec.opened_at, "opened_ts": rec.opened_ts,
            "derived_fields": rec.derived_fields,
        },
    )  # fmt: skip


def _hits(services: Any, q: str, filters: dict[str, Any], user: User, k: int) -> list[TicketHit]:
    from ai.ticket_search import build_where, role_where, search_tickets  # P2 public API

    ticket_store, _ = services.stores()
    if q.strip():
        return search_tickets(q, filters, k, user.role, ticket_store)
    lister = getattr(ticket_store, "list", None)
    if lister is None:
        return []
    where = build_where(role_where(filters, user.role))
    return [_record_hit(r) for r in lister(where=where, limit=k)]


@router.get("/search", summary="FR-07: hybrid search + filters, role-filtered and paged")
def search(
    services: ServicesDep,
    conn: DbDep,
    user: UserDep,
    q: str = "",
    category: str | None = None,
    subcategory: str | None = None,
    priority: str | None = None,
    status: str | None = None,
    assignment_group: str | None = None,
    application: str | None = None,
    environment: str | None = None,
    source: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    page: Annotated[int, Query(ge=1)] = 1,
    size: Annotated[int | None, Query(ge=1)] = None,
) -> dict[str, Any]:
    cfg = backend_config().get("search", {})
    size = min(size or int(cfg.get("default_size", 20)), int(cfg.get("max_size", 200)))
    params = {k: v for k, v in locals().items() if k in (*EQ_PARAMS, "date_from", "date_to")}
    filters = build_filters(params, user)
    hits = _hits(services, q, filters, user, int(cfg.get("max_results", 1000)))
    visible = [h for h in hits if access.hit_visible(user, h)]
    if len(visible) < len(hits):
        audit.append(
            conn, actor=user.username, action="access.denied", entity="tickets",
            entity_id="search",
            payload={"what": "restricted hits", "rows": len(hits) - len(visible)},
        )  # fmt: skip
    items = [h.model_dump(mode="json") for h in visible[(page - 1) * size : page * size]]
    return {"items": items, "total": len(visible), "page": page, "size": size}


@router.get("/{ticket_id}", summary="One ticket (historical, verified or analysed here)")
def get_ticket(ticket_id: str, services: ServicesDep, conn: DbDep, user: UserDep) -> dict[str, Any]:
    ticket_store, _ = services.stores()
    recs = ticket_store.get([ticket_id])
    if recs:
        rec = recs[0]
        if not access.ticket_visible(user, rec.restricted, rec.assignment_group):
            raise access.deny(conn, user, f"ticket {ticket_id}")
        return rec.model_dump(mode="json")
    row = repo.get_ticket(conn, ticket_id)
    if row is None:
        raise not_found(f"ticket {ticket_id}")
    if not access.ticket_visible(user, bool(row["restricted"]), row["assignment_group"]):
        raise access.deny(conn, user, f"ticket {ticket_id}")
    return {
        "ticket_id": ticket_id,
        "title": row["title"],
        "description": row["description_redacted"],
        "status": row["status"],
        "application": row["application"],
        "environment": row["environment"],
        "location": row["location"],
        "assignment_group": row["assignment_group"],
        "restricted": bool(row["restricted"]),
        "opened_at": row["created_at"],
        "collection": "app",
    }
