"""``POST /events``, ``GET /metrics/mttr``, ``GET /metrics/summary``, ``GET /trial/tickets``."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import APIRouter, Query

from backend import access, mttr, repo
from backend.deps import DbDep, ServicesDep, UserDep
from backend.errors import bad_request
from shared.constants import UI_EVENT_TYPES
from shared.schemas import Event

router = APIRouter(tags=["metrics"])
MANUAL_EVENTS = frozenset({"manual.started", "manual.decision"})


@router.post("/events", summary="UI timing events (ui.viewed, manual.started, manual.decision)")
def post_event(body: Event, services: ServicesDep, conn: DbDep, user: UserDep) -> dict[str, bool]:
    if body.event not in UI_EVENT_TYPES:
        raise bad_request(f"the UI may only post {', '.join(UI_EVENT_TYPES)}")
    if body.event in MANUAL_EVENTS:
        access.require(conn, user, access.TRIAL_ROLES, "manual-mode trial")
        if body.mode != "manual":
            raise bad_request(f"{body.event} needs mode=manual")
        if not body.analysis_id:
            raise bad_request(f"{body.event} needs analysis_id (the manual session id)")
    repo.record_event(
        conn, ticket_id=body.ticket_id, analysis_id=body.analysis_id, event=body.event,
        actor=user.username, mode=body.mode, ts_utc=repo.iso(services.clock()),
    )  # fmt: skip
    return {"ok": True}


@router.get("/metrics/mttr", summary="TTR / TTD / MTTR grouped by category|priority|mode|decision")
def metrics_mttr(
    conn: DbDep,
    user: UserDep,
    group_by: Annotated[Literal["category", "priority", "mode", "decision"], Query()] = "category",
    metric: Annotated[Literal["ttr", "ttd", "mttr"], Query()] = "mttr",
    mode: Annotated[Literal["assisted", "manual"] | None, Query()] = None,
    date_from: Annotated[str | None, Query(alias="from")] = None,
    date_to: Annotated[str | None, Query(alias="to")] = None,
) -> dict[str, Any]:
    return mttr.metrics_mttr(
        mttr.timing_rows(conn),
        group_by=group_by,
        metric=metric,
        mode=mode,
        date_from=date_from,
        date_to=date_to,
    )


@router.get("/metrics/summary", summary="Counts, feedback rates, degraded rates, TTD by mode")
def metrics_summary(services: ServicesDep, conn: DbDep, user: UserDep) -> dict[str, Any]:
    return mttr.summary(conn, services.settings.tickets_enriched_path)


@router.get("/trial/tickets", summary="Trial tickets without gold labels")
def trial_tickets(services: ServicesDep, conn: DbDep, user: UserDep) -> dict[str, Any]:
    access.require(conn, user, access.TRIAL_ROLES, "manual-mode trial")
    return {"items": mttr.trial_tickets(services.settings.trial_tickets_path)}
