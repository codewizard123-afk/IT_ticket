"""``POST /analyze`` and ``POST /tickets/{ticket_id}/clarify`` (architecture §10)."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import APIRouter, Query
from pydantic import BaseModel

from backend import access, intake, pipeline, repo
from backend.config import backend_config
from backend.deps import DbDep, ServicesDep, UserDep
from shared.schemas import AnalysisResponse, TicketIn, User

router = APIRouter(tags=["analyze"])
View = Annotated[Literal["full", "contract"], Query()]


class ClarifyIn(BaseModel):
    analysis_id: str
    answer: str


def render(resp: AnalysisResponse, user: User, view: str) -> dict[str, Any]:
    """Role filter for display, then the requested view."""
    shown = access.filter_response(resp.model_dump(mode="json"), user)
    if view == "contract":
        shown.pop("details", None)
    return shown


@router.post("/analyze", summary="Analyse a ticket (contract + details)")
def analyze(
    ticket: TicketIn, services: ServicesDep, conn: DbDep, user: UserDep, view: View = "full"
) -> dict[str, Any]:
    prefix = str(backend_config().get("intake", {}).get("id_prefix", "INC"))
    validated = intake.validate(
        ticket,
        assign_id=lambda: intake.new_ticket_id(
            services.clock(), lambda p: repo.count_ticket_ids(conn, p), prefix
        ),
    )
    resp = pipeline.analyze(services, conn, validated, user)
    return render(resp, user, view)


@router.post("/tickets/{ticket_id}/clarify", summary="Answer the clarifying questions once")
def clarify(
    ticket_id: str,
    body: ClarifyIn,
    services: ServicesDep,
    conn: DbDep,
    user: UserDep,
    view: View = "full",
) -> dict[str, Any]:
    resp = pipeline.clarify(services, conn, ticket_id, body.analysis_id, body.answer, user)
    return render(resp, user, view)
