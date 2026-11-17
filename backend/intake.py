"""Intake: validation, label-leak guard and ID assignment (architecture §4.2, §8.1).

* ``metadata.expected_*`` (and any top-level ``expected_*``) is dropped here, so gold labels can
  never reach the engine (CLAUDE.md rule 7).
* Incoming ``category`` / ``subcategory`` / ``priority`` are kept only as ``reporter_*`` fields and
  are never used for prediction.
* Tickets without an id get ``<prefix>-<yyyymmdd>-<seq>``.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from backend.config import backend_config
from backend.errors import bad_request
from shared.schemas import TicketIn

GOLD_PREFIX = "expected_"
_SPACES = re.compile(r"\s+")


class IntakeTicket(BaseModel):
    """A validated ticket. Holds raw text until the pipeline redacts it; never persisted as is."""

    ticket_id: str
    title: str
    description: str
    reporter_category: str | None = None
    reporter_subcategory: str | None = None
    reporter_priority: str | None = None
    application: str | None = None
    environment: str | None = None
    location: str | None = None
    assignment_group: str | None = None
    restricted: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


def strip_gold(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    return {k: v for k, v in (metadata or {}).items() if not str(k).startswith(GOLD_PREFIX)}


def _limits(cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    cfg = cfg if cfg is not None else backend_config()
    return dict(cfg.get("intake", {}))


def new_ticket_id(now: datetime, count_with_prefix: Callable[[str], int], prefix: str) -> str:
    stem = f"{prefix}-{now:%Y%m%d}-"
    return f"{stem}{count_with_prefix(stem) + 1:04d}"


def validate(
    ticket_in: TicketIn | Mapping[str, Any],
    *,
    assign_id: Callable[[], str],
    cfg: Mapping[str, Any] | None = None,
    check_description_max: bool = True,
) -> IntakeTicket:
    lim = _limits(cfg)
    if isinstance(ticket_in, Mapping):
        ticket_in = TicketIn.model_validate(
            {k: v for k, v in ticket_in.items() if not str(k).startswith(GOLD_PREFIX)}
        )
    title = _SPACES.sub(" ", ticket_in.title or "").strip()
    description = (ticket_in.description or "").strip()
    if not lim.get("title_min", 5) <= len(title) <= lim.get("title_max", 200):
        raise bad_request(
            f"title must be {lim.get('title_min', 5)}-{lim.get('title_max', 200)} characters"
        )
    d_min, d_max = lim.get("description_min", 10), lim.get("description_max", 5000)
    if len(description) < d_min or (check_description_max and len(description) > d_max):
        raise bad_request(f"description must be {d_min}-{d_max} characters")
    ticket_id = (ticket_in.id or "").strip() or assign_id()
    if len(ticket_id) > 64 or not re.fullmatch(r"[\w.:-]+", ticket_id):
        raise bad_request("id may only contain letters, digits, '.', ':', '_' and '-' (max 64)")
    return IntakeTicket(
        ticket_id=ticket_id,
        title=title,
        description=description,
        reporter_category=ticket_in.category,
        reporter_subcategory=ticket_in.subcategory,
        reporter_priority=ticket_in.priority,
        application=ticket_in.application,
        environment=ticket_in.environment,
        location=ticket_in.location,
        assignment_group=ticket_in.assignment_group,
        restricted=ticket_in.restricted,
        metadata=strip_gold(ticket_in.metadata),
    )
