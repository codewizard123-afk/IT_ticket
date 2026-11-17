"""Analyst feedback, final decisions and their effects (FR-06, architecture §8.8).

* ``kb_prior``: ``fb(kb) = (useful − incorrect) / (useful + incorrect + smoothing)``, clipped to
  ±clip, passed to P2 as ``ctx.feedback_prior`` on the next request.
* Classification corrections are stored as feedback rows; P2 reads them via
  ``GET /internal/corrections`` (few-shot examples).
* A decision of ``resolved`` with notes calls ``TicketStore.mark_verified()``: the ticket becomes an
  ``analyst_verified`` historical incident that later analyses can cite.
* Free text (comments, notes, corrected values) is redacted before it is stored or embedded.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import uuid
from collections.abc import Mapping
from typing import Any

from backend import access, audit, repo
from backend.config import backend_config
from backend.errors import ApiError, not_found
from backend.services import Services
from shared.schemas import Decision, Feedback, User

log = logging.getLogger(__name__)
_KB_REF = re.compile(r"^(KB-\d+)")


def _cfg(cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict((cfg if cfg is not None else backend_config()).get("feedback", {}))


def kb_of(target_ref: str | None) -> str | None:
    m = _KB_REF.match(target_ref or "")
    return m.group(1) if m else None


def kb_prior(conn: sqlite3.Connection, cfg: Mapping[str, Any] | None = None) -> dict[str, float]:
    c = _cfg(cfg)
    smoothing, clip = float(c.get("smoothing", 5)), float(c.get("clip", 0.5))
    targets = tuple(c.get("prior_targets", ["kedb_match"]))
    marks = ",".join("?" for _ in targets)
    counts: dict[str, list[int]] = {}
    for row in conn.execute(
        f"SELECT target_ref, verdict FROM feedback WHERE target_type IN ({marks})", targets
    ):
        kb = kb_of(row["target_ref"])
        if kb is None:
            continue
        useful_incorrect = counts.setdefault(kb, [0, 0])
        if row["verdict"] == "useful":
            useful_incorrect[0] += 1
        elif row["verdict"] == "incorrect":
            useful_incorrect[1] += 1
    prior = {}
    for kb, (u, i) in counts.items():
        if u or i:
            prior[kb] = round(max(-clip, min(clip, (u - i) / (u + i + smoothing))), 4)
    return prior


def _redact(services: Services, text: str | None) -> str | None:
    return services.redact(text).text if text else text


def _visible_analysis(conn: sqlite3.Connection, user: User, analysis_id: str) -> dict[str, Any]:
    a = repo.get_analysis(conn, analysis_id)
    if a is None:
        raise not_found(f"analysis {analysis_id}")
    full = repo.response_of(a).model_dump(mode="json")
    if not access.can_act_on_analysis(user, a, access.group_of(full)):
        raise access.deny(conn, user, f"analysis {analysis_id}")
    return a


def record_feedback(services: Services, conn: sqlite3.Connection, fb: Feedback, user: User) -> str:
    a = _visible_analysis(conn, user, fb.analysis_id)
    issues = 1 + len((a["details"] or {}).get("secondary_issues", []))
    if not 0 <= fb.issue_idx < issues:
        raise ApiError(400, "validation", f"issue_idx must be 0..{issues - 1}")
    row = {
        "feedback_id": str(uuid.uuid4()),
        **fb.model_dump(),
        "corrected_value": _redact(services, fb.corrected_value),
        "comment": _redact(services, fb.comment),
        "user": user.username,
        "created_at": repo.iso(services.clock()),
    }
    conn.execute(
        "INSERT INTO feedback (feedback_id, analysis_id, issue_idx, target_type, target_ref, "
        "verdict, corrected_value, comment, user, created_at) VALUES (:feedback_id, "
        ":analysis_id, :issue_idx, :target_type, :target_ref, :verdict, :corrected_value, "
        ":comment, :user, :created_at)",
        row,
    )
    audit.append(
        conn, actor=user.username, action="feedback.created", entity="analysis",
        entity_id=fb.analysis_id, payload=row,
    )  # fmt: skip
    return str(row["feedback_id"])


def corrections(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Classification corrections with the redacted ticket text, newest first (for P2)."""
    rows = conn.execute(
        "SELECT f.feedback_id, f.analysis_id, f.issue_idx, f.corrected_value, f.comment, "
        "f.created_at, a.contract_json, t.title, t.description_redacted "
        "FROM feedback f JOIN analyses a ON a.analysis_id = f.analysis_id "
        "LEFT JOIN tickets t ON t.ticket_id = a.ticket_id "
        "WHERE f.target_type = 'classification' AND f.corrected_value IS NOT NULL "
        "ORDER BY f.created_at DESC"
    ).fetchall()
    import json

    out = []
    for r in rows:
        predicted = json.loads(r["contract_json"])["classification"]
        out.append(
            {
                "feedback_id": r["feedback_id"],
                "analysis_id": r["analysis_id"],
                "issue_idx": r["issue_idx"],
                "title": r["title"],
                "description": r["description_redacted"],
                "predicted": {k: predicted[k] for k in ("category", "subcategory")},
                "corrected_value": r["corrected_value"],
                "comment": r["comment"],
                "created_at": r["created_at"],
            }
        )
    return out


def ticket_for(conn: sqlite3.Connection, analysis_id: str) -> str | None:
    """Ticket of an analysis, or of a manual-mode session (from its ``manual.started`` event)."""
    a = repo.get_analysis(conn, analysis_id)
    if a is not None:
        return str(a["ticket_id"])
    evs = repo.events(conn, analysis_id=analysis_id)
    return str(evs[0]["ticket_id"]) if evs else None


def _overrides(analysis: Mapping[str, Any], dec: Decision) -> list[tuple[str, str, str]]:
    contract = analysis["contract"]
    current = {
        "category": contract["classification"]["category"],
        "subcategory": contract["classification"]["subcategory"],
        "priority": contract["priority"]["level"],
    }
    wanted = {
        "category": dec.final_category,
        "subcategory": dec.final_subcategory,
        "priority": dec.final_priority,
    }
    return [(f, current[f], str(v)) for f, v in wanted.items() if v and v != current[f]]


def record_decision(
    services: Services, conn: sqlite3.Connection, dec: Decision, user: User
) -> None:
    ticket_id = ticket_for(conn, dec.analysis_id)
    if ticket_id is None:
        raise not_found("unknown analysis (manual mode: post manual.started first)")
    analysis = repo.get_analysis(conn, dec.analysis_id)
    if analysis is not None:
        _visible_analysis(conn, user, dec.analysis_id)
    now = repo.iso(services.clock())
    notes = _redact(services, dec.resolution_notes)
    row = {**dec.model_dump(), "resolution_notes": notes, "user": user.username, "decided_at": now}
    conn.execute(
        "INSERT OR REPLACE INTO decisions (analysis_id, mode, final_decision, final_category, "
        "final_subcategory, final_priority, kb_or_incident_used, resolution_notes, user, "
        "decided_at) VALUES (:analysis_id, :mode, :final_decision, :final_category, "
        ":final_subcategory, :final_priority, :kb_or_incident_used, :resolution_notes, :user, "
        ":decided_at)",
        row,
    )
    if analysis is not None:
        for field, old, new in _overrides(analysis, dec):
            conn.execute(
                "INSERT INTO overrides (analysis_id, field, old_value, new_value, user, ts_utc) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (dec.analysis_id, field, old, new, user.username, now),
            )
    audit.append(
        conn, actor=user.username, action="decision.submitted", entity="analysis",
        entity_id=dec.analysis_id, payload=row,
    )  # fmt: skip

    def event(name: str) -> None:
        repo.record_event(
            conn, ticket_id=ticket_id, analysis_id=dec.analysis_id, event=name,
            actor=user.username, mode=dec.mode, ts_utc=repo.iso(services.clock()),
        )  # fmt: skip

    if dec.mode == "assisted":
        event("decision.submitted")
    if dec.final_decision == "resolved":
        event("ticket.resolved")
        conn.execute(
            "UPDATE tickets SET status='resolved', updated_at=? WHERE ticket_id=?", (now, ticket_id)
        )
        if notes and analysis is not None:
            _mark_verified(services, ticket_id, notes, user)


def _mark_verified(services: Services, ticket_id: str, notes: str, user: User) -> None:
    if services.ticket_store is None:
        return
    try:
        services.ticket_store.mark_verified(ticket_id, notes, by=user.username)
    except Exception as exc:  # noqa: BLE001 - the decision is recorded either way
        log.warning("mark_verified failed for %s: %s", ticket_id, type(exc).__name__)
