"""Orchestrator (architecture §8.1): intake → redact → engine → decision → assemble → persist.

The response returned here is the *unfiltered* full view; routers apply the role filter
(``access.filter_response``) per viewer, so the stored record and the audit trail stay complete.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import time
import uuid
from collections.abc import Mapping
from typing import Any

from backend import access, approval, assemble, audit, decision, escalate, feedback, llm_audit, repo
from backend.config import backend_config, escalation_config
from backend.errors import ApiError, bad_request, not_found
from backend.intake import IntakeTicket
from backend.services import Services
from shared.constants import PIPELINE_VERSION
from shared.schemas import (
    AnalysisResponse,
    ClassificationResult,
    EngineContext,
    EngineResult,
    IssueResult,
    PriorityResult,
    Recommendation,
    RedactionSummary,
    SanitizedTicket,
    Signals,
    TicketRecord,
    User,
)

log = logging.getLogger(__name__)
CLARIFY_MARK = "\n[Clarification] "
MAX_ROUNDS_SENTINEL = 99  # makes the gate skip its clarify step


def _ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fallback_result(ticket: SanitizedTicket, reason: str) -> EngineResult:
    """Used only if the engine raises: a no-evidence result the gate will escalate."""
    return EngineResult(
        signals=Signals(),
        issues=[
            IssueResult(
                text_span=f"{ticket.title}\n{ticket.description}",
                classification=ClassificationResult(
                    category="Inquiry/Help",
                    subcategory="Other",
                    confidence=0.0,
                    band="low",
                    rationale="engine unavailable",
                ),
                priority=PriorityResult(
                    code="P3",
                    level="Medium",
                    reason="Baseline (engine unavailable)",
                    impact=3,
                    urgency=2,
                ),
                recommendation=Recommendation(insufficient_evidence=True),
            )
        ],
        degraded=[reason],
        versions={"pipeline": PIPELINE_VERSION},
    )


def context_inputs(
    services: Services, conn: sqlite3.Connection, user: User
) -> tuple[dict[str, float], dict[str, Any]]:
    """``(feedback_prior, retrieval filters)`` for the engine context."""
    return feedback.kb_prior(conn), access.retrieval_filters(user)


def _evidence_rows(analysis_id: str, er: EngineResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, issue in enumerate(er.issues):
        for kind, hits in (("similar", issue.similar), ("keyword_only", issue.keyword_only)):
            for rank, h in enumerate(hits, 1):
                rows.append(
                    {
                        "analysis_id": analysis_id,
                        "issue_idx": idx,
                        "kind": kind,
                        "ref_id": h.ticket_id,
                        "rank": rank,
                        "score": h.score,
                        "tier": h.tier,
                        "excerpt": h.excerpt,
                        "matched_field": None,
                        "why": h.why,
                        "origin": h.resolution_origin,
                    }  # fmt: skip
                )
        for rank, m in enumerate(issue.kb_matches, 1):
            rows.append(
                {
                    "analysis_id": analysis_id,
                    "issue_idx": idx,
                    "kind": "kedb",
                    "ref_id": m.kb_id,
                    "rank": rank,
                    "score": m.score,
                    "tier": None,
                    "excerpt": m.excerpt,
                    "matched_field": m.matched_field,
                    "why": m.why,
                    "origin": "kedb",
                }  # fmt: skip
            )
        for rank, step in enumerate(issue.recommendation.steps, 1):
            for cite in step.citations:
                rows.append(
                    {
                        "analysis_id": analysis_id,
                        "issue_idx": idx,
                        "kind": "citation",
                        "ref_id": cite,
                        "rank": rank,
                        "score": None,
                        "tier": step.role,
                        "excerpt": step.text[:300],
                        "matched_field": step.type,
                        "why": None,
                        "origin": "kedb" if "#" in cite else "incident",
                    }  # fmt: skip
                )
    return rows


def _audit_payload(
    ticket: IntakeTicket,
    title: str,
    description: str,
    er: EngineResult,
    resp: AnalysisResponse,
    decisions: list[decision.IssueDecision],
    user: User,
    llm_calls: list[dict[str, Any]],
    clarification_round: int,
) -> dict[str, Any]:
    return {
        "ticket_id": ticket.ticket_id,
        "input_redacted": {"title": title, "description": description},
        "signals": er.signals.model_dump(mode="json"),
        "issues": [
            {
                "classification": i.classification.model_dump(mode="json"),
                "priority": i.priority.model_dump(mode="json"),
                "evidence": {
                    "similar": [
                        {k: getattr(h, k) for k in ("ticket_id", "score", "tier", "why")}
                        | {"origin": h.resolution_origin}
                        for h in i.similar
                    ],
                    "kedb": [
                        {"kb_id": m.kb_id, "score": m.score, "why": m.why} for m in i.kb_matches
                    ],
                },
                "steps": [s.model_dump(mode="json") for s in i.recommendation.steps],
                "decision": d.decision,
                "escalation_reasons": [r.model_dump(mode="json") for r in d.reasons],
            }
            for i, d in zip(er.issues, decisions, strict=True)
        ],
        "decision": resp.details.decision,
        "escalation_required": resp.escalation_required,
        "human_approval_required": resp.human_approval_required,
        "approval_reasons": resp.details.approval_reasons,
        "versions": resp.details.versions,
        "timings_ms": resp.details.timings_ms,
        "degraded": resp.details.degraded,
        "role": user.role,
        "clarification_round": clarification_round,
        "llm_calls": [
            {k: c.get(k) for k in ("purpose", "model", "tokens", "cache_hit", "status")}
            for c in llm_calls
        ],
    }


def _upsert_live(
    services: Services,
    ticket: IntakeTicket,
    title: str,
    desc: str,
    resp: AnalysisResponse,
    now_iso: str,
    ts: int,
) -> None:  # noqa: E501
    """Best effort: make the ticket visible to similar-ticket search and live clustering."""
    if services.ticket_store is None:
        return
    rec = TicketRecord(
        ticket_id=ticket.ticket_id,
        title=title,
        description=desc,
        category=resp.classification.category,
        subcategory=resp.classification.subcategory,
        priority=resp.details.priority.code,
        status="new",
        source="live",
        application=ticket.application,
        environment=ticket.environment,
        location=ticket.location,
        assignment_group=ticket.assignment_group,
        restricted=ticket.restricted,
        opened_at=now_iso,
        opened_ts=ts,
        collection="tickets_live",
    )
    try:
        services.ticket_store.upsert_live(rec)
    except Exception as exc:  # noqa: BLE001 - never fail an analysis on a store write
        log.warning("upsert_live failed for %s: %s", ticket.ticket_id, type(exc).__name__)


def analyze(
    services: Services,
    conn: sqlite3.Connection,
    ticket: IntakeTicket,
    user: User,
    *,
    clarification_round: int = 0,
) -> AnalysisResponse:
    t0 = time.perf_counter()
    ticket_store, kb_store = services.stores()
    now = services.clock()
    now_iso = repo.iso(now)
    analysis_id = str(uuid.uuid4())
    actor = user.username

    def event(name: str) -> None:
        ts = repo.iso(services.clock())
        repo.record_event(
            conn, ticket_id=ticket.ticket_id, analysis_id=analysis_id, event=name,
            actor=actor, mode="assisted", ts_utc=ts,
        )  # fmt: skip

    event("ticket.submitted")

    # 1. privacy: nothing below this line sees raw text
    t = time.perf_counter()
    raw = f"{ticket.title}\n{ticket.description}"
    red = services.redact(raw)
    title, _, description = red.text.partition("\n")
    existing = repo.get_ticket(conn, ticket.ticket_id)
    repo.upsert_ticket(
        conn,
        {
            "ticket_id": ticket.ticket_id,
            "title": title,
            "description_redacted": description,
            "reporter_category": ticket.reporter_category,
            "reporter_priority": ticket.reporter_priority,
            "application": ticket.application,
            "environment": ticket.environment,
            "location": ticket.location,
            "assignment_group": ticket.assignment_group,
            "restricted": int(ticket.restricted),
            "status": "analysed",
            "created_at": (existing or {}).get("created_at", now_iso),
            "updated_at": now_iso,
            "raw_hash": _sha256(raw),
        },  # fmt: skip
    )
    sanitize_ms = _ms(t)

    # 2. engine
    sanitized = SanitizedTicket(
        ticket_id=ticket.ticket_id,
        title=title,
        description=description,
        application=ticket.application,
        environment=ticket.environment,
        location=ticket.location,
        assignment_group=ticket.assignment_group,
        restricted=ticket.restricted,
        secret_detected=red.secret_detected,
    )
    prior, filters = context_inputs(services, conn, user)
    ctx = EngineContext(
        role=user.role,
        ticket_store=ticket_store,
        kb_store=kb_store,
        llm=services.llm,
        feedback_prior=prior,
        filters=filters,
        clarification_round=clarification_round,
    )
    engine_failed = False
    with llm_audit.capture() as llm_calls:
        try:
            er = services.engine.run(sanitized, ctx)
        except Exception as exc:  # noqa: BLE001 - the API must still answer (and escalate)
            log.error("engine failed for %s: %s", ticket.ticket_id, type(exc).__name__)
            er = fallback_result(sanitized, "engine_error")
            engine_failed = True
    er = access.filter_engine_result(er, user)

    # 3. decision gate, routing, approval, assembly
    t = time.perf_counter()
    gate = decision.GateInput(
        signals=er.signals,
        text=red.text,
        secret_detected=red.secret_detected,
        environment=ticket.environment,
        # a broken engine will not do better after a clarifying answer: escalate straight away
        clarification_round=MAX_ROUNDS_SENTINEL if engine_failed else clarification_round,
    )
    decisions = [decision.decide(issue, gate) for issue in er.issues]
    escalations = [
        escalate.route(issue, d, idx)
        for idx, (issue, d) in enumerate(zip(er.issues, decisions, strict=True))
    ]
    level = assemble.most_severe(er.issues).priority.level
    appr = approval.required(er.issues, decision.ticket_decision(decisions), level, er.degraded)
    decide_ms = _ms(t)
    timings = {**er.timings_ms, "sanitize": sanitize_ms, "decide": decide_ms}
    timings["total"] = _ms(t0)
    resp = assemble.build(
        assemble.AssemblyInput(
            ticket_id=ticket.ticket_id,
            analysis_id=analysis_id,
            er=er,
            decisions=decisions,
            escalations=escalations,
            approval=appr,
            redactions=RedactionSummary(
                count=len(red.items), types=sorted({i.type for i in red.items})
            ),
            timings_ms=timings,
            versions={
                "pipeline": PIPELINE_VERSION,
                "backend": str(backend_config().get("backend_version", "")),
            },
        ),
        services.kb_article,
        escalation_config(),
    )

    # 4. persist + audit
    versions = resp.details.versions
    repo.save_analysis(
        conn,
        {
            "analysis_id": analysis_id,
            "ticket_id": ticket.ticket_id,
            "created_at": now_iso,
            "user": actor,
            "role": user.role,
            "clarification_round": clarification_round,
            "contract_json": resp.to_contract().model_dump_json(),
            "details_json": resp.details.model_dump_json(),
            "signals_json": er.signals.model_dump_json(),
            "decision": resp.details.decision,
            "escalation_required": int(resp.escalation_required),
            "human_approval_required": int(resp.human_approval_required),
            "pipeline_version": versions.get("pipeline"),
            "embed_model": versions.get("embed_model"),
            "llm_models": ",".join(
                v for k, v in versions.items() if k.endswith("_model") and k != "embed_model"
            ),
            "degraded": ",".join(resp.details.degraded),
            "timings_json": _json(timings),
        },  # fmt: skip
        _evidence_rows(analysis_id, er),
        [
            {
                "call_id": str(uuid.uuid4()),
                "analysis_id": analysis_id,
                "purpose": c.get("purpose"),
                "model": c.get("model"),
                "prompt_hash": None,
                "tokens_in": c.get("tokens"),
                "tokens_out": None,
                "latency_ms": c.get("latency_ms"),
                "status": str(c.get("status") or c.get("error")),
                "cached": int(bool(c.get("cache_hit"))),
                "created_at": c.get("created_at"),
            }  # fmt: skip
            for c in llm_calls
        ],
    )
    audit.append(
        conn,
        actor=actor,
        action="analysis.created",
        entity="analysis",
        entity_id=analysis_id,
        payload=_audit_payload(
            ticket, title, description, er, resp, decisions, user, llm_calls, clarification_round
        ),
    )
    _upsert_live(services, ticket, title, description, resp, now_iso, int(now.timestamp()))
    event("analysis.completed")
    if resp.details.decision == "clarify":
        event("clarify.requested")
    return resp


def _json(obj: Mapping[str, Any]) -> str:
    import json

    return json.dumps(obj, sort_keys=True)


def clarify(
    services: Services,
    conn: sqlite3.Connection,
    ticket_id: str,
    analysis_id: str,
    answer: str,
    user: User,
) -> AnalysisResponse:
    """Append the answer to the description and re-run once (architecture §8.3)."""
    prev = repo.get_analysis(conn, analysis_id)
    if prev is None or prev["ticket_id"] != ticket_id:
        raise not_found("unknown analysis for this ticket")
    full = repo.response_of(prev).model_dump(mode="json")
    if not access.can_act_on_analysis(user, prev, access.group_of(full)):
        raise access.deny(conn, user, f"clarify {analysis_id}")
    max_rounds = int(escalation_config().get("thresholds", {}).get("max_clarify_rounds", 1))
    if int(prev["clarification_round"] or 0) >= max_rounds:
        raise bad_request("only one clarification round is allowed")
    answer = answer.strip()
    limit = int(backend_config().get("intake", {}).get("clarification_max", 1000))
    if not answer or len(answer) > limit:
        raise bad_request(f"answer must be 1-{limit} characters")
    row = repo.get_ticket(conn, ticket_id)
    if row is None:
        raise ApiError(404, "not_found", f"ticket {ticket_id}")
    ticket = IntakeTicket(
        ticket_id=ticket_id,
        title=row["title"],
        description=row["description_redacted"] + CLARIFY_MARK + answer,
        reporter_category=row["reporter_category"],
        reporter_priority=row["reporter_priority"],
        application=row["application"],
        environment=row["environment"],
        location=row["location"],
        assignment_group=row["assignment_group"],
        restricted=bool(row["restricted"]),
    )
    repo.record_event(
        conn, ticket_id=ticket_id, analysis_id=analysis_id, event="clarify.answered",
        actor=user.username, mode="assisted", ts_utc=repo.iso(services.clock()),
    )  # fmt: skip
    return analyze(
        services, conn, ticket, user, clarification_round=int(prev["clarification_round"] or 0) + 1
    )
