"""Escalation routing and the diagnostic bundle (architecture §8.2).

Routing: the category's assignment group, overridden by E5 → security group, Database with an
admin-only KB article → DBA, and Network + E4 → the major-incident network group. Tier is L3 when a
hard rule (E4/E5) fired, otherwise L2.

The bundle is rebuilt on demand from the stored analysis (no extra table): the redacted ticket,
signals, classification components, priority drivers, reasons, top-5 similar incidents, top-3 KB
matches, attempted steps, remaining steps, group, tier and timestamps.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping
from typing import Any

from backend.config import escalation_config
from backend.decision import IssueDecision
from shared.schemas import Escalation, IssueResult


def assignment_group(
    issue: IssueResult, decision: IssueDecision, cfg: Mapping[str, Any] | None = None
) -> str:
    cfg = cfg if cfg is not None else escalation_config()
    routing = cfg.get("routing", {})
    category = issue.classification.category
    rules = set(decision.rules)
    if "E5" in rules:
        return str(routing.get("security_group", "SEC-OPS"))
    requires_admin = issue.recommendation.requires_admin or any(
        m.requires_admin for m in issue.kb_matches[:1]
    )
    if category == "Database" and requires_admin:
        return str(routing.get("database_admin_group", "DBA"))
    if category == "Network" and "E4" in rules:
        return str(routing.get("network_major_group", "NET-OPS-MAJOR"))
    return str(
        routing.get("by_category", {}).get(category, routing.get("default_group", "SERVICE-DESK"))
    )


def route(
    issue: IssueResult,
    decision: IssueDecision,
    issue_idx: int = 0,
    cfg: Mapping[str, Any] | None = None,
    new_id: Callable[[], str] = lambda: str(uuid.uuid4()),
) -> Escalation | None:
    """An ``Escalation`` for issues the gate escalated, else ``None``."""
    if decision.decision != "escalate":
        return None
    cfg = cfg if cfg is not None else escalation_config()
    tiers = cfg.get("tiers", {})
    tier = tiers.get("hard", "L3") if decision.hard else tiers.get("default", "L2")
    return Escalation(
        reasons=decision.reasons,
        assignment_group=assignment_group(issue, decision, cfg),
        tier=tier,
        bundle_id=new_id(),
        issue_idx=issue_idx,
    )


# =============================================================================================
# Diagnostic bundle
# =============================================================================================
def find_escalation(details: Mapping[str, Any], bundle_id: str) -> dict[str, Any] | None:
    escs = [details.get("escalation")] + [
        s.get("escalation") for s in details.get("secondary_issues", [])
    ]
    for esc in escs:
        if esc and esc.get("bundle_id") == bundle_id:
            return dict(esc)
    return None


def build_bundle(
    response: Mapping[str, Any],
    ticket: Mapping[str, Any],
    bundle_id: str,
    created_at: str,
    *,
    masked: bool = False,
) -> dict[str, Any] | None:
    """Bundle for ``bundle_id`` from a (role-filtered) full response, or ``None`` if absent."""
    details = response.get("details", {})
    esc = find_escalation(details, bundle_id)
    if esc is None:
        return None
    idx = int(esc.get("issue_idx", 0))
    secondaries = details.get("secondary_issues", [])
    issue = secondaries[idx - 1] if 0 < idx <= len(secondaries) else None
    src: Mapping[str, Any] = issue or response
    det: Mapping[str, Any] = issue or details
    steps = det.get("steps", [])
    kedb = det.get("kedb")
    kb_matches = []
    if kedb:
        kb_matches.append(
            {k: kedb.get(k) for k in ("kb_id", "title", "score", "excerpt", "why", "matched_field")}
        )
        kb_matches += [dict(a) for a in kedb.get("alternatives", [])[:2]]
    cls_details = (
        details.get("classification") if issue is None else issue.get("classification_details")
    )
    pri_details = details.get("priority") if issue is None else issue.get("priority_details")
    return {
        "bundle_id": bundle_id,
        "ticket_id": response.get("ticket_id"),
        "title": ticket.get("title"),
        "description_redacted": ticket.get("description_redacted"),
        "signals": details.get("signals"),
        "classification": src.get("classification"),
        "classification_components": (cls_details or {}).get("components", {}),
        "priority": src.get("priority"),
        "priority_drivers": (pri_details or {}).get("drivers", []),
        "assignment_group": esc.get("assignment_group"),
        "tier": esc.get("tier"),
        "reasons": esc.get("reasons", []),
        "similar_incidents": list(response.get("similar_incidents", []))[:5],
        "kb_matches": kb_matches[:3],
        "kedb": kedb,
        "attempted_steps": [s["text"] for s in steps if s.get("already_attempted")],
        "remaining_steps": [s["text"] for s in steps if not s.get("already_attempted")],
        "created_at": created_at,
        "masked": masked,
    }


def bundle_markdown(bundle: Mapping[str, Any]) -> str:
    def bullets(items: list[str]) -> list[str]:
        return [f"- {i}" for i in items] or ["- none"]

    cls = bundle.get("classification") or {}
    pri = bundle.get("priority") or {}
    lines = [
        f"# Escalation bundle {bundle['bundle_id']}",
        "",
        f"**Ticket:** {bundle.get('ticket_id')} - {bundle.get('title')}",
        f"**Route to:** {bundle.get('assignment_group')} ({bundle.get('tier')})",
        f"**Classification:** {cls.get('category')} / {cls.get('subcategory')} "
        f"(confidence {cls.get('confidence')})",
        f"**Priority:** {pri.get('level')} - {pri.get('reason')}",
        f"**Created:** {bundle.get('created_at')}",
        "",
        "## Reasons",
        *bullets([f"{r['rule']}: {r['text']}" for r in bundle.get("reasons", [])]),
        "",
        "## Ticket (redacted)",
        str(bundle.get("description_redacted") or ""),
        "",
        "## Similar incidents",
        *bullets(
            [
                f"{s['incident_id']} ({s['similarity_score']}): "
                f"{s.get('resolution') or 'no resolution'}"
                for s in bundle.get("similar_incidents", [])
            ]
        ),
        "",
        "## KB matches",
        *bullets([f"{k.get('kb_id')} ({k.get('score')})" for k in bundle.get("kb_matches", [])]),
        "",
        "## Attempted steps",
        *bullets(list(bundle.get("attempted_steps", []))),
        "",
        "## Remaining KB steps",
        *bullets(list(bundle.get("remaining_steps", []))),
    ]
    return "\n".join(lines) + "\n"
