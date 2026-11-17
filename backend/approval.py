"""Human approval rule (architecture §8.5).

``human_approval_required`` is true if any of these hold, and each one is recorded in
``details.approval_reasons``: the decision is not recommend; the ticket priority is Critical or
High; a step has role L2 or the top KB article requires admin; a confidence band is not high;
a recommendation has no grounded source; the engine ran degraded. Either way the system never
executes anything; this flag only says whether an L1 analyst needs supervisor sign-off.
"""

from __future__ import annotations

from collections.abc import Sequence

from shared.schemas import DecisionKind, IssueResult, PriorityLevel

APPROVAL_LEVELS: frozenset[str] = frozenset({"Critical", "High"})


def approval_reasons(
    issues: Sequence[IssueResult],
    ticket_decision: DecisionKind,
    ticket_level: PriorityLevel,
    degraded: Sequence[str] = (),
) -> list[str]:
    reasons: list[str] = []
    if ticket_decision != "recommend":
        reasons.append(f"decision {ticket_decision}")
    if ticket_level in APPROVAL_LEVELS:
        reasons.append(f"priority {ticket_level}")
    if any(s.role == "L2" for i in issues for s in i.recommendation.steps):
        reasons.append("step role L2")
    for issue in issues:
        top = issue.kb_matches[0] if issue.kb_matches else None
        if top is not None and (top.requires_admin or issue.recommendation.requires_admin):
            reasons.append(f"{top.kb_id} requires admin")
        elif top is None and issue.recommendation.requires_admin:
            reasons.append("step requires admin")
    for band in dict.fromkeys(i.classification.band for i in issues):
        if band != "high":
            reasons.append(f"confidence band {band}")
    if any(i.recommendation.source == "None" for i in issues):
        reasons.append("source None")
    if degraded:
        reasons.append("degraded: " + ", ".join(degraded))
    return list(dict.fromkeys(reasons))


def required(
    issues: Sequence[IssueResult],
    ticket_decision: DecisionKind,
    ticket_level: PriorityLevel,
    degraded: Sequence[str] = (),
) -> tuple[bool, list[str]]:
    reasons = approval_reasons(issues, ticket_decision, ticket_level, degraded)
    return bool(reasons), reasons
