"""Assemble the API response: the 8 contract keys plus ``details`` (architecture §4).

Contract objects are built from the strict models in ``shared/schemas.py`` so an extra key can
never slip in; everything else goes into ``details``. ``view=contract`` is
``AnalysisResponse.to_contract()``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from backend.config import escalation_config
from backend.decision import IssueDecision, ticket_decision
from shared.constants import (
    KEDB_SOLUTION_MAX_CHARS,
    MAX_CLARIFYING_QUESTIONS,
    MAX_SIMILAR_INCIDENTS,
    PRIORITY_CODES,
)
from shared.schemas import (
    AnalysisResponse,
    Classification,
    ClassificationDetails,
    Details,
    EngineResult,
    Escalation,
    IssueResult,
    KBAlternative,
    KBArticle,
    KedbDetails,
    KedbMatch,
    Priority,
    PriorityDetails,
    RecommendedSolution,
    RedactionSummary,
    SecondaryIssue,
    SimilarDetails,
    SimilarIncident,
    TicketHit,
)

KBLookup = Callable[[str], KBArticle | None]


@dataclass
class AssemblyInput:
    ticket_id: str
    analysis_id: str
    er: EngineResult
    decisions: Sequence[IssueDecision]
    escalations: Sequence[Escalation | None]
    approval: tuple[bool, list[str]]
    redactions: RedactionSummary = field(default_factory=RedactionSummary)
    timings_ms: Mapping[str, int] = field(default_factory=dict)
    versions: Mapping[str, str] = field(default_factory=dict)
    intent: str | None = None


def most_severe(issues: Sequence[IssueResult]) -> IssueResult:
    """The issue whose priority sets the ticket priority (first one wins ties)."""
    return min(issues, key=lambda i: PRIORITY_CODES.index(i.priority.code))


def _clamp01(x: float) -> float:
    return round(min(1.0, max(0.0, float(x))), 2)


def classification(issue: IssueResult) -> Classification:
    c = issue.classification
    return Classification(
        category=c.category, subcategory=c.subcategory, confidence=_clamp01(c.confidence)
    )


def classification_details(issue: IssueResult) -> ClassificationDetails:
    c = issue.classification
    return ClassificationDetails(
        band=c.band,
        components=c.components,
        alternatives=c.alternatives,
        rationale=c.rationale,
        conflict=c.conflict,
    )


def priority(issue: IssueResult) -> Priority:
    return Priority(level=issue.priority.level, reason=issue.priority.reason)


def priority_details(issue: IssueResult) -> PriorityDetails:
    p = issue.priority
    return PriorityDetails(code=p.code, impact=p.impact, urgency=p.urgency, drivers=p.drivers)


def similar_incidents(hits: Sequence[TicketHit]) -> list[SimilarIncident]:
    return [
        SimilarIncident(
            incident_id=h.ticket_id, similarity_score=_clamp01(h.score), resolution=h.resolution
        )
        for h in hits[:MAX_SIMILAR_INCIDENTS]
    ]


def similar_details(hits: Sequence[TicketHit]) -> list[SimilarDetails]:
    return [
        SimilarDetails(
            incident_id=h.ticket_id,
            tier=h.tier or "SIMILAR",
            excerpt=h.excerpt,
            why=h.why,
            resolution_origin=h.resolution_origin,
            category=h.category,
            subcategory=h.subcategory,
        )
        for h in hits[:MAX_SIMILAR_INCIDENTS]
    ]


def _kedb_solution(issue: IssueResult, kb_lookup: KBLookup) -> str | None:
    top = issue.kb_matches[0]
    art = kb_lookup(top.kb_id)
    if art is not None and art.resolution.steps:
        return art.resolution.steps[0][:KEDB_SOLUTION_MAX_CHARS]
    for s in issue.recommendation.steps:
        if s.type == "resolution" and any(c.startswith(f"{top.kb_id}#") for c in s.citations):
            return s.text[:KEDB_SOLUTION_MAX_CHARS]
    return (top.excerpt or "")[:KEDB_SOLUTION_MAX_CHARS] or None


def kedb_match(issue: IssueResult, kb_lookup: KBLookup, tau_kb: float) -> KedbMatch:
    found = bool(issue.kb_matches) and issue.kb_relevance >= tau_kb
    if not found:
        return KedbMatch(found=False, error_code=None, solution=None)
    return KedbMatch(
        found=True,
        error_code=issue.kb_matches[0].kb_id,
        solution=_kedb_solution(issue, kb_lookup),
    )


def kedb_details(issue: IssueResult, tau_show: float) -> KedbDetails | None:
    if not issue.kb_matches or issue.kb_matches[0].score < tau_show:
        return None
    top = issue.kb_matches[0]
    return KedbDetails(
        kb_id=top.kb_id,
        title=top.title,
        score=round(top.score, 2),
        matched_error_code=top.matched_error_code,
        matched_field=top.matched_field,
        excerpt=top.excerpt,
        why=top.why,
        alternatives=[
            KBAlternative(kb_id=m.kb_id, score=round(m.score, 2)) for m in issue.kb_matches[1:3]
        ],
    )


def recommended_solution(issue: IssueResult) -> RecommendedSolution:
    rec = issue.recommendation
    return RecommendedSolution(steps=[s.text for s in rec.steps], source=rec.source)


def build(
    inp: AssemblyInput,
    kb_lookup: KBLookup,
    cfg: Mapping[str, Any] | None = None,
) -> AnalysisResponse:
    cfg = cfg if cfg is not None else escalation_config()
    th = cfg.get("thresholds", {})
    tau_kb, tau_show = float(th.get("tau_kb", 0.5)), float(th.get("tau_show", 0.35))
    er = inp.er
    primary, secondaries = er.issues[0], er.issues[1:]
    sev = most_severe(er.issues)
    escalation_required = any(d.decision == "escalate" for d in inp.decisions)
    questions: list[str] = []
    for d in inp.decisions:
        if d.decision == "clarify":
            questions += d.questions
    approval_required, approval_reasons = inp.approval

    secondary_issues = [
        SecondaryIssue(
            classification=classification(issue),
            priority=priority(issue),
            kedb_match=kedb_match(issue, kb_lookup, tau_kb),
            recommended_solution=recommended_solution(issue),
            decision=dec.decision,
            escalation=esc,
            classification_details=classification_details(issue),
            priority_details=priority_details(issue),
            kedb=kedb_details(issue, tau_show),
            steps=issue.recommendation.steps,
        )
        for issue, dec, esc in zip(secondaries, inp.decisions[1:], inp.escalations[1:], strict=True)
    ]
    details = Details(
        analysis_id=inp.analysis_id,
        decision=ticket_decision(inp.decisions),
        primary_decision=inp.decisions[0].decision,
        clarifying_questions=list(dict.fromkeys(questions))[:MAX_CLARIFYING_QUESTIONS],
        intent=inp.intent or primary.classification.intent,
        classification=classification_details(primary),
        priority=priority_details(sev),
        similar=similar_details(primary.similar),
        keyword_only=similar_details(primary.keyword_only),
        kedb=kedb_details(primary, tau_show),
        steps=primary.recommendation.steps,
        escalation=next((e for e in inp.escalations if e is not None), None),
        approval_reasons=approval_reasons,
        secondary_issues=secondary_issues,
        signals=er.signals,
        redactions=inp.redactions,
        degraded=list(er.degraded),
        timings_ms=dict(inp.timings_ms or er.timings_ms),
        versions={**er.versions, **inp.versions},
    )
    return AnalysisResponse(
        ticket_id=inp.ticket_id,
        classification=classification(primary),
        priority=priority(sev),
        similar_incidents=similar_incidents(primary.similar),
        kedb_match=kedb_match(primary, kb_lookup, tau_kb),
        recommended_solution=recommended_solution(primary),
        escalation_required=escalation_required,
        human_approval_required=approval_required,
        details=details,
    )
