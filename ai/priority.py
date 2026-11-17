"""Priority engine (architecture §7.4): impact × urgency matrix, baseline floor, KB hint.

    if intent in {how_to, service_request}: p = p_matrix
    elif no impact/urgency signal:          p = p_baseline
    else:                                   p = most_severe(p_matrix, p_baseline,
                                                            p_kb_hint if kb_relevance ≥ τ_kb)

Impact:  I1 ≥10 users, site/org scope, production down, data loss or security
         I2 multiple/team scope or 2–9 users · I3 single user (default)
Urgency: U1 deadline today, month-end, fully blocked, or security
         U2 degraded or a deadline within days (default for incidents) · U3 how-to/request

`reason` = the top 1–2 drivers joined with "; ", or "Baseline for <subcategory>".
"""

from __future__ import annotations

from typing import Any

from ai.config import ai_config
from ai.signals import Signals
from ai.types import ClassificationResult, PriorityResult

MATRIX: dict[tuple[int, int], str] = {
    (1, 1): "P1", (1, 2): "P2", (1, 3): "P3",
    (2, 1): "P2", (2, 2): "P3", (2, 3): "P3",
    (3, 1): "P3", (3, 2): "P3", (3, 3): "P4",
}  # fmt: skip

_RANK = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}


def severity(code: str) -> int:
    return _RANK.get(code, 4)


def most_severe(*codes: str | None) -> str:
    present = [c for c in codes if c in _RANK]
    return min(present, key=severity) if present else "P4"


def impact_of(signals: Signals) -> tuple[int | None, str | None]:
    """Return `(impact, driver)`; impact None when there is no impact signal."""
    users = signals.affected_users
    ev = signals.scope_evidence
    if signals.security_flags:
        return 1, f"security: {signals.security_flags[0]}"
    if (users is not None and users >= 10) or signals.scope in ("site", "org"):
        return 1, f"{ev or signals.scope} affected"
    if signals.business_critical and signals.environment == "production":
        return 1, "production system affected"
    if (users is not None and users >= 2) or signals.scope in ("multiple", "team"):
        return 2, f"{ev or signals.scope} affected"
    if users == 1:
        return 3, "single user"
    return None, None


def urgency_of(signals: Signals, intent: str) -> tuple[int | None, str | None]:
    if signals.security_flags:
        return 1, None  # the security driver is already reported as impact
    if signals.urgency_level is not None:
        terms = ", ".join(signals.urgency_terms[:2])
        return signals.urgency_level, terms
    if intent in ("how_to", "service_request"):
        return 3, None
    return None, None


def compute(
    signals: Signals,
    classification: ClassificationResult,
    *,
    kb_id: str | None = None,
    kb_hint: str | None = None,
    kb_relevance: float = 0.0,
    cfg: dict[str, Any] | None = None,
) -> PriorityResult:
    cfg = cfg if cfg is not None else ai_config()
    pcfg = cfg.get("priority", {})
    levels: dict[str, str] = pcfg.get(
        "level", {"P1": "Critical", "P2": "High", "P3": "Medium", "P4": "Low"}
    )
    tau_kb = float(cfg.get("retrieval", {}).get("kb", {}).get("tau_kb", 0.50))
    max_drivers = int(pcfg.get("max_drivers_in_reason", 2))

    baseline = pcfg.get("baseline", {}).get(classification.category, "P3")
    sub = classification.subcategory
    intent = classification.intent

    impact, impact_driver = impact_of(signals)
    urgency, urgency_driver = urgency_of(signals, intent)
    drivers: list[str] = []
    reason_parts: list[str] = []
    if impact_driver:
        drivers.append(f"{impact_driver} → I{impact}")
        reason_parts.append(impact_driver)
    if urgency_driver:
        drivers.append(f"{urgency_driver} → U{urgency}")
        reason_parts.append(urgency_driver)

    hint = kb_hint if (kb_hint in _RANK and kb_relevance >= tau_kb) else None

    if intent in ("how_to", "service_request"):
        i = impact or 3
        u = urgency or 3
        code = MATRIX[(i, u)]
    elif impact is None and urgency is None:
        code = baseline
        drivers.append(f"baseline {baseline} for {sub}")
        return PriorityResult(
            code=code,
            level=levels.get(code, code),
            reason=f"Baseline for {sub}",
            impact=None,
            urgency=None,
            drivers=drivers,
        )
    else:
        i = impact or 3
        u = urgency or 2
        p_matrix = MATRIX[(i, u)]
        code = most_severe(p_matrix, baseline, hint)
        if severity(code) < severity(p_matrix):
            # the floor raised the matrix result: name the source (baseline wins ties)
            if severity(baseline) == severity(code):
                floor = f"floor: {sub} baseline"
            else:
                floor = f"floor: {kb_id} hint {hint}"
            drivers.append(floor)
            reason_parts.append(floor)
        impact, urgency = i, u

    reason = "; ".join(reason_parts[:max_drivers]) or f"Baseline for {sub}"
    return PriorityResult(
        code=code,
        level=levels.get(code, code),
        reason=reason,
        impact=impact,
        urgency=urgency,
        drivers=drivers,
    )


def ticket_priority(per_issue: list[PriorityResult]) -> PriorityResult:
    """The ticket priority is the most severe across issues (ties: first issue)."""
    if not per_issue:
        raise ValueError("no issues")
    best = per_issue[0]
    for p in per_issue[1:]:
        if severity(p.code) < severity(best.code):
            best = p
    return best
