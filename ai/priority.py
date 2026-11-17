"""Priority engine: impact × urgency matrix with floors (architecture §7.4). Owner: P2.

```
if intent in {how_to, service_request}: p = p_matrix
elif no impact/urgency signal: p = p_baseline
else: p = most_severe(p_matrix, p_baseline, p_kb_hint if kb_relevance ≥ τ_kb)
```
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from ai.config import ai_config
from ai.signals import urgency_levels
from shared.constants import BASELINE_PRIORITY, PRIORITY_CODES, PRIORITY_LEVEL
from shared.schemas import ClassificationResult, PriorityResult, Signals

_SCOPE_TEXT = {
    "multiple": "multiple users affected",
    "team": "whole team affected",
    "site": "whole site affected",
    "org": "organisation-wide impact",
}


@dataclass
class _Driver:
    text: str  # e.g. "deadline today → U1" (stored in details.priority.drivers)
    reason: str  # e.g. "deadline today" (used in the contract reason)


def most_severe(codes: Iterable[str | None]) -> str:
    """``P1`` beats ``P2`` … ``P4``. Ignores ``None``; defaults to ``P4``."""
    valid = [c for c in codes if c in PRIORITY_CODES]
    return min(valid, key=PRIORITY_CODES.index) if valid else "P4"


def baseline(category: str) -> str:
    return BASELINE_PRIORITY.get(category, "P4")


def _impact(signals: Signals, text: str, cfg: dict[str, Any]) -> tuple[int, list[_Driver]]:
    users = signals.affected_users
    if users is not None and users >= int(cfg.get("impact_users_i1", 10)):
        d = f"{users} users affected"
        return 1, [_Driver(f"{d} → I1", d)]
    if signals.scope in ("site", "org"):
        d = _SCOPE_TEXT[signals.scope]
        return 1, [_Driver(f"{d} → I1", d)]
    if signals.security_flags:
        d = f"security concern ({signals.security_flags[0]})"
        return 1, [_Driver(f"{d} → I1", d)]
    if signals.business_critical and signals.environment == "production":
        d = "production system affected"
        return 1, [_Driver(f"{d} → I1", d)]
    if users is not None and users >= 2:
        d = f"{users} users affected"
        return 2, [_Driver(f"{d} → I2", d)]
    if signals.scope in ("multiple", "team"):
        d = _SCOPE_TEXT[signals.scope]
        return 2, [_Driver(f"{d} → I2", d)]
    pat = cfg.get("shared_resource")
    if pat and text and re.search(pat, text, re.IGNORECASE):
        d = "shared resource affected"
        return 2, [_Driver(f"{d} → I2", d)]
    return 3, []


def _urgency(signals: Signals, intent: str) -> tuple[int | None, list[_Driver]]:
    if signals.security_flags:
        d = "security incident"
        return 1, [_Driver(f"{d} → U1", d)]
    levels = urgency_levels()
    best: int | None = None
    drivers: list[_Driver] = []
    for term in signals.urgency_terms:
        lvl = levels.get(term, 2)
        if best is None or lvl < best:
            best = lvl
            drivers = [_Driver(f"{term} → U{lvl}", term)]
    if best is None and intent in ("how_to", "service_request"):
        return 3, []
    return best, drivers


def compute_priority(
    classification: ClassificationResult,
    signals: Signals,
    *,
    text: str = "",
    kb_id: str | None = None,
    kb_hint: str | None = None,
    kb_relevance: float = 0.0,
    cfg: dict[str, Any] | None = None,
) -> PriorityResult:
    """Priority for one issue. ``kb_hint`` is the top KEDB article's ``priority_hint``."""
    full = cfg or ai_config()
    pcfg = full.get("priority") or {}
    tau_kb = float(full["retrieval"]["kb"]["tau_kb"])
    matrix = {int(i): {int(u): str(c) for u, c in row.items()} for i, row in pcfg["matrix"].items()}

    intent = classification.intent
    impact, i_drivers = _impact(signals, text, pcfg)
    u, u_drivers = _urgency(signals, intent)
    has_signal = bool(i_drivers or u_drivers)
    urgency = u if u is not None else 2
    if intent in ("how_to", "service_request") and not u_drivers:
        urgency = 3
    p_matrix = matrix[impact][urgency]

    sub = classification.subcategory
    p_base = baseline(classification.category)
    drivers = i_drivers + u_drivers

    if intent in ("how_to", "service_request"):
        code = p_matrix
    elif not has_signal:
        code = p_base
    else:
        hint = kb_hint if kb_hint in PRIORITY_CODES and kb_relevance >= tau_kb else None
        code = most_severe([p_matrix, p_base, hint])
        if code != p_matrix and PRIORITY_CODES.index(code) < PRIORITY_CODES.index(p_matrix):
            if code == p_base:
                drivers.append(_Driver(f"floor: {sub} baseline {code}", f"floor: {sub} baseline"))
            else:
                drivers.append(
                    _Driver(f"floor: {kb_id} hint {code}", f"floor: {kb_id} hint {code}")
                )

    reason = "; ".join(d.reason for d in drivers[:2]) if drivers else f"Baseline for {sub}"
    return PriorityResult(
        code=code,  # type: ignore[arg-type]
        level=PRIORITY_LEVEL[code],  # type: ignore[arg-type]
        reason=reason,
        impact=impact,
        urgency=urgency,
        drivers=[d.text for d in drivers],
    )
