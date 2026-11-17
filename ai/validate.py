"""Citation validator (architecture §7.8, FR-04). Owner: P2.

* Unknown citation IDs are removed. A step left with no citation is dropped
  (``unsupported_steps_removed++``). Synthetic resolutions are never in the pack, so they can
  never be cited.
* Grounding: ROUGE-L between the step and each cited text must be ≥ 0.30; citations below it are
  removed, and a step with no grounded citation is dropped.
* Safety lint: steps matching ``validator.safety_patterns`` are dropped and counted as
  ``safety_violation`` (logged without the step text).
* Fewer than ``min_steps`` (2) steps left → ``insufficient_evidence = true``.
* ``source`` follows §4.2.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from ai.config import ai_config
from ai.text import grounding
from shared.schemas import Recommendation, Step

log = logging.getLogger(__name__)


@lru_cache(maxsize=8)
def _compile(patterns: tuple[str, ...]) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(p, re.IGNORECASE) for p in patterns)


def is_unsafe(text: str, cfg: dict[str, Any] | None = None) -> bool:
    patterns = tuple((cfg or ai_config())["validator"].get("safety_patterns") or ())
    return any(p.search(text) for p in _compile(patterns))


def source_for(steps: list[Step]) -> str:
    """``KEDB`` if all citations are KB · ``Historical Incident`` if all are verified incidents ·
    ``KEDB + Historical Incident`` if mixed · ``None`` if there are no grounded steps."""
    cites = [c for s in steps for c in s.citations]
    has_kb = any("#" in c for c in cites)
    has_inc = any("#" not in c for c in cites)
    if has_kb and has_inc:
        return "KEDB + Historical Incident"
    if has_kb:
        return "KEDB"
    if has_inc:
        return "Historical Incident"
    return "None"


@dataclass
class ValidationReport:
    steps: list[Step] = field(default_factory=list)
    unsupported_removed: int = 0
    ungrounded_removed: int = 0
    safety_violations: int = 0
    duplicates_removed: int = 0

    @property
    def removed(self) -> int:
        return self.unsupported_removed + self.ungrounded_removed + self.safety_violations


def validate_steps(
    steps: list[Step], citable: dict[str, str], cfg: dict[str, Any] | None = None
) -> ValidationReport:
    """Apply every §7.8 check to ``steps`` against the evidence pack's ``citable`` texts."""
    cfg = cfg or ai_config()
    vcfg = cfg["validator"]
    g_min = float(vcfg["grounding_min"])
    report = ValidationReport()
    seen: set[str] = set()
    for step in steps:
        if is_unsafe(step.text, cfg):
            report.safety_violations += 1
            log.warning("safety_violation: unsafe step removed (citations=%s)", step.citations)
            continue
        known = [c for c in dict.fromkeys(step.citations) if c in citable]
        if not known:
            report.unsupported_removed += 1
            continue
        scores = {c: grounding(step.text, citable[c]) for c in known}
        grounded = [c for c in known if scores[c] >= g_min]
        if not grounded:
            report.ungrounded_removed += 1
            continue
        key = step.text.strip().lower()
        if key in seen:
            report.duplicates_removed += 1
            continue
        seen.add(key)
        report.steps.append(
            step.model_copy(
                update={
                    "citations": grounded,
                    "grounding": round(max(scores[c] for c in grounded), 2),
                }
            )
        )
    diag = [s for s in report.steps if s.type == "diagnostic"]
    res = [s for s in report.steps if s.type != "diagnostic"]
    report.steps = diag + res
    return report


def validate_recommendation(
    rec: Recommendation, citable: dict[str, str], cfg: dict[str, Any] | None = None
) -> Recommendation:
    """Validated copy of ``rec``: steps filtered, ``source`` and ``insufficient_evidence`` set."""
    cfg = cfg or ai_config()
    report = validate_steps(rec.steps, citable, cfg)
    min_steps = int(cfg["validator"]["min_steps"])
    return rec.model_copy(
        update={
            "steps": report.steps,
            "source": source_for(report.steps),
            "insufficient_evidence": rec.insufficient_evidence or len(report.steps) < min_steps,
            "unsupported_steps_removed": rec.unsupported_steps_removed + report.removed,
            "safety_violations": getattr(rec, "safety_violations", 0) + report.safety_violations,
        }
    )
