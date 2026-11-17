"""Citation validator (architecture §7.8, FR-04).

- Unknown citation IDs are removed; a step left with no citation is dropped
  (`unsupported_steps_removed`).
- Grounding: ROUGE-L between the step and its best cited text must be ≥ `grounding_min`
  (0.30). The score is LCS / len(step tokens): the share of the step that is supported by
  the cited evidence. Ungrounded steps are dropped.
- Safety lint: steps matching the `validator.safety_patterns` (disable antivirus/firewall/
  UAC/Defender, share password, format drive, delete registry, grant DBA/owner/admin) are
  dropped and a `safety_violation` is logged. Steps echoing prompt-injection phrases are
  dropped too.
- Fewer than `min_steps` steps left → `insufficient_evidence = true`.
- `source` is set per §4.2.

Only rule names are logged, never step or ticket text.
"""

from __future__ import annotations

import logging
import re
from dataclasses import replace
from functools import cache
from typing import Any

from ai.config import ai_config
from ai.synthesize import EvidencePack, source_for
from ai.text import rouge_l_recall
from ai.types import Recommendation, Step

log = logging.getLogger(__name__)


@cache
def _rx(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


def safety_violation(text: str, cfg: dict[str, Any] | None = None) -> str | None:
    """Return the matching safety pattern, or None if the text is safe."""
    cfg = cfg if cfg is not None else ai_config()
    for pat in cfg.get("validator", {}).get("safety_patterns", []):
        if _rx(pat).search(text):
            return pat
    return None


def injection_echo(text: str, cfg: dict[str, Any] | None = None) -> str | None:
    cfg = cfg if cfg is not None else ai_config()
    for pat in cfg.get("validator", {}).get("injection_patterns", []):
        if _rx(pat).search(text):
            return pat
    return None


def grounding(step_text: str, citations: list[str], citable: dict[str, str]) -> float:
    return max(
        (rouge_l_recall(step_text, citable[c]) for c in citations if c in citable),
        default=0.0,
    )


def validate(
    rec: Recommendation, pack: EvidencePack, cfg: dict[str, Any] | None = None
) -> Recommendation:
    cfg = cfg if cfg is not None else ai_config()
    vcfg = cfg.get("validator", {})
    g_min = float(vcfg.get("grounding_min", 0.30))
    min_steps = int(vcfg.get("min_steps", 2))
    citable = pack.citable()

    report: dict[str, Any] = {
        "unknown_citations_removed": 0,
        "unsupported_steps_removed": 0,
        "ungrounded_steps_removed": 0,
        "safety_violations": 0,
        "injection_steps_removed": 0,
        "grounding": [],
    }
    kept: list[Step] = []
    for step in rec.steps:
        cits = [c for c in step.citations if c in citable]
        report["unknown_citations_removed"] += len(step.citations) - len(cits)
        if not cits:
            report["unsupported_steps_removed"] += 1
            continue
        rule = safety_violation(step.text, cfg)
        if rule:
            report["safety_violations"] += 1
            log.warning("safety_violation: step dropped (rule=%s)", rule)
            continue
        if injection_echo(step.text, cfg):
            report["injection_steps_removed"] += 1
            log.warning("prompt-injection echo: step dropped")
            continue
        g = grounding(step.text, cits, citable)
        if g < g_min:
            report["ungrounded_steps_removed"] += 1
            continue
        report["grounding"].append(round(g, 2))
        kept.append(replace(step, citations=cits))

    kept = [replace(s, n=i) for i, s in enumerate(kept, 1)]
    cause = dict(rec.probable_cause or {})
    if cause.get("citations"):
        known = set(citable) | {f"{a.kb_id}#title" for a in pack.articles}
        cause["citations"] = [c for c in cause["citations"] if c in known]
        if not cause["citations"]:
            cause = {}
    if cause.get("text") and (
        safety_violation(cause["text"], cfg) or injection_echo(cause["text"], cfg)
    ):
        cause = {}

    insufficient = rec.insufficient_evidence or len(kept) < min_steps
    return replace(
        rec,
        steps=kept,
        source=source_for(kept),
        probable_cause=cause,
        insufficient_evidence=insufficient,
        requires_admin=rec.requires_admin and bool(kept),
        validation=report,
    )
