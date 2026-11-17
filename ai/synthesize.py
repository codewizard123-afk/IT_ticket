"""Resolution synthesiser (architecture §7.7, FR-03, FR-04). Owner: P2.

Builds an evidence pack (top 1–2 KB articles with numbered ``diag.n`` / ``res.n`` steps, and up to
3 *verified* incidents — synthetic resolutions are never included), asks Groq (call #2) for cited
steps, and falls back to rendering the KB steps verbatim when the LLM is unavailable or its output
fails validation twice (``llm_synth_fallback``). Every returned step carries ≥ 1 citation.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from ai.config import ai_config
from ai.prompts import render
from ai.text import grounding, token_overlap
from ai.validate import source_for, validate_steps
from shared.constants import CITABLE_ORIGINS
from shared.interfaces import KBStore, LLMClient, LLMUnavailableError
from shared.schemas import (
    ClassificationResult,
    KBArticle,
    KBHit,
    Recommendation,
    Signals,
    Step,
    TicketHit,
)

log = logging.getLogger(__name__)


# --- evidence pack -----------------------------------------------------------------------------
@dataclass
class EvidencePack:
    issue_text: str
    signals: Signals
    articles: list[KBArticle] = field(default_factory=list)
    incidents: list[TicketHit] = field(default_factory=list)
    citable: dict[str, str] = field(default_factory=dict)  # citation id → cited text
    kinds: dict[str, str] = field(
        default_factory=dict
    )  # citation id → diagnostic|resolution|incident

    @property
    def empty(self) -> bool:
        return not self.citable

    def as_prompt_json(self, max_chars: int) -> str:
        data = {
            "kb_articles": [
                {
                    "kb_id": a.kb_id,
                    "title": a.title,
                    "root_cause": a.root_cause,
                    "steps": {
                        cid: txt for cid, txt in self.citable.items() if cid.startswith(a.kb_id)
                    },
                    "escalation_criteria": a.escalation_criteria,
                    "requires_admin": a.resolution.requires_admin,
                    "estimated_time": a.resolution.estimated_time,
                }
                for a in self.articles
            ],
            "verified_incidents": [
                {"incident_id": i.ticket_id, "title": i.title, "resolution": i.resolution}
                for i in self.incidents
            ],
        }
        text = json.dumps(data, ensure_ascii=False)
        return text[:max_chars]


def build_pack(
    issue_text: str,
    signals: Signals,
    classification: ClassificationResult | None,
    kb_hits: list[KBHit],
    similar: list[TicketHit],
    kb_store: KBStore,
    cfg: dict[str, Any] | None = None,
) -> EvidencePack:
    cfg = cfg or ai_config()
    scfg = cfg["synthesis"]
    pack = EvidencePack(issue_text=issue_text, signals=signals)
    for hit in kb_hits[: int(scfg["max_kb_articles"])]:
        try:
            art = kb_store.get_article(hit.kb_id)
        except Exception:  # noqa: BLE001
            continue
        if art.status != "approved":
            continue
        pack.articles.append(art)
        for i, s in enumerate(art.diagnostic_steps, 1):
            pack.citable[f"{art.kb_id}#diag.{i}"] = s
            pack.kinds[f"{art.kb_id}#diag.{i}"] = "diagnostic"
        for i, s in enumerate(art.resolution.steps, 1):
            pack.citable[f"{art.kb_id}#res.{i}"] = s
            pack.kinds[f"{art.kb_id}#res.{i}"] = "resolution"
    min_cos = float(scfg["verified_min_cos"])
    for h in similar:
        if len(pack.incidents) >= int(scfg["max_verified_incidents"]):
            break
        if h.resolution_origin not in CITABLE_ORIGINS or not h.resolution:
            continue  # synthetic resolutions are never evidence (CLAUDE.md rule 4)
        if h.score < min_cos:
            continue
        if classification is not None and h.subcategory != classification.subcategory:
            continue
        pack.incidents.append(h)
        pack.citable[h.ticket_id] = h.resolution
        pack.kinds[h.ticket_id] = "incident"
    return pack


# --- step post-processing ----------------------------------------------------------------------
def _rules(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("synthesis_rules") or {}


def is_l2(text: str, cfg: dict[str, Any]) -> bool:
    pat = _rules(cfg).get("l2_pattern")
    return bool(pat and re.search(pat, text, re.IGNORECASE))


def already_attempted(text: str, tried: list[str], cfg: dict[str, Any]) -> bool:
    need = float(_rules(cfg).get("attempted_min_overlap", 0.6))
    return any(token_overlap(t, text) >= need for t in tried)


def _finish(steps: list[Step], pack: EvidencePack, cfg: dict[str, Any]) -> list[Step]:
    """Role tags, already-attempted flags, diagnostics before resolution (stable)."""
    out = []
    for s in steps:
        role = "L2" if (s.role == "L2" or is_l2(s.text, cfg)) else "L1"
        tried = s.already_attempted or already_attempted(s.text, pack.signals.tried_steps, cfg)
        out.append(s.model_copy(update={"role": role, "already_attempted": tried}))
    return [s for s in out if s.type == "diagnostic"] + [s for s in out if s.type != "diagnostic"]


def _recommendation(
    steps: list[Step], pack: EvidencePack, cfg: dict[str, Any], **extra: Any
) -> Recommendation:
    """Run the §7.8 validator over ``steps`` and wrap the survivors in a ``Recommendation``."""
    report = validate_steps(steps, pack.citable, cfg)
    kept = report.steps
    used = {c.split("#", 1)[0] for s in kept for c in s.citations if "#" in c}
    arts = [a for a in pack.articles if a.kb_id in used]
    min_steps = int(cfg["validator"]["min_steps"])
    return Recommendation(
        steps=kept,
        source=source_for(kept),  # type: ignore[arg-type]
        insufficient_evidence=len(kept) < min_steps,
        requires_admin=any(a.resolution.requires_admin for a in arts),
        estimated_time=arts[0].resolution.estimated_time if arts else None,
        unsupported_steps_removed=report.removed,
        safety_violations=report.safety_violations,
        **extra,
    )


# --- template fallback -------------------------------------------------------------------------
def template_steps(pack: EvidencePack, cfg: dict[str, Any] | None = None) -> list[Step]:
    """KB diagnostic + resolution steps verbatim, in order, with citations (offline path)."""
    cfg = cfg or ai_config()
    link_min = float(_rules(cfg).get("incident_link_min", 0.30))
    steps: list[Step] = []
    if pack.articles:
        top = pack.articles[0]
        for i, text in enumerate(top.diagnostic_steps, 1):
            steps.append(Step(text=text, type="diagnostic", citations=[f"{top.kb_id}#diag.{i}"]))
        for i, text in enumerate(top.resolution.steps, 1):
            cites = [f"{top.kb_id}#res.{i}"]
            for inc in pack.incidents:
                if grounding(text, inc.resolution or "") >= link_min:
                    cites.append(inc.ticket_id)
            steps.append(Step(text=text, type="resolution", citations=cites))
    else:
        for inc in pack.incidents:
            steps.append(
                Step(text=inc.resolution or "", type="resolution", citations=[inc.ticket_id])
            )
    return _finish(steps, pack, cfg)


def template_recommendation(pack: EvidencePack, cfg: dict[str, Any] | None = None):
    cfg = cfg or ai_config()
    if pack.empty:
        return Recommendation(insufficient_evidence=True, source="None")
    steps = template_steps(pack, cfg)
    top = pack.articles[0] if pack.articles else None
    return _recommendation(
        steps,
        pack,
        cfg,
        summary=top.title if top else "Verified incident resolution",
        probable_cause=top.root_cause if top else None,
    )


# --- LLM path ------------------------------------------------------------------------------------
class _LLMStep(BaseModel):
    n: int | None = None
    type: Literal["diagnostic", "resolution"] = "resolution"
    text: str = Field(min_length=3)
    citations: list[str] = Field(default_factory=list)
    role: Literal["L1", "L2"] = "L1"
    already_attempted: bool = False


class _LLMCause(BaseModel):
    text: str | None = None
    citations: list[str] = Field(default_factory=list)


class _LLMSynthOut(BaseModel):
    insufficient_evidence: bool = False
    summary: str | None = None
    probable_cause: _LLMCause | str | None = None
    steps: list[_LLMStep] = Field(default_factory=list)
    alternatives: list[Any] = Field(default_factory=list)
    estimated_time: str | None = None
    requires_admin: bool = False


def build_synth_prompt(pack: EvidencePack, cfg: dict[str, Any]) -> tuple[str, str]:
    system = render("synthesize.j2")
    sig = pack.signals.model_dump(
        include={"error_codes", "products", "scope", "affected_users", "tried_steps"}
    )
    user = render(
        "synthesize_user.j2",
        pack=pack.as_prompt_json(int(cfg["synthesis"]["max_pack_chars"])),
        signals=json.dumps(sig, ensure_ascii=False),
        tried=json.dumps(pack.signals.tried_steps, ensure_ascii=False),
        ticket=pack.issue_text.replace("</ticket>", "</ ticket>"),
    )
    return system, user


def _llm_synthesize(pack: EvidencePack, llm: LLMClient, cfg: dict[str, Any]) -> Recommendation:
    """Groq call #2. Raises ``LLMUnavailableError`` after two invalid outputs."""
    system, user = build_synth_prompt(pack, cfg)
    for attempt in range(2):
        prompt = user if attempt == 0 else user + "\n\nReturn valid JSON exactly as specified."
        raw = llm.complete_json(
            purpose="synthesize",
            system=system,
            user=prompt,
            model=cfg["llm"]["synth_model"],
            max_tokens=900,
        )
        try:
            out = _LLMSynthOut.model_validate(raw)
        except ValidationError:
            continue
        if out.insufficient_evidence and not out.steps:
            return Recommendation(insufficient_evidence=True, source="None")
        steps = []
        for s in out.steps:
            with contextlib.suppress(ValidationError):
                steps.append(
                    Step(
                        text=s.text.strip(),
                        type=s.type,
                        citations=s.citations or ["<none>"],
                        role=s.role,
                        already_attempted=s.already_attempted,
                    )
                )
        cause = out.probable_cause
        cause_text = cause.text if isinstance(cause, _LLMCause) else cause
        rec = _recommendation(
            _finish(steps, pack, cfg), pack, cfg, summary=out.summary, probable_cause=cause_text
        )
        if not rec.steps:
            continue  # nothing survived validation: counts as a validation failure
        return rec
    raise LLMUnavailableError("synthesiser output invalid twice")


def synthesize(
    issue_text: str,
    signals: Signals,
    classification: ClassificationResult | None,
    kb_hits: list[KBHit],
    similar: list[TicketHit],
    kb_store: KBStore,
    llm: LLMClient | None,
    cfg: dict[str, Any] | None = None,
) -> tuple[Recommendation, EvidencePack, list[str]]:
    """Cited recommendation for one issue, the evidence pack it used, and degraded flags."""
    cfg = cfg or ai_config()
    pack = build_pack(issue_text, signals, classification, kb_hits, similar, kb_store, cfg)
    if pack.empty:
        return Recommendation(insufficient_evidence=True, source="None"), pack, []
    if llm is not None and llm.available:
        try:
            return _llm_synthesize(pack, llm, cfg), pack, []
        except LLMUnavailableError:
            log.info("synthesiser falling back to KB template")
    return template_recommendation(pack, cfg), pack, ["llm_synth_fallback"]
