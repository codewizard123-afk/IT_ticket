"""Resolution synthesiser (architecture §7.7, FR-03).

Evidence pack: top 1–2 KB articles (numbered `diag.n` / `res.n`) and up to 3 *verified*
incidents (`curated_verified` / `analyst_verified`, cosine ≥ 0.75, same subcategory).
Synthetic resolutions (`synthetic_*`) are never put in the pack, so they can never be cited.

Groq call #2 writes cited JSON steps. A validator (`ai.validate.validate`, injected) checks
them; after `max_validation_failures` failures, or when the LLM is unavailable, the template
fallback renders the KB diagnostic and resolution steps verbatim with their citations
(`degraded += llm_synth_fallback`).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cache
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from ai.config import ai_config
from ai.prompts import render_messages
from ai.signals import Signals, step_already_attempted
from ai.types import (
    VERIFIED_ORIGINS,
    KBSearchResult,
    KBStore,
    LLMBadOutput,
    LLMClient,
    LLMUnavailable,
    Recommendation,
    SimilarIncident,
    Step,
)

SOURCE_KB = "KEDB"
SOURCE_INC = "Historical Incident"
SOURCE_BOTH = "KEDB + Historical Incident"
SOURCE_NONE = "None"

Validator = Callable[[Recommendation, "EvidencePack"], Recommendation]


# --------------------------------------------------------------------------- evidence pack


@dataclass
class PackStep:
    id: str
    text: str


@dataclass
class PackArticle:
    kb_id: str
    title: str
    root_cause: str
    diagnostics: list[PackStep]
    resolutions: list[PackStep]
    escalation: list[str]
    requires_admin: bool
    estimated_time: str | None


@dataclass
class PackIncident:
    id: str
    text: str
    resolution: str
    subcategory: str | None
    score: float


@dataclass
class EvidencePack:
    articles: list[PackArticle] = field(default_factory=list)
    incidents: list[PackIncident] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.articles and not self.incidents

    def citable(self) -> dict[str, str]:
        """Citation id → cited text (used by the validator)."""
        out: dict[str, str] = {}
        for a in self.articles:
            for s in a.diagnostics + a.resolutions:
                out[s.id] = s.text
        for inc in self.incidents:
            out[inc.id] = inc.resolution
        return out

    def requires_admin(self, kb_id: str) -> bool:
        return any(a.requires_admin for a in self.articles if a.kb_id == kb_id)


def build_pack(
    kb_result: KBSearchResult,
    similar: list[SimilarIncident],
    kb_store: KBStore,
    *,
    subcategory: str | None,
    cfg: dict[str, Any] | None = None,
) -> EvidencePack:
    cfg = cfg if cfg is not None else ai_config()
    scfg = cfg.get("synth", {})
    max_art = int(scfg.get("max_kb_articles", 2))
    max_inc = int(scfg.get("max_verified_incidents", 3))
    min_cos = float(scfg.get("verified_min_cos", 0.75))

    articles: list[PackArticle] = []
    for m in kb_result.matches[:max_art]:
        art = kb_store.get_article(m.kb_id)
        articles.append(
            PackArticle(
                kb_id=art.kb_id,
                title=art.title,
                root_cause=art.root_cause,
                diagnostics=[
                    PackStep(f"{art.kb_id}#diag.{i}", s)
                    for i, s in enumerate(art.diagnostic_steps, 1)
                ],
                resolutions=[
                    PackStep(f"{art.kb_id}#res.{i}", s)
                    for i, s in enumerate(art.resolution.steps, 1)
                ],
                escalation=list(art.escalation_criteria),
                requires_admin=art.resolution.requires_admin,
                estimated_time=art.resolution.estimated_time,
            )
        )

    incidents: list[PackIncident] = []
    for s in similar:
        if len(incidents) >= max_inc:
            break
        if s.resolution_origin not in VERIFIED_ORIGINS or not s.resolution:
            continue  # synthetic or unresolved: never evidence
        if s.similarity_score < min_cos or (
            subcategory and s.subcategory != subcategory
        ):
            continue
        incidents.append(
            PackIncident(
                s.incident_id,
                s.excerpt,
                s.resolution,
                s.subcategory,
                s.similarity_score,
            )
        )
    return EvidencePack(articles, incidents)


# --------------------------------------------------------------------------- helpers


@cache
def _rx(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


def citation_kind(cid: str) -> str:
    return "kb" if re.match(r"^KB-\d+#(?:diag|res)\.\d+$", cid) else "incident"


def source_for(steps: list[Step]) -> str:
    """§4.2: KEDB if all citations are KB, Historical Incident if all are verified incidents,
    mixed → both, no grounded steps → None."""
    kinds = {citation_kind(c) for s in steps for c in s.citations}
    if not kinds:
        return SOURCE_NONE
    if kinds == {"kb"}:
        return SOURCE_KB
    if kinds == {"incident"}:
        return SOURCE_INC
    return SOURCE_BOTH


def role_for(
    text: str, step_type: str, requires_admin: bool, cfg: dict[str, Any] | None = None
) -> str:
    cfg = cfg if cfg is not None else ai_config()
    scfg = cfg.get("synth", {})
    if any(_rx(p).search(text) for p in scfg.get("l2_patterns", [])):
        return "L2"
    if step_type == "resolution" and requires_admin:
        if any(
            _rx(p).search(text.strip()) for p in scfg.get("verification_patterns", [])
        ):
            return "L1"
        return "L2"
    return "L1"


def _finalise_steps(
    steps: list[Step], pack: EvidencePack, signals: Signals, cfg: dict[str, Any]
) -> list[Step]:
    """Diagnostics first (stable), deterministic L2 tagging and already-attempted marking,
    renumbered 1..n. The LLM can add an L2 tag or an attempted mark but never remove one."""
    ordered = [s for s in steps if s.type == "diagnostic"] + [
        s for s in steps if s.type != "diagnostic"
    ]
    out = []
    for n, s in enumerate(ordered, 1):
        kb_ids = {c.split("#")[0] for c in s.citations if citation_kind(c) == "kb"}
        req_admin = any(pack.requires_admin(k) for k in kb_ids)
        role = "L2" if s.role == "L2" else role_for(s.text, s.type, req_admin, cfg)
        attempted = s.already_attempted or step_already_attempted(s.text, signals)
        out.append(
            Step(
                text=s.text,
                type=s.type,
                citations=list(dict.fromkeys(s.citations)),
                role=role,  # type: ignore[arg-type]
                already_attempted=attempted,
                n=n,
            )
        )
    return out


# --------------------------------------------------------------------------- template fallback


def template_recommendation(
    pack: EvidencePack, signals: Signals, cfg: dict[str, Any] | None = None
) -> Recommendation:
    """KB diagnostic and resolution steps of the top article, verbatim and cited; verified
    incident resolutions when there is no article."""
    cfg = cfg if cfg is not None else ai_config()
    steps: list[Step] = []
    summary = ""
    cause: dict[str, Any] = {}
    est = None
    req_admin = False
    if pack.articles:
        art = pack.articles[0]
        steps += [Step(s.text, "diagnostic", [s.id]) for s in art.diagnostics]
        steps += [Step(s.text, "resolution", [s.id]) for s in art.resolutions]
        summary = f"Follow {art.kb_id}: {art.title}."
        cause = (
            {"text": art.root_cause, "citations": [f"{art.kb_id}#title"]}
            if art.root_cause
            else {}
        )
        est = art.estimated_time
        req_admin = art.requires_admin
    elif pack.incidents:
        for inc in pack.incidents:
            steps.append(Step(inc.resolution, "resolution", [inc.id]))
        summary = f"Apply the resolution of verified incident {pack.incidents[0].id}."
    final = _finalise_steps(steps, pack, signals, cfg)
    return Recommendation(
        steps=final,
        source=source_for(final),
        summary=summary,
        probable_cause=cause,
        estimated_time=est,
        requires_admin=req_admin or any(s.role == "L2" for s in final),
        insufficient_evidence=not final,
    )


# --------------------------------------------------------------------------- LLM path


class _LLMStep(BaseModel):
    n: int = 0
    type: str = "resolution"
    text: str = Field(min_length=3, max_length=600)
    citations: list[str] = Field(default_factory=list)
    role: str = "L1"
    already_attempted: bool = False


class _LLMCause(BaseModel):
    text: str = ""
    citations: list[str] = Field(default_factory=list)


class LLMSynthOut(BaseModel):
    summary: str = ""
    probable_cause: _LLMCause = Field(default_factory=_LLMCause)
    steps: list[_LLMStep] = Field(default_factory=list)
    alternatives: list[str] = Field(default_factory=list)
    estimated_time: str | None = None
    requires_admin: bool = False
    insufficient_evidence: bool = False


def _from_llm(
    out: LLMSynthOut, pack: EvidencePack, signals: Signals, cfg: dict[str, Any]
) -> Recommendation:
    steps = [
        Step(
            text=" ".join(s.text.split()),
            type="diagnostic" if s.type.lower().startswith("diag") else "resolution",
            citations=[c.strip() for c in s.citations if c and c.strip()],
            role="L2" if s.role.upper() == "L2" else "L1",
            already_attempted=s.already_attempted,
        )
        for s in out.steps
    ]
    final = _finalise_steps(steps, pack, signals, cfg)
    return Recommendation(
        steps=final,
        source=source_for(final),
        summary=out.summary[:500],
        probable_cause=out.probable_cause.model_dump(),
        estimated_time=out.estimated_time,
        requires_admin=out.requires_admin or any(s.role == "L2" for s in final),
        insufficient_evidence=out.insufficient_evidence,
    )


def _messages(
    issue_text: str, signals: Signals, pack: EvidencePack, retry_reason: str | None
) -> list[dict[str, str]]:
    sig = signals.to_dict()
    brief = {
        k: sig[k]
        for k in (
            "error_codes",
            "products",
            "affected_users",
            "scope",
            "urgency_terms",
            "environment",
            "onset",
            "change_trigger",
        )
    }
    return render_messages(
        "synthesize.j2",
        pack=pack,
        signals_json=json.dumps(brief, ensure_ascii=False),
        tried=signals.tried_steps,
        issue_text=issue_text,
        retry_reason=retry_reason,
    )


def recommend(
    issue_text: str,
    signals: Signals,
    kb_result: KBSearchResult,
    similar: list[SimilarIncident],
    kb_store: KBStore,
    llm: LLMClient,
    *,
    subcategory: str | None,
    validator: Validator | None = None,
    cfg: dict[str, Any] | None = None,
    analysis_id: str | None = None,
) -> tuple[Recommendation, EvidencePack, list[str]]:
    """Return `(recommendation, evidence_pack, degraded_flags)`."""
    cfg = cfg if cfg is not None else ai_config()
    scfg = cfg.get("synth", {})
    min_steps = int(cfg.get("validator", {}).get("min_steps", 2))
    pack = build_pack(kb_result, similar, kb_store, subcategory=subcategory, cfg=cfg)
    if pack.empty:
        return (
            Recommendation(
                source=SOURCE_NONE,
                insufficient_evidence=True,
                summary="No approved KB article or verified incident matches.",
            ),
            pack,
            [],
        )

    def check(rec: Recommendation) -> Recommendation:
        return validator(rec, pack) if validator is not None else rec

    degraded: list[str] = []
    if llm.available:
        model = cfg.get("llm", {}).get("synth_model", "")
        reason: str | None = None
        for _attempt in range(int(scfg.get("max_validation_failures", 2))):
            try:
                raw = llm.complete_json(
                    purpose="synthesize",
                    model=model,
                    messages=_messages(issue_text, signals, pack, reason),
                    analysis_id=analysis_id,
                )
                out = LLMSynthOut.model_validate(raw)
            except LLMUnavailable:
                break
            except (LLMBadOutput, ValidationError):
                reason = "the JSON did not match the required shape"
                continue
            if out.insufficient_evidence and not out.steps:
                rec = Recommendation(
                    source=SOURCE_NONE,
                    insufficient_evidence=True,
                    summary=out.summary[:500],
                )
                return rec, pack, degraded
            rec = check(_from_llm(out, pack, signals, cfg))
            if len(rec.steps) >= min_steps and not rec.insufficient_evidence:
                return rec, pack, degraded
            reason = (
                f"only {len(rec.steps)} step(s) survived validation; every step needs a "
                "citation from the pack and wording close to the cited evidence"
            )
    degraded.append("llm_synth_fallback")
    return check(template_recommendation(pack, signals, cfg)), pack, degraded
