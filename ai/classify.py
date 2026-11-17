"""Fused classifier (architecture §7.3): kNN + LLM + rules voters.

    score(s) = w_llm·v_llm + w_knn·v_knn + w_rule·v_rule
    conf     = clip(0.6·score(pred) + 0.25·margin + 0.15·agree, 0, 1)
    bands    : high ≥ 0.75 · medium ≥ 0.55 · low otherwise
    conflict : LLM top-1 and kNN top-1 in different categories, both ≥ 0.4
    LLM down : knn 0.65 / rule 0.35, conf capped at 0.80, degraded += llm_classify_fallback

`agree` is the fraction of *available* voters whose top-1 equals the prediction (3 voters
online, 2 when the LLM is down; the 0.80 cap keeps the offline case conservative).
Weights, bands and mixes come from `config/ai.yaml`; rules and taxonomy from
`config/rules.yaml`.
"""

from __future__ import annotations

import json
import re
from functools import cache
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

from ai.config import ai_config, rules_config
from ai.prompts import render_messages
from ai.signals import Signals
from ai.types import (
    ClassificationResult,
    LLMBadOutput,
    LLMClient,
    LLMUnavailable,
    TicketHit,
    TicketStore,
)

Label = tuple[str, str]
OTHER = "Other"


def label_key(label: Label) -> str:
    return f"{label[0]}|{label[1]}"


# --------------------------------------------------------------------------- LLM schema


class _Alt(BaseModel):
    category: str
    subcategory: str
    score: float = Field(ge=0, le=1)


class LLMIssue(BaseModel):
    text_span: str = ""
    category: str
    subcategory: str
    intent: str = "incident"
    confidence: float = Field(ge=0, le=1)
    alternatives: list[_Alt] = Field(default_factory=list)
    rationale: str = ""

    @field_validator("rationale", "text_span")
    @classmethod
    def _short(cls, v: str) -> str:
        return v[:2000]


class LLMClassifyOut(BaseModel):
    issues: list[LLMIssue] = Field(min_length=1)


# --------------------------------------------------------------------------- taxonomy


def taxonomy(rcfg: dict[str, Any] | None = None) -> dict[str, list[str]]:
    rcfg = rcfg if rcfg is not None else rules_config()
    return {k: list(v) for k, v in (rcfg.get("taxonomy") or {}).items()}


def is_valid_label(label: Label, tax: dict[str, list[str]]) -> bool:
    cat, sub = label
    return cat in tax and (sub in tax[cat] or sub == OTHER)


# --------------------------------------------------------------------------- voters


def knn_votes(hits: list[TicketHit]) -> dict[Label, float]:
    """v_knn(s) = Σ sim_i·[label_i = s] / Σ sim_i."""
    total = 0.0
    votes: dict[Label, float] = {}
    for h in hits:
        cat = h.metadata.get("category")
        sub = h.metadata.get("subcategory")
        if not cat or not sub or h.score <= 0:
            continue
        votes[(cat, sub)] = votes.get((cat, sub), 0.0) + h.score
        total += h.score
    return {k: v / total for k, v in votes.items()} if total else {}


@cache
def _rx(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


def rule_votes(
    text: str, rcfg: dict[str, Any] | None = None
) -> tuple[dict[Label, float], list[str]]:
    rcfg = rcfg if rcfg is not None else rules_config()
    counts: dict[Label, int] = {}
    fired: list[str] = []
    for rule in rcfg.get("rules", []):
        n = len(_rx(rule["pattern"]).findall(text))
        if n:
            lab = (rule["category"], rule["subcategory"])
            counts[lab] = counts.get(lab, 0) + n
            fired.append(rule["name"])
    total = sum(counts.values())
    return ({k: v / total for k, v in counts.items()} if total else {}), fired


def llm_votes(issue: LLMIssue, tax: dict[str, list[str]]) -> dict[Label, float]:
    votes: dict[Label, float] = {}
    top = (issue.category, issue.subcategory)
    if is_valid_label(top, tax):
        votes[top] = issue.confidence
    for alt in issue.alternatives:
        lab = (alt.category, alt.subcategory)
        if is_valid_label(lab, tax) and lab not in votes:
            votes[lab] = alt.score
    total = sum(votes.values())
    if total > 1.0:
        votes = {k: v / total for k, v in votes.items()}
    return votes


def detect_intent(text: str, rcfg: dict[str, Any] | None = None) -> str:
    rcfg = rcfg if rcfg is not None else rules_config()
    for intent in ("access_request", "how_to", "service_request"):
        for pat in (rcfg.get("intent_rules") or {}).get(intent, []):
            if _rx(pat).search(text):
                return intent
    return "incident"


# --------------------------------------------------------------------------- LLM call #1


def neighbours(
    store: TicketStore, text: str, k: int, where: dict[str, Any] | None = None
) -> list[TicketHit]:
    return store.query(
        text, k, where=where, collections=("tickets_hist", "tickets_verified")
    )


def llm_classify(
    ticket_text: str,
    signals: Signals,
    nbrs: list[TicketHit],
    llm: LLMClient,
    *,
    cfg: dict[str, Any] | None = None,
    rcfg: dict[str, Any] | None = None,
    analysis_id: str | None = None,
) -> LLMClassifyOut | None:
    """Groq call #1 (classify + split). Retries once on invalid output, then returns None."""
    cfg = cfg if cfg is not None else ai_config()
    rcfg = rcfg if rcfg is not None else rules_config()
    if not llm.available:
        return None
    tax = taxonomy(rcfg)
    sig = signals.to_dict()
    sig_brief = {
        k: sig[k]
        for k in (
            "error_codes",
            "products",
            "affected_users",
            "scope",
            "urgency_terms",
            "security_flags",
            "multi_issue_markers",
        )
    }
    messages = render_messages(
        "classify.j2",
        taxonomy=tax,
        intents=rcfg.get("intents", []),
        few_shot=rcfg.get("few_shot", []),
        neighbours=[
            {
                "text": n.text[:160],
                "category": n.metadata.get("category"),
                "subcategory": n.metadata.get("subcategory"),
                "score": n.score,
            }
            for n in nbrs[:5]
        ],
        signals_json=json.dumps(sig_brief, ensure_ascii=False),
        ticket_text=ticket_text,
    )
    model = cfg.get("llm", {}).get("classify_model", "")
    for _attempt in range(2):
        try:
            raw = llm.complete_json(
                purpose="classify",
                model=model,
                messages=messages,
                analysis_id=analysis_id,
            )
            out = LLMClassifyOut.model_validate(raw)
        except LLMUnavailable:
            return None
        except (LLMBadOutput, ValidationError):
            continue
        out.issues = [
            i for i in out.issues if is_valid_label((i.category, i.subcategory), tax)
        ]
        if out.issues:
            return out
    return None


# --------------------------------------------------------------------------- fusion


def _top(votes: dict[Label, float]) -> tuple[Label | None, float]:
    if not votes:
        return None, 0.0
    lab = max(votes, key=lambda k: (votes[k], label_key(k)))
    return lab, votes[lab]


def fuse(
    v_knn: dict[Label, float],
    v_rule: dict[Label, float],
    v_llm: dict[Label, float] | None,
    *,
    cfg: dict[str, Any] | None = None,
    intent: str = "incident",
    rationale: str = "",
) -> ClassificationResult:
    cfg = cfg if cfg is not None else ai_config()
    ccfg = cfg.get("classifier", {})
    llm_ok = v_llm is not None
    if llm_ok:
        w = ccfg.get("weights", {"llm": 0.45, "knn": 0.35, "rule": 0.20})
    else:
        w = {"llm": 0.0, **ccfg.get("weights_no_llm", {"knn": 0.65, "rule": 0.35})}
    # a voter with no opinion gives its weight to the others
    active = {
        name: float(w.get(name, 0.0))
        for name, votes in (("llm", v_llm or {}), ("knn", v_knn), ("rule", v_rule))
        if votes and float(w.get(name, 0.0)) > 0
    }
    wsum = sum(active.values()) or 1.0
    labels = set(v_knn) | set(v_rule) | set(v_llm or {})
    scores: dict[Label, float] = {}
    for lab in labels:
        s = 0.0
        if "llm" in active:
            s += active["llm"] * (v_llm or {}).get(lab, 0.0)
        if "knn" in active:
            s += active["knn"] * v_knn.get(lab, 0.0)
        if "rule" in active:
            s += active["rule"] * v_rule.get(lab, 0.0)
        scores[lab] = s / wsum

    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], label_key(kv[0])))
    bands = ccfg.get("bands", {"high": 0.75, "medium": 0.55})
    mix = ccfg.get("conf_mix", {"score": 0.6, "margin": 0.25, "agree": 0.15})

    if not ranked or ranked[0][1] < float(ccfg.get("other_min_score", 0.20)):
        cat = ranked[0][0][0] if ranked else "Inquiry/Help"
        return ClassificationResult(
            category=cat,
            subcategory=OTHER,
            confidence=round(ranked[0][1] if ranked else 0.0, 2),
            band="low",
            intent=intent,  # type: ignore[arg-type]
            components={"llm": 0.0, "knn": 0.0, "rule": 0.0, "margin": 0.0, "agree": 0},
            alternatives=[
                {"category": c, "subcategory": s, "score": round(v, 2)}
                for (c, s), v in ranked[:3]
            ],
            rationale=rationale or "no label scored high enough",
            scores={label_key(k): round(v, 4) for k, v in ranked},
        )

    pred, top_score = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = top_score - second

    voters = {"knn": v_knn, "rule": v_rule}
    if llm_ok:
        voters["llm"] = v_llm or {}
    available = [n for n, v in voters.items() if v]
    agree = sum(1 for n in available if _top(voters[n])[0] == pred)
    agree_frac = agree / len(available) if available else 0.0

    conf = mix["score"] * top_score + mix["margin"] * margin + mix["agree"] * agree_frac
    conf = max(0.0, min(1.0, conf))
    if not llm_ok:
        conf = min(conf, float(ccfg.get("no_llm_conf_cap", 0.80)))
    conf = round(conf, 2)
    band = (
        "high"
        if conf >= bands["high"]
        else "medium"
        if conf >= bands["medium"]
        else "low"
    )

    conflict = False
    if llm_ok:
        l_top, l_s = _top(v_llm or {})
        k_top, k_s = _top(v_knn)
        cmin = float(ccfg.get("conflict_min", 0.4))
        conflict = bool(
            l_top and k_top and l_top[0] != k_top[0] and l_s >= cmin and k_s >= cmin
        )

    return ClassificationResult(
        category=pred[0],
        subcategory=pred[1],
        confidence=conf,
        band=band,  # type: ignore[arg-type]
        intent=intent,  # type: ignore[arg-type]
        conflict=conflict,
        components={
            "llm": round((v_llm or {}).get(pred, 0.0), 2) if llm_ok else 0.0,
            "knn": round(v_knn.get(pred, 0.0), 2),
            "rule": round(v_rule.get(pred, 0.0), 2),
            "margin": round(margin, 2),
            "agree": agree,
        },
        alternatives=[
            {"category": c, "subcategory": s, "score": round(v, 2)}
            for (c, s), v in ranked[1:4]
        ],
        rationale=rationale,
        scores={label_key(k): round(v, 4) for k, v in ranked},
    )


def _rationale(
    fired: list[str], v_knn: dict[Label, float], pred_hint: Label | None
) -> str:
    parts = []
    if fired:
        parts.append("rules: " + ", ".join(dict.fromkeys(fired)))
    if v_knn:
        lab, share = _top(v_knn)
        if lab:
            parts.append(f"kNN: {share:.0%} of neighbour weight is {lab[0]}/{lab[1]}")
    return "; ".join(parts)


def classify_issue(
    text: str,
    signals: Signals,
    store: TicketStore,
    llm_issue: LLMIssue | None,
    *,
    llm_was_available: bool,
    cfg: dict[str, Any] | None = None,
    rcfg: dict[str, Any] | None = None,
    where: dict[str, Any] | None = None,
    nbrs: list[TicketHit] | None = None,
) -> ClassificationResult:
    """Classify one issue span. `llm_issue` comes from Groq call #1 (None if unavailable)."""
    cfg = cfg if cfg is not None else ai_config()
    rcfg = rcfg if rcfg is not None else rules_config()
    k = int(cfg.get("classifier", {}).get("knn_k", 15))
    nb = nbrs if nbrs is not None else neighbours(store, text, k, where)
    v_knn = knn_votes(nb[:k])
    v_rule, fired = rule_votes(text, rcfg)
    v_llm = llm_votes(llm_issue, taxonomy(rcfg)) if llm_issue is not None else None
    if v_llm is None and llm_was_available:
        v_llm = {}  # LLM answered but had no opinion on this span: keep online weights
    intent = (
        llm_issue.intent
        if llm_issue is not None and llm_issue.intent in rcfg.get("intents", [])
        else detect_intent(text, rcfg)
    )
    rationale = (llm_issue.rationale if llm_issue else "") or _rationale(
        fired, v_knn, None
    )
    return fuse(v_knn, v_rule, v_llm, cfg=cfg, intent=intent, rationale=rationale)


def preliminary_scores(
    text: str,
    store: TicketStore,
    *,
    cfg: dict[str, Any] | None = None,
    rcfg: dict[str, Any] | None = None,
) -> dict[Label, float]:
    """Quick kNN + rules scores used only by the split trigger."""
    cfg = cfg if cfg is not None else ai_config()
    k = int(cfg.get("classifier", {}).get("knn_k", 15))
    res = fuse(
        knn_votes(neighbours(store, text, k)), rule_votes(text, rcfg)[0], None, cfg=cfg
    )
    out: dict[Label, float] = {}
    for key, v in res.scores.items():
        cat, sub = key.split("|", 1)
        out[(cat, sub)] = v
    return out
