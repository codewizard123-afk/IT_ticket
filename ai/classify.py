"""Fused classifier: kNN + LLM + rules (architecture §7.3). Owner: P2.

``score(s) = w_llm·v_llm + w_knn·v_knn + w_rule·v_rule``
``conf     = clip(0.6·score(pred) + 0.25·margin + 0.15·agree/n_voters, 0, 1)``

When the LLM is unavailable the weights fall back to kNN 0.65 / rules 0.35, confidence is capped
at 0.80 and ``llm_classify_fallback`` is reported. The LLM call (#1) also proposes the issue split
(§7.2), which ``ai/split.py`` consumes.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from ai.config import ai_config, load_yaml
from ai.prompts import render
from shared.constants import INTENTS, SUBCATEGORY_OTHER, TAXONOMY
from shared.interfaces import LLMClient, LLMUnavailableError, TicketStore
from shared.schemas import Alternative, ClassificationResult, Signals, TicketHit

log = logging.getLogger(__name__)

Label = tuple[str, str]
FALLBACK_LABEL: Label = ("Inquiry/Help", SUBCATEGORY_OTHER)

FEW_SHOT: tuple[tuple[str, str, str], ...] = (
    ("VPN client says gateway not responding when working from home", "Network/VPN", "incident"),
    ("Wi-Fi keeps dropping in the meeting room", "Network/Wireless", "incident"),
    ("Oracle report times out with ORA-01013", "Database/Oracle", "incident"),
    ("New joiner gets SQL Server error 18456", "Database/MS SQL Server", "incident"),
    ("External monitor flickers and goes black", "Hardware/Monitor", "incident"),
    ("Laptop CPU at 100 percent and fan very loud", "Hardware/CPU", "incident"),
    ("Outlook crashes at start-up with 0xc0000005", "Software/Email", "incident"),
    ("Windows update fails with 0x80070002", "Software/Operating System", "incident"),
    ("Antivirus quarantined an attachment", "Inquiry/Help/Antivirus", "incident"),
    ("Need access to the HR portal", "Inquiry/Help/Internal Application", "access_request"),
)


# --- labels ------------------------------------------------------------------------------------
def parse_label(raw: str) -> Label | None:
    """``"Inquiry/Help/Antivirus"`` → ``("Inquiry/Help", "Antivirus")``; invalid → ``None``."""
    if "/" not in raw:
        return None
    cat, sub = (p.strip() for p in raw.rsplit("/", 1))
    if cat not in TAXONOMY:
        return None
    if sub not in TAXONOMY[cat] and sub != SUBCATEGORY_OTHER:
        return None
    return cat, sub


def label_str(label: Label) -> str:
    return f"{label[0]}/{label[1]}"


def _normalise(scores: dict[Label, float]) -> dict[Label, float]:
    scores = {k: max(0.0, v) for k, v in scores.items() if v > 0}
    total = sum(scores.values())
    return {k: v / total for k, v in scores.items()} if total > 0 else {}


def _top(scores: dict[Label, float]) -> tuple[Label | None, float]:
    if not scores:
        return None, 0.0
    label = max(scores, key=lambda k: (scores[k], -_order(k)))
    return label, scores[label]


@lru_cache(maxsize=1)
def _label_order() -> dict[Label, int]:
    order = [(c, s) for c, subs in TAXONOMY.items() for s in subs]
    return {lab: i for i, lab in enumerate(order)}


def _order(label: Label) -> int:
    return _label_order().get(label, 999)


# --- config ------------------------------------------------------------------------------------
@lru_cache(maxsize=4)
def rules_config(config_dir: Path | None = None) -> dict[str, Any]:
    return load_yaml("rules.yaml", config_dir)


# --- voter A: kNN ------------------------------------------------------------------------------
def knn_vote(
    text: str, ticket_store: TicketStore, k: int, *, power: float = 1.0, min_rel: float = 0.0
) -> tuple[dict[Label, float], list[TicketHit]]:
    """``v_knn(s) = Σ sim_i^p·[label_i = s] / Σ sim_i^p`` over historical neighbours.

    ``p = 1`` and ``min_rel = 0`` is the plain §7.3 formula. ``power`` sharpens the vote and
    ``min_rel`` drops neighbours scoring below ``min_rel × best`` (both tuned by P1's eval).
    """
    try:
        hits = ticket_store.query(text, k=k, collections=("tickets_hist",))
    except Exception:  # noqa: BLE001 - store failure → no kNN vote, keep going
        log.warning("kNN query failed")
        return {}, []
    votes: dict[Label, float] = defaultdict(float)
    best = max((h.score for h in hits), default=0.0)
    for h in hits:
        if not h.category or not h.subcategory or h.score < min_rel * best:
            continue
        label = parse_label(f"{h.category}/{h.subcategory}")
        if label:
            votes[label] += max(0.0, h.score) ** power
    return _normalise(dict(votes)), hits


def _knn(
    text: str, ticket_store: TicketStore, cfg: dict[str, Any]
) -> tuple[dict[Label, float], list[TicketHit]]:
    c = cfg["classifier"]
    return knn_vote(
        text,
        ticket_store,
        int(c["knn_k"]),
        power=float(c.get("knn_power", 1.0)),
        min_rel=float(c.get("knn_min_rel", 0.0)),
    )


# --- voter C: rules ----------------------------------------------------------------------------
def rule_vote(text: str, cfg: dict[str, Any] | None = None) -> tuple[dict[Label, float], list[str]]:
    """Weighted keyword/regex rules → normalised label distribution + matched terms."""
    cfg = cfg if cfg is not None else rules_config()
    votes: dict[Label, float] = defaultdict(float)
    matched: list[str] = []
    for rule in cfg.get("subcategory_rules") or []:
        label = parse_label(rule["label"])
        if label is None:
            continue
        flags = 0 if rule.get("case_sensitive") else re.IGNORECASE
        m = re.search(rule["pattern"], text, flags)
        if m:
            votes[label] += float(rule.get("weight", 1.0))
            term = m.group(0).strip()
            if term.lower() not in (t.lower() for t in matched):
                matched.append(term)
    return _normalise(dict(votes)), matched


def rule_intent(text: str, cfg: dict[str, Any] | None = None) -> str:
    cfg = cfg if cfg is not None else rules_config()
    for intent in ("access_request", "how_to", "service_request"):
        pat = (cfg.get("intent_rules") or {}).get(intent)
        if pat and re.search(pat, text, re.IGNORECASE):
            return intent
    return "incident"


# --- voter B: LLM ------------------------------------------------------------------------------
class LLMIssue(BaseModel):
    text_span: str = ""
    category: str
    subcategory: str
    intent: str = "incident"
    scores: dict[str, float] = Field(default_factory=dict)


class LLMClassifyOut(BaseModel):
    issues: list[LLMIssue] = Field(min_length=1)
    rationale: str | None = None


@dataclass
class LLMIssueVote:
    text_span: str
    scores: dict[Label, float]
    intent: str


def _taxonomy_text() -> str:
    return "\n".join(f"- {c}: {', '.join(subs)}" for c, subs in TAXONOMY.items())


def _examples_text() -> str:
    return "\n".join(
        json.dumps(
            {
                "ticket": text,
                "issues": [
                    {
                        "category": parse_label(lab)[0],  # type: ignore[index]
                        "subcategory": parse_label(lab)[1],  # type: ignore[index]
                        "intent": intent,
                    }
                ],
            }
        )
        for text, lab, intent in FEW_SHOT
    )


def build_classify_prompt(
    text: str, signals: Signals, neighbours: list[TicketHit]
) -> tuple[str, str]:
    system = render("classify.j2", taxonomy=_taxonomy_text(), examples=_examples_text())
    nb = "\n".join(
        f"- {h.category}/{h.subcategory}: {(h.excerpt or h.title or '')[:160]}"
        for h in neighbours[:5]
    )
    sig = signals.model_dump(include={"error_codes", "products", "scope", "multi_issue_markers"})
    user = render(
        "classify_user.j2",
        neighbours=nb or "- (none)",
        signals=json.dumps(sig),
        ticket=text.replace("</ticket>", "</ ticket>"),
    )
    return system, user


def _validate_llm(raw: dict[str, Any]) -> tuple[list[LLMIssueVote], str | None]:
    out = LLMClassifyOut.model_validate(raw)
    votes: list[LLMIssueVote] = []
    for issue in out.issues[:2]:
        scores: dict[Label, float] = {}
        for k, v in issue.scores.items():
            lab = parse_label(k)
            if lab is not None:
                scores[lab] = scores.get(lab, 0.0) + float(v)
        top = parse_label(f"{issue.category}/{issue.subcategory}")
        if top is None and not scores:
            raise ValueError(f"label outside taxonomy: {issue.category}/{issue.subcategory}")
        if top is not None and top not in scores:
            scores[top] = max(scores.values(), default=0.0) + 0.5 if scores else 1.0
        intent = issue.intent if issue.intent in INTENTS else "incident"
        votes.append(LLMIssueVote(issue.text_span.strip(), _normalise(scores), intent))
    return votes, out.rationale


def llm_vote(
    text: str,
    signals: Signals,
    neighbours: list[TicketHit],
    llm: LLMClient,
    model: str | None = None,
) -> tuple[list[LLMIssueVote], str | None]:
    """Groq call #1 (classify + split). Validated with Pydantic, retried once, else raises."""
    system, user = build_classify_prompt(text, signals, neighbours)
    last_err: Exception | None = None
    for attempt in range(2):
        prompt = user if attempt == 0 else user + "\n\nReturn valid JSON exactly as specified."
        raw = llm.complete_json(
            purpose="classify", system=system, user=prompt, model=model, max_tokens=400
        )
        try:
            return _validate_llm(raw)
        except (ValidationError, ValueError, TypeError) as exc:
            last_err = exc
    raise LLMUnavailableError(f"classifier output invalid twice: {type(last_err).__name__}")


# --- fusion ------------------------------------------------------------------------------------
@dataclass
class Votes:
    llm: dict[Label, float] = field(default_factory=dict)
    knn: dict[Label, float] = field(default_factory=dict)
    rule: dict[Label, float] = field(default_factory=dict)
    llm_ok: bool = False
    intent: str = "incident"
    rule_terms: list[str] = field(default_factory=list)
    rationale: str | None = None


def fuse(votes: Votes, cfg: dict[str, Any] | None = None) -> tuple[ClassificationResult, dict]:
    """Combine the voters into a ``ClassificationResult`` and the fused score table."""
    ccfg = (cfg or ai_config())["classifier"]
    if votes.llm_ok:
        w = ccfg["weights"]
        weights = {"llm": w["llm"], "knn": w["knn"], "rule": w["rule"]}
    else:
        w = ccfg["weights_no_llm"]
        weights = {"llm": 0.0, "knn": w["knn"], "rule": w["rule"]}
    voters = {"llm": votes.llm, "knn": votes.knn, "rule": votes.rule}
    labels = set().union(*voters.values())
    fused = {
        lab: sum(weights[name] * v.get(lab, 0.0) for name, v in voters.items()) for lab in labels
    }
    ranked = sorted(fused, key=lambda k: (-fused[k], _order(k)))
    pred = ranked[0] if ranked else FALLBACK_LABEL
    top = fused.get(pred, 0.0)
    second = fused[ranked[1]] if len(ranked) > 1 else 0.0
    margin = top - second

    active = [n for n in voters if weights[n] > 0]
    tops = {n: _top(voters[n])[0] for n in active}
    agree = sum(1 for n in active if tops[n] == pred)
    n_voters = 3 if votes.llm_ok else 2
    conf = 0.6 * top + 0.25 * margin + 0.15 * agree / n_voters
    conf = min(max(conf, 0.0), 1.0)
    if not votes.llm_ok:
        conf = min(conf, float(ccfg["no_llm_conf_cap"]))
    if sum(1 for n in active if voters[n]) <= 1:  # a lone voter is never confident
        conf = min(conf, float(ccfg["single_voter_cap"]))

    category, subcategory = pred
    if top < float(ccfg["other_min"]):
        subcategory = SUBCATEGORY_OTHER
        if not ranked:
            category = FALLBACK_LABEL[0]
    bands = ccfg["bands"]
    if subcategory == SUBCATEGORY_OTHER:
        band = "low"
        conf = min(conf, float(bands["medium"]) - 0.01)
    elif conf >= bands["high"]:
        band = "high"
    elif conf >= bands["medium"]:
        band = "medium"
    else:
        band = "low"

    llm_top, llm_s = _top(votes.llm)
    knn_top, knn_s = _top(votes.knn)
    cmin = float(ccfg["conflict_min"])
    conflict = bool(
        llm_top and knn_top and llm_top[0] != knn_top[0] and llm_s >= cmin and knn_s >= cmin
    )

    alternatives = [
        Alternative(category=c, subcategory=s, score=round(fused[(c, s)], 2))
        for c, s in ranked[1:4]
    ]
    rationale_parts = []
    if votes.rationale:
        rationale_parts.append(votes.rationale.strip()[:200])
    if votes.rule_terms:
        rationale_parts.append("mentions " + ", ".join(f"'{t}'" for t in votes.rule_terms[:4]))
    if knn_top:
        rationale_parts.append(f"nearest tickets lean {label_str(knn_top)} ({knn_s:.2f})")
    result = ClassificationResult(
        category=category,
        subcategory=subcategory,
        confidence=round(conf, 2),
        band=band,  # type: ignore[arg-type]
        intent=votes.intent,  # type: ignore[arg-type]
        components={
            "llm": round(votes.llm.get(pred, 0.0), 2),
            "knn": round(votes.knn.get(pred, 0.0), 2),
            "rule": round(votes.rule.get(pred, 0.0), 2),
            "margin": round(margin, 2),
            "agree": float(agree),
        },
        alternatives=alternatives,
        rationale="; ".join(rationale_parts) or "no strong signal; low confidence",
        conflict=conflict,
    )
    return result, fused


@dataclass
class ClassifyOutcome:
    """Ticket-level classification plus everything split/search need afterwards."""

    result: ClassificationResult
    fused: dict[Label, float]
    votes: Votes
    llm_issues: list[LLMIssueVote] = field(default_factory=list)
    knn_hits: list[TicketHit] = field(default_factory=list)
    degraded: list[str] = field(default_factory=list)


def classify_text(
    text: str,
    ticket_store: TicketStore,
    *,
    llm_scores: dict[Label, float] | None = None,
    llm_intent: str | None = None,
    rationale: str | None = None,
    cfg: dict[str, Any] | None = None,
) -> ClassifyOutcome:
    """Classify a text span without a new LLM call (``llm_scores`` may come from call #1)."""
    cfg = cfg or ai_config()
    knn, hits = _knn(text, ticket_store, cfg)
    rule, terms = rule_vote(text)
    intent = llm_intent or rule_intent(text)
    votes = Votes(
        llm=llm_scores or {},
        knn=knn,
        rule=rule,
        llm_ok=bool(llm_scores),
        intent=intent,
        rule_terms=terms,
        rationale=rationale,
    )
    result, fused = fuse(votes, cfg)
    return ClassifyOutcome(result=result, fused=fused, votes=votes, knn_hits=hits)


def classify_ticket(
    text: str,
    signals: Signals,
    ticket_store: TicketStore,
    llm: LLMClient | None,
    *,
    cfg: dict[str, Any] | None = None,
) -> ClassifyOutcome:
    """Ticket-level classification, including Groq call #1 (with deterministic fallback)."""
    cfg = cfg or ai_config()
    knn, hits = _knn(text, ticket_store, cfg)
    rule, terms = rule_vote(text)
    degraded: list[str] = []
    llm_issues: list[LLMIssueVote] = []
    rationale = None
    if llm is not None and llm.available:
        try:
            llm_issues, rationale = llm_vote(
                text, signals, hits, llm, model=cfg["llm"]["classify_model"]
            )
        except LLMUnavailableError:
            llm_issues = []
    if not llm_issues:
        degraded.append("llm_classify_fallback")

    # Ticket-level LLM distribution: the primary issue's scores (split handled later).
    llm_scores = llm_issues[0].scores if llm_issues else {}
    intent = llm_issues[0].intent if llm_issues else rule_intent(text)
    votes = Votes(
        llm=llm_scores,
        knn=knn,
        rule=rule,
        llm_ok=bool(llm_scores),
        intent=intent,
        rule_terms=terms,
        rationale=rationale,
    )
    result, fused = fuse(votes, cfg)
    return ClassifyOutcome(
        result=result,
        fused=fused,
        votes=votes,
        llm_issues=llm_issues,
        knn_hits=hits,
        degraded=degraded,
    )


def top_categories(fused: dict[Label, float]) -> list[tuple[str, float]]:
    """Fused score summed per category, best first (used by the split trigger)."""
    per_cat: dict[str, float] = defaultdict(float)
    for (cat, _), v in fused.items():
        per_cat[cat] += v
    return sorted(per_cat.items(), key=lambda kv: -kv[1])
