"""Decision gate: recommend, clarify or escalate, per issue (FR-05, architecture §8.2-§8.4).

Order (§8.3):

1. any hard rule (E4, E5)                                → escalate
2. clarification round 0 and an ambiguity trigger        → clarify
3. any soft rule (E1, E2, E3, E6, E7)                    → escalate
4. otherwise                                             → recommend

Round 1 skips step 2, so a second low-confidence result escalates: the loop always terminates.
Everything here is deterministic and pure over its inputs.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from backend.config import clarify_config, escalation_config, kb_clauses
from shared.constants import HARD_RULES
from shared.schemas import DecisionKind, EscalationReason, IssueResult, Signals

_WORD = re.compile(r"\w+")


@dataclass
class IssueDecision:
    decision: DecisionKind
    reasons: list[EscalationReason] = field(default_factory=list)
    clarify_triggers: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)

    @property
    def hard(self) -> bool:
        return any(r.rule in HARD_RULES for r in self.reasons)

    @property
    def rules(self) -> list[str]:
        return [r.rule for r in self.reasons]


@dataclass(frozen=True)
class GateInput:
    """Ticket-level facts the gate needs besides the engine's issue result."""

    signals: Signals
    text: str  # redacted title + description (+ clarification)
    secret_detected: bool = False
    environment: str | None = None
    clarification_round: int = 0


# =============================================================================================
# Rules
# =============================================================================================
def _thresholds(cfg: Mapping[str, Any]) -> dict[str, Any]:
    return dict(cfg.get("thresholds", {}))


def _fmt(cfg: Mapping[str, Any], rule: str, **kw: Any) -> str:
    template = cfg.get("rule_text", {}).get(rule, rule)
    try:
        return template.format(**kw)
    except (KeyError, IndexError):
        return template


def rule_e1(issue: IssueResult, cfg: Mapping[str, Any]) -> EscalationReason | None:
    th = _thresholds(cfg)
    if issue.kb_matches and issue.kb_relevance >= th["tau_kb"]:
        return None
    origins = set(cfg.get("verified_origins", ["curated_verified", "analyst_verified"]))
    if any(
        h.resolution_origin in origins and h.score >= th["verified_min_cos"] for h in issue.similar
    ):
        return None
    return EscalationReason(rule="E1", text=_fmt(cfg, "E1", **th))


def rule_e2(issue: IssueResult, cfg: Mapping[str, Any]) -> EscalationReason | None:
    c = issue.classification
    if c.band != "low" and not c.conflict:
        return None
    return EscalationReason(
        rule="E2",
        text=_fmt(cfg, "E2", band=c.band, conflict=" (voter conflict)" if c.conflict else ""),
    )


def rule_e3(issue: IssueResult, cfg: Mapping[str, Any]) -> EscalationReason | None:
    rec = issue.recommendation
    if not rec.insufficient_evidence and rec.steps:
        return None
    return EscalationReason(rule="E3", text=_fmt(cfg, "E3"))


def rule_e4(issue: IssueResult, cfg: Mapping[str, Any]) -> EscalationReason | None:
    p = issue.priority
    if p.code != "P1" and not issue.possible_major_incident:
        return None
    major = "; possible major incident" if issue.possible_major_incident else ""
    return EscalationReason(
        rule="E4", text=_fmt(cfg, "E4", code=p.code, level=p.level, major=major)
    )


def rule_e5(gate: GateInput, cfg: Mapping[str, Any]) -> EscalationReason | None:
    flags = list(gate.signals.security_flags)
    if gate.secret_detected:
        flags.append("secret in ticket text")
    if not flags:
        return None
    return EscalationReason(rule="E5", text=_fmt(cfg, "E5", flags=", ".join(flags)))


def clause_holds(when: Mapping[str, Any], gate: GateInput) -> bool:
    """Evaluate one ``kb_escalation_clauses.yaml`` predicate (all keys must hold)."""
    s = gate.signals
    text = gate.text.lower()
    for key, want in when.items():
        if key == "scope_in":
            ok = s.scope in want
        elif key == "min_users":
            ok = (s.affected_users or 0) >= int(want)
        elif key == "business_critical":
            ok = s.business_critical is bool(want)
        elif key == "security":
            ok = bool(s.security_flags or gate.secret_detected) is bool(want)
        elif key == "environment_in":
            env = (s.environment or gate.environment or "").lower()
            ok = env in [str(w).lower() for w in want]
        elif key == "change_trigger":
            ok = bool(s.change_trigger) is bool(want)
        elif key == "text_any":
            ok = any(str(p).lower() in text for p in want)
        elif key == "text_all":
            ok = all(str(p).lower() in text for p in want)
        else:  # unknown predicate: fail closed so a typo never escalates everything
            ok = False
        if not ok:
            return False
    return True


def rule_e6(
    issue: IssueResult,
    gate: GateInput,
    cfg: Mapping[str, Any],
    clauses: Mapping[str, list[dict[str, Any]]],
) -> list[EscalationReason]:
    th = _thresholds(cfg)
    top = [m for m in issue.kb_matches if m.score >= th["tau_show"]][: th.get("e6_top_articles", 2)]
    out: list[EscalationReason] = []
    for m in top:
        for clause in clauses.get(m.kb_id, []):
            if clause_holds(clause.get("when", {}), gate):
                out.append(EscalationReason(rule="E6", text=f"{m.kb_id}: {clause['text']}"))
    return out


def rule_e7(issue: IssueResult, cfg: Mapping[str, Any]) -> EscalationReason | None:
    steps = issue.recommendation.steps
    l1 = [s for s in steps if s.role == "L1"]
    remaining = [s for s in steps if not s.already_attempted]
    if not l1 or not remaining:
        return None
    if all(s.already_attempted for s in l1) and all(s.role == "L2" for s in remaining):
        return EscalationReason(rule="E7", text=_fmt(cfg, "E7"))
    return None


def escalation_reasons(
    issue: IssueResult,
    gate: GateInput,
    cfg: Mapping[str, Any] | None = None,
    clauses: Mapping[str, list[dict[str, Any]]] | None = None,
) -> list[EscalationReason]:
    """Every rule that fires for this issue, hard rules first."""
    cfg = cfg if cfg is not None else escalation_config()
    clauses = clauses if clauses is not None else kb_clauses()
    hard = [r for r in (rule_e4(issue, cfg), rule_e5(gate, cfg)) if r]
    soft = [r for r in (rule_e1(issue, cfg), rule_e2(issue, cfg), rule_e3(issue, cfg)) if r]
    soft += rule_e6(issue, gate, cfg, clauses)
    e7 = rule_e7(issue, cfg)
    if e7:
        soft.append(e7)
    return hard + soft


# =============================================================================================
# Clarify
# =============================================================================================
def word_count(text: str) -> int:
    return len(_WORD.findall(text))


def clarify_triggers(issue: IssueResult, gate: GateInput, cfg: Mapping[str, Any]) -> list[str]:
    th = _thresholds(cfg)
    c = issue.classification
    out: list[str] = []
    if issue.kb_ambiguous:
        out.append("kb_ambiguous")
    if c.conflict:
        out.append("conflict")
    s = gate.signals
    if word_count(gate.text) < th["sparse_words"] and not s.products and not s.error_codes:
        out.append("sparse")
    if c.band == "low":
        runner_up = max((a.score for a in c.alternatives), default=0.0)
        if c.confidence - runner_up < th["low_gap"]:
            out.append("low_gap")
    return out


def _pair_question(issue: IssueResult, tmpl: Mapping[str, Any]) -> str | None:
    subs = [m.subcategory or m.title for m in issue.kb_matches[:2]]
    if len(subs) < 2 or not all(subs):
        alts = issue.classification.alternatives
        if not alts:
            return None
        subs = [issue.classification.subcategory, alts[0].subcategory]
    a, b = str(subs[0]), str(subs[1])
    if a == b:
        return None
    for entry in tmpl.get("ambiguous_pairs", []):
        if {x.lower() for x in entry["pair"]} == {a.lower(), b.lower()}:
            return str(entry["question"])
    generic = tmpl.get("generic_pair")
    return generic.format(a=a, b=b) if generic else None


def clarifying_questions(
    issue: IssueResult,
    gate: GateInput,
    triggers: Iterable[str],
    tmpl: Mapping[str, Any] | None = None,
) -> list[str]:
    """≤ ``max_questions`` deterministic questions (§8.3)."""
    tmpl = tmpl if tmpl is not None else clarify_config()
    qs = tmpl.get("questions", {})
    out: list[str] = []
    triggers = set(triggers)
    if triggers & {"kb_ambiguous", "conflict", "low_gap"}:
        q = _pair_question(issue, tmpl)
        if q:
            out.append(q)
    s = gate.signals
    if not s.error_codes and qs.get("no_error_code"):
        out.append(qs["no_error_code"])
    if s.affected_users is None and s.scope == "single" and qs.get("unknown_scope"):
        out.append(qs["unknown_scope"])
    if not s.onset and not s.change_trigger and qs.get("no_onset"):
        out.append(qs["no_onset"])
    return list(dict.fromkeys(out))[: int(tmpl.get("max_questions", 3))]


# =============================================================================================
# Gate
# =============================================================================================
def decide(
    issue: IssueResult,
    gate: GateInput,
    cfg: Mapping[str, Any] | None = None,
    clauses: Mapping[str, list[dict[str, Any]]] | None = None,
    tmpl: Mapping[str, Any] | None = None,
) -> IssueDecision:
    cfg = cfg if cfg is not None else escalation_config()
    reasons = escalation_reasons(issue, gate, cfg, clauses)
    hard = [r for r in reasons if r.rule in HARD_RULES]
    if hard:
        return IssueDecision("escalate", reasons)
    max_rounds = int(_thresholds(cfg).get("max_clarify_rounds", 1))
    if gate.clarification_round < max_rounds:
        triggers = clarify_triggers(issue, gate, cfg)
        if triggers:
            questions = clarifying_questions(issue, gate, triggers, tmpl)
            return IssueDecision("clarify", reasons, triggers, questions)
    if reasons:
        return IssueDecision("escalate", reasons)
    return IssueDecision("recommend", [])


def ticket_decision(decisions: Iterable[IssueDecision]) -> DecisionKind:
    """Ticket level (§8.4): escalate if any issue escalates, else clarify if any, else recommend."""
    kinds = {d.decision for d in decisions}
    if "escalate" in kinds:
        return "escalate"
    if "clarify" in kinds:
        return "clarify"
    return "recommend"
