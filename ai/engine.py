"""AI engine entry point (architecture §7). Owner: P2.

``run(ticket, ctx) -> EngineResult``: signals → classify (+ Groq #1) → split → per issue:
ticket search, KEDB match, priority, synthesise (+ Groq #2) + validate. Pure over its inputs:
stores and the LLM client come from ``ctx`` (injected by P3's ``backend/deps.py``).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from ai.classify import ClassifyOutcome, classify_text, classify_ticket, rule_vote
from ai.config import ai_config
from ai.kb_search import KBSearchResult, match_kb
from ai.priority import compute_priority
from ai.signals import extract, extract_signals, signals_config, ticket_text
from ai.split import IssueSpan, split_issues
from ai.synthesize import synthesize
from ai.ticket_search import find_similar
from ai.validate import validate_steps
from shared.constants import EMBED_MODEL, PIPELINE_VERSION, PRIORITY_CODES
from shared.interfaces import KBStore, LLMClient
from shared.schemas import (
    ClassificationResult,
    EngineContext,
    EngineResult,
    IssueResult,
    SanitizedTicket,
    Signals,
    Step,
)

log = logging.getLogger(__name__)

ENGINE_VERSION = "ai.engine/1.0"


class _Timer:
    def __init__(self, clock: Callable[[], float]) -> None:
        self._clock = clock
        self.ms: dict[str, int] = {}

    def add(self, name: str, t0: float) -> float:
        now = self._clock()
        self.ms[name] = self.ms.get(name, 0) + int(round((now - t0) * 1000))
        return now


def _llm_offline(llm: LLMClient | None) -> bool:
    if llm is None:
        return True
    mode = getattr(llm, "mode", None)
    if mode is not None:
        return mode == "offline"
    return not llm.available


def _phrases(kb_store: KBStore) -> list[str]:
    try:
        return list(kb_store.all_error_phrases())
    except Exception:  # noqa: BLE001
        return []


def _cross_link(
    issue: IssueResult, secondary: ClassificationResult, kb: KBSearchResult, kb_store: KBStore
) -> IssueResult:
    """Causal cross-link (§7.2): add a cited diagnostic step from the primary issue's top-2 KB
    articles that mentions the secondary issue's category (e.g. KB-0019 "Check endpoint CPU…")."""
    present = {c for s in issue.recommendation.steps for c in s.citations}
    for hit in kb.ranked[:2]:
        try:
            art = kb_store.get_article(hit.kb_id)
        except Exception:  # noqa: BLE001
            continue
        for i, text in enumerate(art.diagnostic_steps, 1):
            cid = f"{art.kb_id}#diag.{i}"
            if cid in present:
                continue
            votes, _ = rule_vote(text)
            if secondary.category not in {cat for cat, _ in votes}:
                continue
            report = validate_steps(
                [Step(text=text, type="diagnostic", citations=[cid])], {cid: text}
            )
            if not report.steps:
                continue
            link = report.steps[0].model_copy(update={"cross_link": True})
            steps = list(issue.recommendation.steps)
            n_diag = sum(1 for s in steps if s.type == "diagnostic")
            steps.insert(n_diag, link)
            rec = issue.recommendation.model_copy(update={"steps": steps, "cross_link": cid})
            return issue.model_copy(update={"recommendation": rec})
    return issue


def _issue(
    span_text: str,
    cls: ClassificationResult,
    sig: Signals,
    ticket: SanitizedTicket,
    ctx: EngineContext,
    timer: _Timer,
    clock: Callable[[], float],
    degraded: list[str],
    cfg: dict[str, Any],
) -> tuple[IssueResult, KBSearchResult]:
    t = clock()
    search = find_similar(
        span_text,
        cls,
        ctx.ticket_store,
        filters=ctx.filters,
        role=ctx.role,
        exclude_ids=(ticket.ticket_id,),
        cfg=cfg,
    )
    kb = match_kb(span_text, sig, cls, ctx.kb_store, ctx.feedback_prior, cfg)
    t = timer.add("retrieve", t)

    top = kb.ranked[0] if kb.ranked else None
    hint = None
    if top is not None:
        try:
            hint = ctx.kb_store.get_article(top.kb_id).priority_hint
        except Exception:  # noqa: BLE001
            hint = None
    priority = compute_priority(
        cls,
        sig,
        text=span_text,
        kb_id=top.kb_id if top else None,
        kb_hint=hint,
        kb_relevance=kb.relevance,
        cfg=cfg,
    )
    t = timer.add("priority", t)

    rec, _pack, synth_deg = synthesize(
        span_text, sig, cls, kb.hits, search.similar, ctx.kb_store, ctx.llm, cfg
    )
    degraded.extend(synth_deg)
    timer.add("synthesize", t)
    issue = IssueResult(
        text_span=span_text,
        classification=cls,
        priority=priority,
        similar=search.similar,
        keyword_only=search.keyword_only,
        kb_matches=kb.hits,
        kb_relevance=round(kb.relevance, 4),
        kb_ambiguous=kb.ambiguous,
        possible_major_incident=search.possible_major_incident,
        recommendation=rec,
        pattern=search.pattern,
        signals=sig.model_dump(),
    )
    return issue, kb


def run(
    ticket: SanitizedTicket,
    ctx: EngineContext,
    *,
    cfg: dict[str, Any] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> EngineResult:
    """Analyse one sanitised ticket. Never raises for LLM problems (deterministic fallbacks)."""
    cfg = cfg or ai_config()
    timer = _Timer(clock)
    t_start = clock()
    if ctx.clarification and not ticket.clarification:
        ticket = ticket.model_copy(update={"clarification": ctx.clarification})

    t = clock()
    signals = extract_signals(ticket, ctx.kb_store)
    text = ticket_text(ticket)
    t = timer.add("signals", t)

    outcome: ClassifyOutcome = classify_ticket(text, signals, ctx.ticket_store, ctx.llm, cfg=cfg)
    spans, split_deg = split_issues(text, signals, outcome, cfg)
    timer.add("classify", t)

    degraded: list[str] = list(outcome.degraded) + list(split_deg)
    phrases = _phrases(ctx.kb_store) if len(spans) > 1 else []

    issues: list[IssueResult] = []
    kbs: list[KBSearchResult] = []
    for span in spans:
        if len(spans) == 1:
            span_text, cls, sig = text, outcome.result, signals
        else:
            span_text, cls, sig = _span_inputs(span, ticket, ctx, phrases, cfg)
        issue, kb = _issue(span_text, cls, sig, ticket, ctx, timer, clock, degraded, cfg)
        issues.append(issue)
        kbs.append(kb)

    # Primary = more severe priority; ties → first mentioned (stable sort).
    order = sorted(range(len(issues)), key=lambda i: PRIORITY_CODES.index(issues[i].priority.code))
    issues = [issues[i] for i in order]
    kbs = [kbs[i] for i in order]
    if len(issues) == 2:
        issues[0] = _cross_link(issues[0], issues[1].classification, kbs[0], ctx.kb_store)

    if _llm_offline(ctx.llm):
        degraded.append("offline")
    timer.add("total", t_start)
    return EngineResult(
        signals=signals,
        issues=issues,
        degraded=list(dict.fromkeys(degraded)),
        timings_ms=timer.ms,
        versions={
            "pipeline": PIPELINE_VERSION,
            "engine": ENGINE_VERSION,
            "embed_model": cfg.get("embedding", {}).get("model", EMBED_MODEL),
            "classify_model": cfg["llm"]["classify_model"],
            "synth_model": cfg["llm"]["synth_model"],
        },
    )


def _span_inputs(
    span: IssueSpan,
    ticket: SanitizedTicket,
    ctx: EngineContext,
    phrases: list[str],
    cfg: dict[str, Any],
) -> tuple[str, ClassificationResult, Signals]:
    sig = extract(
        span.text,
        phrases,
        environment=ticket.environment,
        location=ticket.location,
        cfg=signals_config(),
    )
    cls = classify_text(
        span.text, ctx.ticket_store, llm_scores=span.llm_scores, llm_intent=span.intent, cfg=cfg
    ).result
    return span.text, cls, sig


class AIEngine:
    """``shared.interfaces.Engine`` implementation (what ``backend/deps.py`` should construct)."""

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        self._cfg = cfg

    def run(self, ticket: SanitizedTicket, ctx: EngineContext) -> EngineResult:
        return run(ticket, ctx, cfg=self._cfg)
