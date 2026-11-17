"""Cards for one analysis result (architecture §9.1, Analyze page).

Rules enforced here (§9.2): no step is rendered without its source badge, synthetic data is
always badged, and every chip has colour plus a text label. All ticket-derived text is escaped
before it goes into HTML.
"""

from __future__ import annotations

import html
import json
from collections.abc import Callable
from typing import Any

import streamlit as st

from shared.schemas import AnalysisResponse, KedbDetails, KedbMatch, Step
from ui.client import ApiClient, ApiError
from ui.components.badges import chip, origin_is_synthetic, plain_chip, source_badges

# feedback hook: (target_type, target_ref, key, issue_idx) -> None. Set by the Analyze page.
FeedbackHook = Callable[[str, str | None, str, int], None]


def _md(text: str) -> None:
    st.markdown(text, unsafe_allow_html=True)


def _esc(text: Any) -> str:
    return html.escape(str(text)) if text is not None else ""


def _no_feedback(target_type: str, target_ref: str | None, key: str, issue_idx: int) -> None:
    return None


# ---------------------------------------------------------------------------- cards
def classification_card(
    classification: Any, cls_details: Any, *, feedback: FeedbackHook = _no_feedback, idx: int = 0
) -> None:
    with st.container(border=True):
        st.markdown("**Classification**")
        band = cls_details.band if cls_details else None
        _md(
            f"{_esc(classification.category)} / <b>{_esc(classification.subcategory)}</b> "
            f"&nbsp;{chip('band', band, 'CONFIDENCE ') if band else ''}"
        )
        st.progress(float(classification.confidence), text=f"{classification.confidence:.0%}")
        if cls_details:
            if cls_details.conflict:
                _md(plain_chip("CONFLICT between voters", "#c62828"))
            if cls_details.components:
                comps = " · ".join(f"{k} {v:g}" for k, v in cls_details.components.items())
                st.caption(f"Components: {comps}")
            if cls_details.rationale:
                st.caption(f"Why: {cls_details.rationale}")
            if cls_details.alternatives:
                alts = ", ".join(
                    f"{a.category}/{a.subcategory} ({a.score:.2f})"
                    for a in cls_details.alternatives
                )
                st.caption(f"Alternatives: {alts}")
        feedback(
            "classification",
            f"{classification.category}/{classification.subcategory}",
            f"cls-{idx}",
            idx,
        )


def priority_card(
    priority: Any, pr_details: Any, *, feedback: FeedbackHook = _no_feedback, idx: int = 0
) -> None:
    with st.container(border=True):
        st.markdown("**Priority**")
        code = f" ({pr_details.code})" if pr_details else ""
        _md(f"{chip('priority', priority.level)}{_esc(code)}")
        st.write(priority.reason)
        if pr_details:
            st.caption(f"Impact {pr_details.impact} × urgency {pr_details.urgency}")
            for d in pr_details.drivers:
                st.caption(f"• {d}")


def similar_card(resp: AnalysisResponse, *, feedback: FeedbackHook = _no_feedback) -> None:
    details = {s.incident_id: s for s in resp.details.similar}
    with st.container(border=True):
        st.markdown(f"**Similar incidents** ({len(resp.similar_incidents)})")
        if not resp.similar_incidents:
            st.caption("No similar incidents above the display threshold.")
        for i, inc in enumerate(resp.similar_incidents):
            det = details.get(inc.incident_id)
            origin = det.resolution_origin if det else "none"
            tier = chip("tier", det.tier) if det else ""
            _md(
                f"<b>{_esc(inc.incident_id)}</b> {tier} {chip('origin', origin)} "
                f"score {inc.similarity_score:.2f}"
            )
            if det and det.excerpt:
                st.caption(f"“{det.excerpt}”")
            if inc.resolution:
                note = (
                    " (synthetic, never cited as evidence)" if origin_is_synthetic(origin) else ""
                )
                st.caption(f"Resolution{note}: {inc.resolution}")
            if det and det.why:
                st.caption(f"Why: {det.why}")
            feedback("similar_incident", inc.incident_id, f"sim-{i}", 0)
        if resp.details.keyword_only:
            with st.expander(f"Keyword-only matches ({len(resp.details.keyword_only)})"):
                for k in resp.details.keyword_only:
                    _md(f"{_esc(k.incident_id)} {chip('origin', k.resolution_origin)}")
                    if k.excerpt:
                        st.caption(k.excerpt)


def kedb_card(
    match: KedbMatch,
    kedb: KedbDetails | None,
    *,
    feedback: FeedbackHook = _no_feedback,
    idx: int = 0,
) -> None:
    with st.container(border=True):
        st.markdown("**KEDB match**")
        if not match.found:
            _md(plain_chip("NO KEDB MATCH", "#9e9e9e"))
            st.caption("No known-error article above the relevance threshold.")
            return
        title = f" · {kedb.title}" if kedb else ""
        _md(f"<b>{_esc(match.error_code)}</b>{_esc(title)}")
        if kedb:
            st.caption(f"Score {kedb.score:.2f} · matched field: {kedb.matched_field or '-'}")
            if kedb.matched_error_code:
                st.caption(f"Matched code in ticket: `{kedb.matched_error_code}`")
            if kedb.excerpt:
                st.caption(f"“{kedb.excerpt}”")
            if kedb.why:
                st.caption(f"Why: {kedb.why}")
            if kedb.alternatives:
                st.caption(
                    "Alternatives: "
                    + ", ".join(f"{a.kb_id} ({a.score:.2f})" for a in kedb.alternatives)
                )
        if match.solution:
            st.caption(f"First resolution step: {match.solution}")
        feedback("kedb_match", match.error_code, f"kedb-{idx}", idx)


def step_html(n: int, step: Step) -> str:
    """One numbered step with its source badge, role tag and 'already tried' styling."""
    badges = source_badges(step.citations)  # raises if uncited -> never shown without a badge
    role = chip("role", step.role)
    kind = plain_chip(step.type.upper(), "#00695c" if step.type == "diagnostic" else "#283593")
    text = _esc(step.text)
    if step.already_attempted:
        return (
            f'<div style="opacity:0.5">{n}. <s>{text}</s> '
            f"{plain_chip('ALREADY TRIED', '#757575')} {role} {kind} "
            f"<code>{_esc(badges)}</code></div>"
        )
    return f"<div>{n}. {text} {role} {kind} <code>{_esc(badges)}</code></div>"


def steps_card(
    steps: list[Step],
    source: str,
    *,
    feedback: FeedbackHook = _no_feedback,
    idx: int = 0,
) -> None:
    with st.container(border=True):
        st.markdown("**Recommended solution**")
        _md(plain_chip(f"SOURCE: {source}", "#37474f"))
        if not steps:
            st.caption("No grounded steps: there was not enough evidence to recommend any.")
        for n, step in enumerate(steps, 1):
            if not step.citations:  # defensive: schema forbids this, never render it
                st.warning(f"Step {n} hidden: it has no citation.")
                continue
            _md(step_html(n, step))
            feedback("step", step.citations[0], f"step-{idx}-{n}", idx)
        if steps:
            feedback("recommendation", source, f"rec-{idx}", idx)


def approval_badge(resp: AnalysisResponse) -> None:
    if resp.human_approval_required:
        reasons = "; ".join(resp.details.approval_reasons) or "see rules"
        _md(f"{plain_chip('HUMAN APPROVAL REQUIRED', '#e65100')} <small>{_esc(reasons)}</small>")
    else:
        _md(plain_chip("L1 MAY APPLY (no approval needed)", "#2e7d32"))


def _bundle_downloads(client: ApiClient, bundle_id: str, key: str) -> None:
    cache = st.session_state.setdefault("bundles", {})
    if bundle_id not in cache:
        try:
            cache[bundle_id] = (
                json.dumps(client.escalation_bundle(bundle_id, "json"), indent=2),
                client.escalation_bundle(bundle_id, "md"),
            )
        except ApiError as err:
            st.caption(f"Bundle not available: {err.message}")
            return
    js, md = cache[bundle_id]
    c1, c2 = st.columns(2)
    c1.download_button(
        "⬇ Bundle (JSON)", js, f"{bundle_id}.json", "application/json", key=f"bj-{key}"
    )
    c2.download_button(
        "⬇ Bundle (Markdown)", md, f"{bundle_id}.md", "text/markdown", key=f"bm-{key}"
    )


def decision_banner(client: ApiClient, resp: AnalysisResponse) -> AnalysisResponse | None:
    """Green recommend / amber clarify (answer box) / red escalate. Returns a re-analysis."""
    d = resp.details
    label = chip("decision", d.decision)
    if d.decision == "recommend":
        _md(f"{label}")
        st.success("Recommend: apply the cited steps below (after any required approval).")
    elif d.decision == "clarify":
        _md(f"{label}")
        st.warning("Clarify: the assistant needs one answer before it can recommend.")
        for q in d.clarifying_questions:
            st.markdown(f"- {q}")
        with st.form(f"clarify-{d.analysis_id}"):
            answer = st.text_area("Your answer (one round only)")
            if st.form_submit_button("Send answer and re-analyse"):
                if not answer.strip():
                    st.error("Please type an answer.")
                    return None
                try:
                    return client.clarify(resp.ticket_id, d.analysis_id, answer.strip())
                except ApiError as err:
                    st.error(f"Clarify failed: {err.message}")
    else:
        _md(f"{label}")
        escs, seen = [d.escalation] + [s.escalation for s in d.secondary_issues], set()
        for esc in [e for e in escs if e]:
            ident = esc.bundle_id or f"{esc.issue_idx}-{esc.assignment_group}"
            if ident in seen:  # the ticket-level escalation may repeat a secondary issue's
                continue
            seen.add(ident)
            reasons = "; ".join(f"{r.rule}: {r.text}" for r in esc.reasons)
            issue = "primary issue" if esc.issue_idx == 0 else f"issue {esc.issue_idx + 1}"
            st.error(
                f"Escalate ({issue}) to **{esc.assignment_group}** · tier **{esc.tier}**\n\n"
                f"Reasons: {reasons}"
            )
            if esc.bundle_id:
                _bundle_downloads(client, esc.bundle_id, f"{d.analysis_id}-{esc.issue_idx}")
    return None


def json_toggle(client: ApiClient, resp: AnalysisResponse) -> None:
    if st.toggle("View JSON", key=f"json-{resp.details.analysis_id}"):
        contract = resp.to_contract().model_dump(mode="json")
        t1, t2 = st.tabs(["Contract (8 keys)", "Full view (with details)"])
        t1.json(contract)
        t2.json(resp.model_dump(mode="json"))


def render_analysis(
    client: ApiClient, resp: AnalysisResponse, *, feedback: FeedbackHook = _no_feedback
) -> AnalysisResponse | None:
    """Render one full-view response. Returns a new response if a clarify answer was sent."""
    d = resp.details
    st.subheader(f"Result for {resp.ticket_id}")
    if d.degraded:
        st.info(f"Degraded mode: {', '.join(d.degraded)}. Results come from fallbacks.")
    if d.redactions.count:
        st.caption(f"🔒 {d.redactions.count} item(s) redacted: {', '.join(d.redactions.types)}")
    new = decision_banner(client, resp)
    approval_badge(resp)

    names = ["Issue 1 (primary)"] + [
        f"Issue {i + 2}: {s.classification.subcategory}" for i, s in enumerate(d.secondary_issues)
    ]
    tabs = st.tabs(names)
    with tabs[0]:
        c1, c2 = st.columns(2)
        with c1:
            classification_card(resp.classification, d.classification, feedback=feedback)
        with c2:
            priority_card(resp.priority, d.priority, feedback=feedback)
        kedb_card(resp.kedb_match, d.kedb, feedback=feedback)
        steps_card(d.steps, resp.recommended_solution.source, feedback=feedback)
        similar_card(resp, feedback=feedback)
    for i, sec in enumerate(d.secondary_issues, 1):
        with tabs[i]:
            _md(chip("decision", sec.decision, f"ISSUE {i + 1}: "))
            c1, c2 = st.columns(2)
            with c1:
                classification_card(
                    sec.classification, sec.classification_details, feedback=feedback, idx=i
                )
            with c2:
                priority_card(sec.priority, sec.priority_details, feedback=feedback, idx=i)
            kedb_card(sec.kedb_match, sec.kedb, feedback=feedback, idx=i)
            steps_card(sec.steps, sec.recommended_solution.source, feedback=feedback, idx=i)
    json_toggle(client, resp)
    return new
