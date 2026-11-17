"""Feedback buttons (👍 ✗ ?) for every item and the final decision form (FR-06, FR-09)."""

from __future__ import annotations

from typing import Any

import streamlit as st

from shared.constants import FINAL_DECISIONS, PRIORITY_LEVELS, TAXONOMY
from ui.client import ApiClient, ApiError
from ui.components.analysis import FeedbackHook

VERDICTS = (
    ("useful", "👍", "Useful"),
    ("incorrect", "✗", "Incorrect"),
    ("unresolved", "?", "Unsure"),
)
CONFIDENCE_PREFIX = "Self-reported confidence: "
DECISION_LABELS = {
    "resolved": "Resolved",
    "escalated": "Escalated",
    "needs_more_info": "Needs more info",
}


def make_feedback_hook(client: ApiClient, analysis_id: str) -> FeedbackHook:
    """A hook for ``render_analysis``: three buttons per item, posted to ``/feedback``."""
    given: dict[str, str] = st.session_state.setdefault("feedback_given", {})

    def hook(target_type: str, target_ref: str | None, key: str, issue_idx: int) -> None:
        state_key = f"{analysis_id}:{key}"
        if state_key in given:
            st.caption(f"Feedback recorded: {given[state_key]}")
            return
        cols = st.columns([1, 1, 1, 6])
        for col, (verdict, icon, label) in zip(cols, VERDICTS, strict=False):
            if col.button(icon, key=f"fb-{state_key}-{verdict}", help=f"{label} ({target_type})"):
                try:
                    client.feedback(
                        {
                            "analysis_id": analysis_id,
                            "issue_idx": issue_idx,
                            "target_type": target_type,
                            "target_ref": target_ref,
                            "verdict": verdict,
                        }
                    )
                    given[state_key] = label
                    st.rerun()
                except ApiError as err:
                    st.error(f"Feedback failed: {err.message}")

    return hook


def classification_correction(client: ApiClient, analysis_id: str, current: tuple[str, str]):
    """Optional correction of the category/subcategory, sent as ``incorrect`` feedback."""
    with st.expander("Correct the classification"), st.form(f"correct-{analysis_id}"):
        cats = list(TAXONOMY)
        cat = st.selectbox(
            "Category", cats, index=cats.index(current[0]) if current[0] in cats else 0
        )
        subs = [*TAXONOMY[cat], "Other"]
        sub = st.selectbox("Subcategory", subs)
        comment = st.text_input("Comment (optional)")
        if st.form_submit_button("Send correction"):
            try:
                client.feedback(
                    {
                        "analysis_id": analysis_id,
                        "target_type": "classification",
                        "target_ref": f"{current[0]}/{current[1]}",
                        "verdict": "incorrect",
                        "corrected_value": f"{cat}/{sub}",
                        "comment": comment or None,
                    }
                )
                st.success(f"Correction sent: {cat}/{sub}")
            except ApiError as err:
                st.error(f"Correction failed: {err.message}")


def decision_form(
    client: ApiClient,
    analysis_id: str,
    *,
    mode: str = "assisted",
    defaults: dict[str, Any] | None = None,
    key: str = "decision",
    ask_confidence: bool = False,
    ticket_id: str | None = None,
) -> dict[str, Any] | None:
    """The final decision form (same in assisted and manual mode). Returns the posted decision.

    ``ask_confidence`` adds the trial's self-reported confidence (1-5), stored as the last line of
    the notes (the frozen ``Decision`` schema has no field for it). In manual mode the
    ``manual.decision`` timer event is posted for ``ticket_id`` right after the decision.
    """
    defaults = defaults or {}
    done: dict[str, dict[str, Any]] = st.session_state.setdefault("decisions_done", {})
    if analysis_id in done:
        d = done[analysis_id]
        st.success(f"Decision recorded: **{DECISION_LABELS[d['final_decision']]}**")
        return None
    cats = list(TAXONOMY)
    with st.form(f"{key}-{analysis_id}"):
        st.markdown("**Final decision**")
        final = st.radio(
            "Decision",
            FINAL_DECISIONS,
            format_func=DECISION_LABELS.get,
            horizontal=True,
            index=FINAL_DECISIONS.index(defaults.get("final_decision", "resolved")),
        )
        c1, c2, c3 = st.columns(3)
        cat = c1.selectbox(
            "Category",
            cats,
            index=cats.index(defaults["category"]) if defaults.get("category") in cats else 0,
        )
        all_subs = sorted({s for subs in TAXONOMY.values() for s in subs} | {"Other"})
        sub = c2.selectbox(
            "Subcategory",
            all_subs,
            index=all_subs.index(defaults["subcategory"])
            if defaults.get("subcategory") in all_subs
            else 0,
        )
        levels = list(PRIORITY_LEVELS)
        prio = c3.selectbox(
            "Priority",
            levels,
            index=levels.index(defaults["priority"]) if defaults.get("priority") in levels else 2,
        )
        used = st.text_input(
            "KB article or incident used (e.g. KB-0018, INC-H0007)",
            value=defaults.get("kb_or_incident_used") or "",
        )
        notes = st.text_area("Resolution notes (one step per line)")
        confidence = (
            st.slider("How confident are you in this decision? (1-5)", 1, 5, 3)
            if ask_confidence
            else None
        )
        if not st.form_submit_button("Submit decision", type="primary"):
            return None
    decision = {
        "analysis_id": analysis_id,
        "final_decision": final,
        "final_category": cat,
        "final_subcategory": sub,
        "final_priority": prio,
        "kb_or_incident_used": used.strip() or None,
        "resolution_notes": notes.strip() or None,
        "mode": mode,
    }
    if confidence is not None:
        line = f"{CONFIDENCE_PREFIX}{confidence}/5"
        decision["resolution_notes"] = "\n".join(filter(None, [decision["resolution_notes"], line]))
    try:
        client.decide(decision)
        if mode == "manual" and ticket_id:
            client.event(ticket_id, "manual.decision", analysis_id, mode="manual")
    except ApiError as err:
        st.error(f"Decision failed: {err.message}")
        return None
    done[analysis_id] = decision
    st.success(f"Decision recorded: **{DECISION_LABELS[final]}**")
    return decision


def propose_kb_draft(client: ApiClient, analysis_id: str) -> None:
    """Shown after an unresolved/incorrect outcome with notes (§8.8); l2+ only."""
    done = st.session_state.get("decisions_done", {}).get(analysis_id)
    negative = {
        v
        for k, v in st.session_state.get("feedback_given", {}).items()
        if k.startswith(f"{analysis_id}:")
    } & {"Incorrect", "Unsure"}
    if not done or not done.get("resolution_notes"):
        return
    if not negative and done["final_decision"] != "resolved":
        return
    if client.role not in ("l2", "kb_owner", "admin"):
        st.caption("An L2 analyst can turn these notes into a KB draft.")
        return
    if st.button("📝 Propose KB draft from my notes", key=f"draft-{analysis_id}"):
        try:
            draft = client.create_kb_draft(analysis_id)
            st.success(f"Draft {draft['draft_id']} created ({draft['article']['kb_id']}).")
        except ApiError as err:
            st.error(f"Draft failed: {err.message}")
