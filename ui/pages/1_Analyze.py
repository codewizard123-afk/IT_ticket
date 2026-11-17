"""Analyze (assisted mode): ticket form → evidence-backed result (architecture §9.1, FR-01…06)."""

from __future__ import annotations

import streamlit as st

from ui.client import ApiError
from ui.components.analysis import render_analysis
from ui.components.feedback import (
    classification_correction,
    decision_form,
    make_feedback_hook,
    propose_kb_draft,
)
from ui.state import header, sample_tickets

client = header("🔎 Analyze a ticket")
samples = sample_tickets()
FIELDS = ("title", "description", "application", "environment", "location")


def _load_sample() -> None:
    sid = st.session_state.get("sample_id")
    ticket = samples.get(sid) if sid else None
    for f in FIELDS:
        st.session_state[f"f_{f}"] = (ticket or {}).get(f) or ""
    st.session_state["f_id"] = sid or ""


st.selectbox(
    "Load sample",
    [""] + list(samples),
    key="sample_id",
    on_change=_load_sample,
    format_func=lambda s: "— type a new ticket —" if not s else f"{s}: {samples[s]['title']}",
)

with st.form("ticket"):
    st.text_input("Title", key="f_title")
    st.text_area("Description", key="f_description", height=140)
    c1, c2, c3 = st.columns(3)
    c1.text_input("Application (optional)", key="f_application")
    c2.text_input("Environment (optional)", key="f_environment")
    c3.text_input("Location (optional)", key="f_location")
    submitted = st.form_submit_button("Analyze", type="primary")

if submitted:
    title = st.session_state.get("f_title", "").strip()
    description = st.session_state.get("f_description", "").strip()
    if not title or not description:
        st.error("Title and description are required.")
    else:
        ticket = {
            "title": title,
            "description": description,
            "status": "new",
            "metadata": {"source": "ui"},
        }
        if st.session_state.get("f_id"):
            ticket["id"] = st.session_state["f_id"]
            ticket["metadata"] = samples[ticket["id"]].get("metadata") or {"source": "demo"}
        for f in ("application", "environment", "location"):
            if st.session_state.get(f"f_{f}", "").strip():
                ticket[f] = st.session_state[f"f_{f}"].strip()
        try:
            with st.spinner("Analysing…"):
                st.session_state["analysis"] = client.analyze(ticket)
        except ApiError as err:
            st.error(f"Analysis failed: {err.message}")

resp = st.session_state.get("analysis")
if resp is not None:
    aid = resp.details.analysis_id
    new = render_analysis(client, resp, feedback=make_feedback_hook(client, aid))
    classification_correction(
        client, aid, (resp.classification.category, resp.classification.subcategory)
    )
    kedb = resp.kedb_match.error_code if resp.kedb_match.found else None
    decision_form(
        client,
        aid,
        mode="assisted",
        defaults={
            "final_decision": {"escalate": "escalated", "clarify": "needs_more_info"}.get(
                resp.details.decision, "resolved"
            ),
            "category": resp.classification.category,
            "subcategory": resp.classification.subcategory,
            "priority": resp.priority.level,
            "kb_or_incident_used": kedb,
        },
    )
    propose_kb_draft(client, aid)
    if st.session_state.get("viewed_analysis") != aid:
        try:
            client.event(resp.ticket_id, "ui.viewed", aid, mode="assisted")
            st.session_state["viewed_analysis"] = aid
        except ApiError as err:
            st.caption(f"Could not record ui.viewed: {err.message}")
    if new is not None:
        st.session_state["analysis"] = new
        st.rerun()
