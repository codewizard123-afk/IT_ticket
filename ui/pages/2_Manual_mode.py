"""Manual mode (FR-09 baseline): the trial ticket text, keyword search only, NO AI output.

This page must never call ``/analyze`` or show any classification, priority, similarity score,
KEDB match, recommendation or decision produced by the assistant (tests enforce it). It only
uses ``/trial/tickets``, ``/tickets/search``, ``/kb/search``, ``/kb/{id}``, ``/events`` and
``/decisions``. Timer events: ``manual.started`` when the ticket is revealed, ``manual.decision``
when the decision is submitted.
"""

from __future__ import annotations

import uuid

import streamlit as st

from ui.client import ApiError
from ui.components.feedback import decision_form
from ui.state import header

client = header("⏱️ Manual mode (trial baseline)")
st.caption(
    "Work the ticket as you normally would: keyword search over past tickets and the KB only. "
    "No AI suggestions are shown on this page. The timer starts when you reveal the ticket."
)
if client.role == "kb_owner":
    st.warning("The manual-mode trial is for l1, l2 and admin users (architecture §8.6).")
    st.stop()

try:
    trial = client.trial_tickets()
except ApiError as err:
    st.error(f"Trial tickets unavailable: {err.message}")
    st.stop()
by_id = {t["id"]: t for t in trial}

ticket_id = st.selectbox(
    "Trial ticket",
    ["", *by_id],
    format_func=lambda t: t or "— choose the next ticket on your schedule —",
    key="m_ticket",
)
sessions: dict[str, str] = st.session_state.setdefault("manual_sessions", {})

start_clicked = (
    bool(ticket_id)
    and ticket_id not in sessions
    and st.button("▶ Start ticket (starts the timer)", type="primary")
)
if start_clicked:
    session_id = f"manual-{ticket_id}-{uuid.uuid4().hex[:8]}"
    try:
        client.event(ticket_id, "manual.started", session_id, mode="manual")
        sessions[ticket_id] = session_id
        st.rerun()
    except ApiError as err:
        st.error(f"Could not start: {err.message}")

if ticket_id and ticket_id in sessions:
    session_id = sessions[ticket_id]
    ticket = by_id[ticket_id]
    with st.container(border=True):
        st.markdown(f"**{ticket_id} · {ticket['title']}**")
        st.write(ticket["description"])

    tab_t, tab_kb = st.tabs(["Search past tickets", "Search the KB"])
    with tab_t:
        q = st.text_input("Keywords", key=f"m_tq_{ticket_id}")
        if q.strip():
            try:
                items = client.search_tickets(q=q.strip(), size=10)["items"]
            except ApiError as err:
                st.error(f"Search failed: {err.message}")
                items = []
            # plain records only: no scores, tiers or match explanations
            for i in items:
                with st.expander(f"{i['ticket_id']} · {i.get('title')}"):
                    st.write(i.get("excerpt"))
                    st.caption(f"Labels: {i.get('category')} / {i.get('subcategory')}")
                    if i.get("resolution"):
                        origin = i.get("resolution_origin", "none")
                        tag = " (SYNTHETIC)" if str(origin).startswith("synthetic") else ""
                        st.caption(f"Recorded resolution{tag}: {i['resolution']}")
    with tab_kb:
        kq = st.text_input("Keywords", key=f"m_kq_{ticket_id}")
        if kq.strip():
            try:
                hits = client.search_kb(kq.strip())["items"][:10]
            except ApiError as err:
                st.error(f"KB search failed: {err.message}")
                hits = []
            for h in hits:
                with st.expander(f"{h['kb_id']} · {h['title']}"):
                    art = client.get_kb(h["kb_id"])
                    if art.diagnostic_steps:
                        st.markdown(
                            "**Diagnostics**\n"
                            + "\n".join(f"{n}. {s}" for n, s in enumerate(art.diagnostic_steps, 1))
                        )
                    st.markdown(
                        "**Resolution**\n"
                        + "\n".join(f"{n}. {s}" for n, s in enumerate(art.resolution.steps, 1))
                    )
                    if art.escalation_criteria:
                        st.markdown(
                            "**Escalate when**\n"
                            + "\n".join(f"- {c}" for c in art.escalation_criteria)
                        )

    decision_form(
        client,
        session_id,
        mode="manual",
        key="manual",
        ask_confidence=True,
        ticket_id=ticket_id,
    )
