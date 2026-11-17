"""Streamlit entry point and Login / role page (architecture §9.1). Run: ``make ui``."""

from __future__ import annotations

import streamlit as st

from shared.constants import ROLES
from ui.client import ApiError
from ui.settings import get_settings, load_ui_config
from ui.state import get_client

st.set_page_config(page_title="IT Ticket Assistant", page_icon="🎫", layout="wide")

settings = get_settings()
client = get_client()

st.title("🎫 IT Ticket Analysis & Resolution Assistant")
st.caption(
    "The assistant recommends, asks for details or escalates. An analyst always reviews; "
    "the system never acts on its own."
)

try:
    health = client.health()
    llm = "available" if health.get("llm_available") else "offline fallback"
    stale = " · ⚠️ index stale, run `make data`" if health.get("index_stale") else ""
    mock = " · 🧪 MOCK API (golden fixtures)" if health.get("mock") else ""
    st.success(f"API status: {health.get('status')} · LLM: {llm}{stale}{mock}")
except ApiError as err:
    st.error(f"API not reachable ({err.message}). Start `make api`, or run the UI with MOCK_API=1.")

if client.role:
    who = client.username or client.role
    group = f" ({client.assignment_group})" if client.assignment_group else ""
    st.info(f"Signed in as **{who}** · role **{client.role}**{group}. Use the pages on the left.")

if settings.demo_auth:
    st.subheader("Choose a role")
    st.markdown("🔓 **DEMO AUTH**: the role is sent as a header; no password needed.")
    defaults = load_ui_config()["mock_api"]["default_group"]
    with st.form("demo_role"):
        role = st.selectbox("Role", ROLES, index=ROLES.index(client.role or "l1"))
        group = st.text_input(
            "Assignment group (optional)", value=client.assignment_group or defaults.get(role) or ""
        )
        if st.form_submit_button("Use this role"):
            client.use_demo_role(role, group.strip() or None)
            st.rerun()

with st.expander("Log in with a user account", expanded=not settings.demo_auth):
    with st.form("login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Log in"):
            try:
                client.login(username, password)
                st.rerun()
            except ApiError as err:
                st.error(f"Login failed: {err.message}")

st.markdown(
    """
**Pages**
1. **Analyze**: assisted analysis with evidence, feedback and the final decision
2. **Manual mode**: MTTR baseline for the timed trial (no AI output)
3. **Search & filter**: historical and live tickets (FR-07)
4. **Review queue**: clarify / escalate / needs-approval items
5. **KB browser**: articles, coverage gaps and drafts
6. **Audit view**: timeline, evidence, hash-chain status and export
7. **MTTR dashboard**: TTR / TTD / MTTR and trial results
"""
)
