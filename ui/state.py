"""Session helpers shared by every page: the API client, the signed-in role and sample tickets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st

from ui.client import ApiClient, http_transport, mock_transport
from ui.mock_backend import MockBackend
from ui.settings import ROOT, get_settings

MOCK_DATA = Path(__file__).parent / "mock_data"


@st.cache_resource
def _shared_mock_backend() -> MockBackend:
    """One mock backend per server process, so pages and users share state like a real API."""
    return MockBackend()


def get_client() -> ApiClient:
    """The session's client (tests inject one via ``session_state["client"]``)."""
    if "client" not in st.session_state:
        s = get_settings()
        if s.mock_api:
            transport = mock_transport(_shared_mock_backend())
        else:
            transport = http_transport(s.api_base_url, s.timeout_s)
        st.session_state["client"] = ApiClient(transport, settings=s, mock=s.mock_api)
    return st.session_state["client"]


def require_login() -> ApiClient:
    """Stop the page with a hint unless a role has been chosen on the Login page."""
    client = get_client()
    if not client.role:
        st.warning("Choose a role or log in on the **Login** page first.")
        st.stop()
    return client


def header(title: str) -> ApiClient:
    """Page title plus the role / mode badges every page shows."""
    client = require_login()
    st.title(title)
    bits = [f"Role: **{client.role}**"]
    if client.assignment_group:
        bits.append(f"Group: **{client.assignment_group}**")
    if not client.token:
        bits.append("🔓 DEMO AUTH")
    if client.mock:
        bits.append("🧪 MOCK API (golden fixtures)")
    st.caption(" · ".join(bits))
    return client


def _strip_gold(ticket: dict[str, Any]) -> dict[str, Any]:
    meta = {k: v for k, v in (ticket.get("metadata") or {}).items() if not k.startswith("expected")}
    return {**ticket, "metadata": meta}


def sample_tickets() -> dict[str, dict[str, Any]]:
    """DEMO-001…007 (real sample file when P1 has added it, else the UI placeholders) and the
    test tickets if present. Gold labels are stripped before they ever reach the form."""
    real = ROOT / "data" / "samples" / "sample_tickets.json"
    demos_path = real if real.exists() else MOCK_DATA / "demo_tickets.json"
    tickets = json.loads(demos_path.read_text(encoding="utf-8"))
    test_path = ROOT / "data" / "test" / "test_tickets.json"
    if test_path.exists():
        seen = {t["id"] for t in tickets}
        tickets += [
            t for t in json.loads(test_path.read_text(encoding="utf-8")) if t["id"] not in seen
        ]
    return {t["id"]: _strip_gold(t) for t in tickets}
