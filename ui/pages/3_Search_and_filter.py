"""Search & filter (FR-07): query + metadata filters, paginated table, detail drawer."""

from __future__ import annotations

from datetime import date
from typing import Any

import streamlit as st

from shared.constants import ASSIGNMENT_GROUPS, PRIORITY_LEVELS, TAXONOMY
from ui.client import ApiError
from ui.components.badges import chip, chip_spec, origin_is_synthetic, priority_code_to_level
from ui.settings import load_ui_config
from ui.state import header

client = header("🗂️ Search & filter tickets")
cfg = load_ui_config()["search"]
ANY = "(any)"


def _opt(value: str) -> str | None:
    return None if value in (ANY, "") else value


with st.form("filters"):
    q = st.text_input("Search text", key="s_q", placeholder="e.g. vpn gateway, ORA-01013")
    c1, c2, c3, c4 = st.columns(4)
    cat = c1.selectbox("Category", [ANY, *TAXONOMY], key="s_category")
    subs = sorted({s for v in TAXONOMY.values() for s in v})
    sub = c2.selectbox("Subcategory", [ANY, *subs], key="s_subcategory")
    prio = c3.selectbox("Priority", [ANY, *PRIORITY_LEVELS], key="s_priority")
    status = c4.selectbox("Status", [ANY, "new", "open", "closed", "resolved"], key="s_status")
    c5, c6, c7, c8 = st.columns(4)
    group = c5.selectbox("Assignment group", [ANY, *ASSIGNMENT_GROUPS], key="s_group")
    app = c6.text_input("Application", key="s_application")
    env = c7.text_input("Environment", key="s_environment")
    source = c8.text_input("Source", key="s_source")
    c9, c10, c11 = st.columns([2, 2, 1])
    use_dates = c9.checkbox("Filter by opened date", key="s_use_dates")
    d_from = c9.date_input("From", value=date(2026, 1, 1), key="s_from")
    d_to = c10.date_input("To", value=date.today(), key="s_to")
    size = c11.selectbox("Rows / page", cfg["page_sizes"], index=1, key="s_size")
    if st.form_submit_button("Search", type="primary"):
        st.session_state["s_page"] = 1
        st.session_state["s_params"] = {
            "q": q.strip(),
            "category": _opt(cat),
            "subcategory": _opt(sub),
            "priority": _opt(prio),
            "status": _opt(status),
            "assignment_group": _opt(group),
            "application": app.strip() or None,
            "environment": env.strip() or None,
            "source": source.strip() or None,
            "date_from": d_from.isoformat() if use_dates else None,
            "date_to": d_to.isoformat() if use_dates else None,
            "size": int(size),
        }

params: dict[str, Any] = st.session_state.get("s_params", {"size": cfg["page_size"]})
page_no = int(st.session_state.get("s_page", 1))
try:
    result = client.search_tickets(page=page_no, **params)
except ApiError as err:
    st.error(f"Search failed: {err.message}")
    st.stop()

items, total, size = result["items"], int(result["total"]), int(params.get("size", 20))
pages = max(1, -(-total // size))
st.caption(
    f"{total} ticket(s) · page {page_no} of {pages}. "
    "Rows you are not allowed to see (restricted) are hidden by the API."
)


def _row(i: dict[str, Any]) -> dict[str, Any]:
    meta = i.get("metadata") or {}
    derived = meta.get("derived_fields") or []
    return {
        "Ticket": i["ticket_id"],
        "Title": i.get("title"),
        "Category": i.get("category"),
        "Subcategory": i.get("subcategory"),
        "Priority": priority_code_to_level(i.get("priority")),
        "Status": i.get("status"),
        "Group": meta.get("assignment_group"),
        "Application": meta.get("application"),
        "Environment": meta.get("environment"),
        "Opened": (meta.get("opened_at") or "")[:10],
        "Resolution origin": chip_spec("origin", i.get("resolution_origin"))[0],
        "Synthetic fields": ", ".join(derived) if derived else "",
        "Score": i.get("score"),
    }


if items:
    st.dataframe([_row(i) for i in items], hide_index=True)
    st.caption(
        "SYNTHETIC: resolutions, dates, groups, applications and environments of historical "
        "tickets are generated and flagged per row (README §15)."
    )
else:
    st.info("No tickets match these filters.")

c_prev, c_mid, c_next = st.columns([1, 4, 1])
if c_prev.button("◀ Previous", disabled=page_no <= 1):
    st.session_state["s_page"] = page_no - 1
    st.rerun()
if c_next.button("Next ▶", disabled=page_no >= pages):
    st.session_state["s_page"] = page_no + 1
    st.rerun()

# ---------------------------------------------------------------------------- detail drawer
if items:
    ids = [i["ticket_id"] for i in items]
    chosen = st.selectbox("Open ticket details", ["", *ids], key="s_open")
    if chosen:
        try:
            rec = client.get_ticket(chosen)
        except ApiError as err:
            st.error(f"Cannot open {chosen}: {err.message}")
        else:
            with st.expander(f"🎫 {chosen}: {rec.get('title')}", expanded=True):
                origin = rec.get("resolution_origin", "none")
                level = priority_code_to_level(rec.get("priority"))
                st.markdown(
                    f"{chip('priority', level)} {chip('origin', origin)} "
                    f"collection: `{rec.get('collection')}`",
                    unsafe_allow_html=True,
                )
                st.write(rec.get("description"))
                if rec.get("resolution"):
                    label = (
                        "Resolution (synthetic)" if origin_is_synthetic(origin) else "Resolution"
                    )
                    st.markdown(f"**{label}:** {rec['resolution']}")
                if rec.get("verified_by"):
                    st.caption(f"Verified by {rec['verified_by']}")
                if rec.get("derived_fields"):
                    st.caption("Synthetic (derived) fields: " + ", ".join(rec["derived_fields"]))
                st.json({k: v for k, v in rec.items() if k not in ("description", "metadata")})
