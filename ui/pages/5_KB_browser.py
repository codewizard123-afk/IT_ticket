"""KB browser (FR-03/06): articles and detail, coverage gaps, drafts and approval (kb_owner)."""

from __future__ import annotations

import streamlit as st

from ui.client import ApiError
from ui.components.badges import plain_chip
from ui.settings import load_ui_config
from ui.state import header

client = header("📚 KB browser")
tab_articles, tab_gaps, tab_drafts = st.tabs(["Articles", "Coverage gaps", "Drafts"])

# ---------------------------------------------------------------------------- articles
with tab_articles:
    c1, c2 = st.columns([3, 1])
    q = c1.text_input("Search the KEDB", key="kb_q", placeholder="symptom, error code, product")
    category = c2.text_input("KB category (optional)", key="kb_cat")
    try:
        hits = client.search_kb(q.strip(), category=category.strip() or None)["items"]
    except ApiError as err:
        st.error(f"KB search failed: {err.message}")
        hits = []
    st.caption(f"{len(hits)} article(s)")
    if hits:
        st.dataframe(
            [
                {
                    "KB": h["kb_id"],
                    "Title": h["title"],
                    "Category": h.get("category"),
                    "Subcategory": h.get("subcategory"),
                    "Score": h.get("score"),
                    "Matched": h.get("matched_field"),
                    "Admin needed": "yes" if h.get("requires_admin") else "",
                }
                for h in hits
            ],
            hide_index=True,
        )
        titles = {h["kb_id"]: h["title"] for h in hits}
        kb_id = st.selectbox(
            "Open article",
            ["", *titles],
            format_func=lambda k: f"{k}: {titles[k]}" if k else "— choose —",
            key="kb_open",
        )
        if kb_id:
            art = client.get_kb(kb_id)
            st.subheader(f"{art.kb_id} · {art.title}")
            st.markdown(
                plain_chip(f"STATUS: {art.status.upper()}", "#2e7d32")
                + " "
                + plain_chip(f"v{art.version} · {art.source}", "#455a64")
                + (
                    " " + plain_chip("REQUIRES ADMIN", "#6a1b9a")
                    if art.resolution.requires_admin
                    else ""
                ),
                unsafe_allow_html=True,
            )
            if art.root_cause:
                st.markdown(f"**Root cause:** {art.root_cause}")
            if art.symptoms:
                st.markdown("**Symptoms**\n" + "\n".join(f"- {s}" for s in art.symptoms))
            if art.common_error_codes:
                st.markdown(
                    "**Error codes:** " + ", ".join(f"`{c}`" for c in art.common_error_codes)
                )
            if art.diagnostic_steps:
                st.markdown(
                    "**Diagnostics**\n"
                    + "\n".join(
                        f"{n}. {s} `[{art.kb_id}·diag.{n}]`"
                        for n, s in enumerate(art.diagnostic_steps, 1)
                    )
                )
            st.markdown(
                "**Resolution**\n"
                + "\n".join(
                    f"{n}. {s} `[{art.kb_id}·res.{n}]`"
                    for n, s in enumerate(art.resolution.steps, 1)
                )
            )
            if art.escalation_criteria:
                st.markdown(
                    "**Escalate when**\n" + "\n".join(f"- {c}" for c in art.escalation_criteria)
                )

# ---------------------------------------------------------------------------- coverage gaps
with tab_gaps:
    st.markdown("**Known KEDB gaps** (no article exists; these tickets escalate by design)")
    st.write(", ".join(load_ui_config()["known_kb_gaps"]))
    try:
        observed = client.metrics_summary().get("kb_gaps", {})
    except ApiError as err:
        st.error(f"Metrics unavailable: {err.message}")
        observed = {}
    st.markdown("**Observed gaps**: analyses that found no KEDB match, by subcategory")
    if observed:
        st.dataframe(
            [
                {"Subcategory": k, "Analyses without a KB match": v}
                for k, v in sorted(observed.items(), key=lambda kv: -kv[1])
            ],
            hide_index=True,
        )
    else:
        st.caption("None recorded yet.")

# ---------------------------------------------------------------------------- drafts
with tab_drafts:
    can_approve = client.role in ("kb_owner", "admin")
    try:
        drafts = client.kb_drafts()
    except ApiError as err:
        st.error(f"Drafts unavailable: {err.message}")
        drafts = []
    if not drafts:
        st.info("No drafts. Analysts propose drafts from the Analyze page after a decision.")
    for d in drafts:
        art = d["article"]
        with st.container(border=True):
            st.markdown(
                f"**{d['draft_id']}** → {art['kb_id']} · {art['title']} "
                + plain_chip(
                    d["status"].upper(), "#b26a00" if d["status"] == "draft" else "#2e7d32"
                ),
                unsafe_allow_html=True,
            )
            st.caption(f"Proposed by {d['proposed_by']} from analysis {d['source_analysis_id']}")
            st.markdown("\n".join(f"{n}. {s}" for n, s in enumerate(art["resolution"]["steps"], 1)))
            if d["status"] == "draft" and can_approve:
                with st.form(f"approve-{d['draft_id']}"):
                    title = st.text_input("Title", value=art["title"])
                    if st.form_submit_button("Approve and publish"):
                        try:
                            new = client.approve_kb_draft(d["draft_id"], {"title": title})
                            st.success(f"{new.kb_id} approved and searchable.")
                            st.rerun()
                        except ApiError as err:
                            st.error(f"Approve failed: {err.message}")
            elif d["status"] == "draft":
                st.caption("Only a KB owner can approve drafts.")
