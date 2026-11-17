"""Audit view (FR-08): per-analysis timeline, evidence, decisions, hash-chain status, export."""

from __future__ import annotations

import json
from datetime import date

import streamlit as st

from ui.client import ApiError
from ui.components.badges import chip, plain_chip
from ui.examples import worked_example_md
from ui.state import header

client = header("🧾 Audit view")

# ---------------------------------------------------------------------------- pick an analysis
known: dict[str, str] = {}
current = st.session_state.get("analysis")
if current is not None:
    known[current.details.analysis_id] = f"{current.ticket_id} (this session)"
try:
    for item in client.review_queue(status="all")["items"]:
        known.setdefault(item["analysis_id"], f"{item['ticket_id']} · {item['title']}")
except ApiError:
    pass

c1, c2 = st.columns([2, 2])
picked = c1.selectbox(
    "Recent analyses",
    ["", *known],
    format_func=lambda a: f"{a} · {known[a]}" if a else "— choose —",
    key="audit_pick",
)
typed = c2.text_input("…or analysis ID", key="audit_id")
analysis_id = typed.strip() or picked

if analysis_id:
    try:
        rec = client.audit(analysis_id)
    except ApiError as err:
        st.error(f"Cannot open audit record {analysis_id}: {err.message}")
        rec = None
    if rec:
        resp = rec["response"]
        chain = rec.get("chain", {})
        status = (
            plain_chip("HASH CHAIN OK", "#2e7d32")
            if chain.get("ok")
            else plain_chip(f"HASH CHAIN BROKEN at row {chain.get('broken_at')}", "#c62828")
        )
        st.markdown(
            f"{status} {chip('decision', resp['details']['decision'])} "
            f"**{rec['ticket_id']}** · analysed by {rec['user']} ({rec['role']}) at "
            f"{rec['created_at']} · filtered for role `{rec['filtered_for_role']}`",
            unsafe_allow_html=True,
        )
        t_time, t_ev, t_dec, t_raw = st.tabs(
            ["Timeline", "Evidence", "Decisions & feedback", "Record"]
        )
        with t_time:
            if rec["timeline"]:
                st.dataframe(
                    [
                        {
                            "Time (UTC)": e["ts_utc"],
                            "Event": e["event"],
                            "Mode": e["mode"],
                            "Actor": e["actor"],
                        }
                        for e in rec["timeline"]
                    ],
                    hide_index=True,
                )
            else:
                st.caption("No events.")
        with t_ev:
            st.markdown(f"**Ticket (redacted):** {rec['ticket']['title']}")
            st.caption(rec["ticket"]["description"])
            if rec["evidence"]:
                st.dataframe(rec["evidence"], hide_index=True)
            st.markdown("**Steps and citations**")
            for n, s in enumerate(resp["details"].get("steps", []), 1):
                st.markdown(f"{n}. {s['text']} `{', '.join(s['citations'])}` ({s['role']})")
        with t_dec:
            if rec.get("decision"):
                st.json(rec["decision"])
            else:
                st.caption("No final decision yet.")
            st.markdown(f"**Feedback ({len(rec['feedback'])})**")
            if rec["feedback"]:
                st.dataframe(
                    [
                        {
                            "Target": f["target_type"],
                            "Ref": f.get("target_ref"),
                            "Verdict": f["verdict"],
                            "Correction": f.get("corrected_value"),
                            "User": f.get("user"),
                            "At": f.get("created_at"),
                        }
                        for f in rec["feedback"]
                    ],
                    hide_index=True,
                )
        with t_raw:
            st.json(rec)
            st.download_button(
                "⬇ Worked example (Markdown)",
                worked_example_md(
                    rec,
                    title=f"Worked example: {rec['ticket_id']}",
                    summary=rec["ticket"]["title"],
                    origin="exported from the Audit view",
                ),
                f"example-{rec['ticket_id']}.md",
                "text/markdown",
            )
            st.download_button(
                "⬇ Audit record (JSON)",
                json.dumps(rec, indent=2),
                f"audit-{analysis_id}.json",
                "application/json",
            )

# ---------------------------------------------------------------------------- chain + export
st.divider()
st.subheader("Hash chain and export")
if client.role == "admin":
    if st.button("Verify the whole audit chain"):
        try:
            result = client.audit_verify()
            if result.get("ok"):
                st.success(f"Audit chain verified ({result.get('rows', '?')} rows).")
            else:
                st.error(f"Audit chain BROKEN at row {result.get('broken_at')}.")
        except ApiError as err:
            st.error(f"Verify failed: {err.message}")
else:
    st.caption("Only an admin can verify the whole chain.")

if client.role == "kb_owner":
    st.caption("KB owners cannot export the audit log.")
else:
    with st.form("export"):
        c1, c2, c3 = st.columns(3)
        fmt = c1.selectbox("Format", ["csv", "jsonl"])
        d_from = c2.date_input("From", value=date(2026, 1, 1))
        d_to = c3.date_input("To", value=date.today())
        go = st.form_submit_button("Prepare export")
    if go:
        try:
            st.session_state["audit_export"] = (
                fmt,
                client.audit_export(fmt, d_from.isoformat(), d_to.isoformat()),
            )
        except ApiError as err:
            st.error(f"Export failed: {err.message}")
    if "audit_export" in st.session_state:
        fmt, text = st.session_state["audit_export"]
        masked = "unmasked" if client.role == "admin" else "masked"
        st.caption(f"{len(text.splitlines())} line(s), {masked} for role {client.role}.")
        st.download_button(
            f"⬇ Download audit export ({fmt})",
            text,
            f"audit-export.{fmt}",
            "text/csv" if fmt == "csv" else "application/x-ndjson",
        )
