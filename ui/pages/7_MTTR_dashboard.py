"""MTTR dashboard (FR-09, team-defined): TTR / TTD / MTTR, trial results, feedback, degraded."""

from __future__ import annotations

import streamlit as st

from ui.client import ApiError
from ui.components.badges import plain_chip
from ui.components.metrics import GROUP_BYS, METRICS, fmt_seconds, group_rows, trial_comparison
from ui.state import header

client = header("📈 MTTR dashboard")

try:
    summary = client.metrics_summary()
except ApiError as err:
    st.error(f"Metrics unavailable: {err.message}")
    st.stop()

# ---------------------------------------------------------------------------- headline tiles
c1, c2, c3, c4 = st.columns(4)
c1.metric("Analyses", summary.get("analyses", 0))
c2.metric("Escalation rate", f"{summary.get('escalation_rate', 0):.0%}")
fb = summary.get("feedback", {})
c3.metric("Feedback: useful", f"{fb.get('rates', {}).get('useful', 0):.0%}", help="of all votes")
deg = summary.get("degraded", {})
c4.metric("Degraded analyses", deg.get("count", 0), help="LLM fallback / offline")

# ---------------------------------------------------------------------------- timing by group
st.subheader("Timing")
c1, c2, c3 = st.columns(3)
metric = c1.selectbox("Metric", list(METRICS), format_func=METRICS.get, key="d_metric")
group_by = c2.selectbox("Group by", GROUP_BYS, key="d_group")
mode = c3.selectbox("Mode", ["", "assisted", "manual"], format_func=lambda m: m or "both")
try:
    data = client.metrics_mttr(group_by=group_by, metric=metric, mode=mode or None)
except ApiError as err:
    st.error(f"/metrics/mttr failed: {err.message}")
    data = {"groups": []}
rows = group_rows(data.get("groups", []))
if rows:
    st.bar_chart(
        {r["Group"]: r["median_s"] for r in rows},
        x_label=group_by,
        y_label=f"median {metric.upper()} (seconds)",
    )
    st.dataframe([{k: v for k, v in r.items() if k != "median_s"} for r in rows], hide_index=True)
else:
    st.info(f"No {metric.upper()} measurements yet for this selection.")

# ---------------------------------------------------------------------------- trial
st.subheader("Timed trial: manual vs assisted (TTD)")
comparison = trial_comparison(summary.get("ttd_by_mode", {}))
if comparison:
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Manual median TTD",
        fmt_seconds(comparison["manual_median_s"]),
        help=f"n={comparison['manual_n']}",
    )
    c2.metric(
        "Assisted median TTD",
        fmt_seconds(comparison["assisted_median_s"]),
        help=f"n={comparison['assisted_n']}",
    )
    c3.metric(
        "Change",
        f"{comparison['pct_change']:+.1f}%",
        delta_color="inverse",
        delta=f"{comparison['pct_change']:+.1f}%",
    )
    st.caption(
        "Indicative only: a small trial whose participants partly built the system (README §15). "
        "Correctness against trial gold is computed offline by `make mttr-report`."
    )
else:
    st.info("The trial needs decisions in both manual and assisted mode before it can compare.")

# ---------------------------------------------------------------------------- feedback / degraded
c1, c2 = st.columns(2)
with c1:
    st.subheader("Feedback")
    if fb.get("total"):
        st.bar_chart(
            {v: fb.get(v, 0) for v in ("useful", "incorrect", "unresolved")},
            x_label="verdict",
            y_label="votes",
        )
        st.caption(
            " · ".join(f"{v}: {fb['rates'].get(v, 0):.0%}" for v in fb.get("rates", {}))
            + f" (n={fb['total']})"
        )
    else:
        st.caption("No feedback yet.")
with c2:
    st.subheader("Degraded mode")
    by_mode = deg.get("by_mode", {})
    if by_mode:
        st.bar_chart(by_mode, x_label="degraded flag", y_label="analyses")
    st.caption(f"{deg.get('count', 0)} degraded analyses ({deg.get('rate', 0):.0%}).")

# ---------------------------------------------------------------------------- synthetic baseline
st.subheader("Historical baseline")
hist = summary.get("historical_baseline") or {}
st.markdown(
    plain_chip("SYNTHETIC", "#757575")
    + " "
    + (hist.get("label") or "Historical MTTR from synthetic timestamps, context only"),
    unsafe_allow_html=True,
)
if hist.get("n"):
    st.write(
        f"Mean {fmt_seconds(hist.get('mean_s'))} · median {fmt_seconds(hist.get('median_s'))} "
        f"over {hist['n']} historical tickets. Not a measurement of this system."
    )
else:
    st.caption("No historical timestamps loaded (P1's enriched data adds them).")
