"""Review queue (FR-05): clarify / escalate / needs-approval items, by priority then age."""

from __future__ import annotations

import streamlit as st

from ui.client import ApiError
from ui.components.badges import chip
from ui.components.feedback import decision_form
from ui.state import header

client = header("📥 Review queue")
STATUSES = {
    "open": "Open (no decision yet)",
    "escalate": "Escalations",
    "clarify": "Waiting for clarification",
    "approval": "Needs approval",
    "all": "All",
}

c1, c2 = st.columns([3, 1])
status = c1.selectbox("Show", list(STATUSES), format_func=STATUSES.get, key="rq_status")
page_no = int(c2.number_input("Page", min_value=1, value=1, step=1, key="rq_page"))
try:
    data = client.review_queue(status=status, page=page_no)
except ApiError as err:
    st.error(f"Review queue unavailable: {err.message}")
    st.stop()

items = data["items"]
st.caption(f"{data['total']} item(s), sorted by priority then age (oldest first).")
if not items:
    st.info("Nothing to review.")
    st.stop()


def _age(seconds: float) -> str:
    minutes = int(seconds // 60)
    return f"{minutes // 60}h {minutes % 60}m" if minutes >= 60 else f"{minutes}m"


st.dataframe(
    [
        {
            "Analysis": i["analysis_id"],
            "Ticket": i["ticket_id"],
            "Title": i["title"],
            "Decision": i["decision"].upper(),
            "Priority": i["priority"],
            "Classification": f"{i['category']}/{i['subcategory']}",
            "Confidence": i["confidence"],
            "Approval": "REQUIRED" if i["human_approval_required"] else "-",
            "Group": i["assignment_group"],
            "Age": _age(float(i["age_s"])),
            "Decided": "yes" if i.get("decided") else "no",
        }
        for i in items
    ],
    hide_index=True,
)

labels = {i["analysis_id"]: f"{i['ticket_id']} · {i['title']}" for i in items}
chosen = st.selectbox(
    "Review item", ["", *labels], format_func=lambda a: labels.get(a, "— choose —"), key="rq_open"
)
if chosen:
    item = next(i for i in items if i["analysis_id"] == chosen)
    st.markdown(
        f"{chip('decision', item['decision'])} {chip('priority', item['priority'])} "
        f"**{item['ticket_id']}** · {item['title']}",
        unsafe_allow_html=True,
    )
    if item["approval_reasons"]:
        st.caption("Approval reasons: " + "; ".join(item["approval_reasons"]))
    st.caption("Override the labels or priority below; the override is stored with the decision.")
    decision_form(
        client,
        chosen,
        mode="assisted",
        defaults={
            "final_decision": {"escalate": "escalated", "clarify": "needs_more_info"}.get(
                item["decision"], "resolved"
            ),
            "category": item["category"],
            "subcategory": item["subcategory"],
            "priority": item["priority"],
        },
        key="review",
    )
