"""P4-T7: MTTR dashboard loads its charts from /metrics/*."""

from __future__ import annotations

from tests.p4_ui.helpers import all_text, page
from ui.components.metrics import fmt_seconds, group_rows, trial_comparison


def test_helpers():
    assert fmt_seconds(None) == "-"
    assert fmt_seconds(42) == "42s"
    assert fmt_seconds(90) == "1.5 min"
    assert fmt_seconds(7200) == "2.0 h"
    rows = group_rows([{"key": "Network", "n": 2, "mean_s": 60, "median_s": 60, "p90_s": 90}])
    assert rows[0]["Median"] == "1.0 min"
    cmp_ = trial_comparison(
        {"manual": {"n": 4, "median_s": 200.0}, "assisted": {"n": 4, "median_s": 150.0}}
    )
    assert cmp_["pct_change"] == -25.0
    assert trial_comparison({"manual": {"n": 1, "median_s": 10.0}}) is None


def simulate(make_client, clock, demo_tickets):
    """Two assisted decisions and one manual decision, with known durations."""
    client = make_client("l2")
    for demo, secs in (("DEMO-001", 60), ("DEMO-003", 120)):
        r = client.analyze(demo_tickets[demo])
        aid = r.details.analysis_id
        client.event(r.ticket_id, "ui.viewed", aid)
        clock.tick(secs)
        client.decide({"analysis_id": aid, "final_decision": "resolved", "mode": "assisted"})
        client.feedback({"analysis_id": aid, "target_type": "recommendation", "verdict": "useful"})
    client.event("T01", "manual.started", "manual-T01-x", mode="manual")
    clock.tick(300)
    client.decide(
        {
            "analysis_id": "manual-T01-x",
            "final_decision": "resolved",
            "final_category": "Network",
            "mode": "manual",
        }
    )
    client.event("T01", "manual.decision", "manual-T01-x", mode="manual")
    return client


def test_dashboard_loads_metrics(make_client, clock, demo_tickets, backend):
    calls = []
    original = backend.handle

    def spy(method, path, **kw):
        calls.append(path)
        return original(method, path, **kw)

    client = simulate(make_client, clock, demo_tickets)
    backend.handle = spy
    at = page("7_MTTR_dashboard.py", client)
    assert not at.exception, at.exception
    assert "/metrics/summary" in calls and "/metrics/mttr" in calls
    assert at.metric[0].value == "2"  # analyses
    # default view: TTD by category -> a chart + a table
    assert len(at.get("arrow_vega_lite_chart")) >= 1 or len(at.get("vega_lite_chart")) >= 1
    table = at.dataframe[0].value
    assert set(table["Group"]) >= {"Network", "Hardware"}
    text = all_text(at)
    assert "SYNTHETIC" in text


def test_trial_comparison_tiles(make_client, clock, demo_tickets):
    client = simulate(make_client, clock, demo_tickets)
    at = page("7_MTTR_dashboard.py", client)
    values = {m.label: m.value for m in at.metric}
    assert values["Manual median TTD"] == "5.0 min"
    assert values["Assisted median TTD"] == "1.5 min"
    assert values["Change"] == "-70.0%"


def test_group_by_mode(make_client, clock, demo_tickets):
    client = simulate(make_client, clock, demo_tickets)
    at = page("7_MTTR_dashboard.py", client)
    at.selectbox(key="d_group").select("mode").run()
    table = at.dataframe[0].value
    assert dict(zip(table["Group"], table["n"], strict=True)) == {"assisted": 2, "manual": 1}


def test_empty_dashboard(make_client):
    at = page("7_MTTR_dashboard.py", make_client("l1"))
    assert not at.exception
    assert "needs decisions in both" in all_text(at)
