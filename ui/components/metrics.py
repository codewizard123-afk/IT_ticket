"""Pure helpers for the MTTR dashboard (no Streamlit, so they are easy to test)."""

from __future__ import annotations

from typing import Any

METRICS = {
    "ttd": "TTD: time to decision (trial measure)",
    "ttr": "TTR: time to recommendation",
    "mttr": "MTTR: submitted to resolved",
}
GROUP_BYS = ("category", "priority", "mode", "decision")


def fmt_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


def group_rows(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """API groups → table rows with readable durations."""
    return [
        {
            "Group": g["key"],
            "n": g["n"],
            "Mean": fmt_seconds(g.get("mean_s")),
            "Median": fmt_seconds(g.get("median_s")),
            "P90": fmt_seconds(g.get("p90_s")),
            "median_s": g.get("median_s"),
        }
        for g in groups
    ]


def trial_comparison(ttd_by_mode: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """Median TTD manual vs assisted and the % change (negative = assisted is faster)."""
    manual, assisted = ttd_by_mode.get("manual"), ttd_by_mode.get("assisted")
    if not manual or not assisted or not manual.get("median_s"):
        return None
    change = (assisted["median_s"] - manual["median_s"]) / manual["median_s"] * 100
    return {
        "manual_median_s": manual["median_s"],
        "assisted_median_s": assisted["median_s"],
        "manual_n": manual["n"],
        "assisted_n": assisted["n"],
        "pct_change": round(change, 1),
    }
