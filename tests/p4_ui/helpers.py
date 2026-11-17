"""Helpers for Streamlit ``AppTest`` page tests."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from ui.client import ApiClient

UI_DIR = Path(__file__).resolve().parents[2] / "ui"


def page(name: str, client: ApiClient) -> AppTest:
    """Load ``ui/pages/<name>`` (or ``ui/app.py``) with an injected client, and run it."""
    path = UI_DIR / "app.py" if name == "app" else UI_DIR / "pages" / name
    at = AppTest.from_file(str(path), default_timeout=30)
    at.session_state["client"] = client
    return at.run()


def all_markdown(at: AppTest) -> str:
    return "\n".join(str(m.value) for m in at.markdown)


def all_text(at: AppTest) -> str:
    parts = [str(m.value) for m in at.markdown]
    parts += [str(c.value) for c in at.caption]
    for kind in ("success", "warning", "error", "info"):
        parts += [str(e.value) for e in getattr(at, kind)]
    return "\n".join(parts)


def click(at: AppTest, label: str) -> AppTest:
    for b in at.button:
        if b.label == label:
            return b.click().run()
    raise AssertionError(f"no button {label!r}; have {[b.label for b in at.button]}")
