"""Collects LLM call records for the ``llm_calls`` table (architecture §8.7, §12).

``ai.llm.groq_client`` calls ``on_call(record)`` from the request's own thread; the pipeline wraps
``engine.run`` in :func:`capture`, so each record lands in the right analysis. Records hold only
metadata (purpose, model, tokens, latency, status, cache hit), never prompt text.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

_calls: ContextVar[list[dict[str, Any]] | None] = ContextVar("llm_calls", default=None)

_KEEP = ("purpose", "model", "cache_hit", "status", "tokens", "latency_ms", "attempts", "error")


def record(rec: dict[str, Any]) -> None:
    """``on_call`` callback for the Groq client."""
    sink = _calls.get()
    if sink is not None:
        row = {k: rec.get(k) for k in _KEEP}
        row["created_at"] = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
        sink.append(row)


@contextmanager
def capture() -> Iterator[list[dict[str, Any]]]:
    sink: list[dict[str, Any]] = []
    token = _calls.set(sink)
    try:
        yield sink
    finally:
        _calls.reset(token)
