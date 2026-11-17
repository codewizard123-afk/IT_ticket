"""Groq chat-completions client (architecture §12). Owner: P2, used by everyone.

* JSON mode, ``temperature=0``, 20 s timeout
* retries on 429/5xx with exponential backoff + jitter, honouring ``retry-after``
* token buckets below the RPM/TPM limits
* SQLite cache keyed by ``sha1(model + prompt)`` (``data/llm_cache.sqlite``)
* circuit breaker: skip Groq for 60 s after 3 consecutive failed calls
* every call is reported through the ``on_call`` callback P3 injects (no prompt text, no key)
* ``offline`` mode when there is no ``GROQ_API_KEY``: every call raises ``LLMUnavailableError``

The HTTP layer is injectable (``transport``) so tests never touch the network.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import random
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai.config import ROOT, ai_config
from shared.interfaces import LLMUnavailableError

log = logging.getLogger(__name__)

OnCall = Callable[[dict[str, Any]], None]


@dataclass
class HttpResponse:
    status: int
    headers: dict[str, str] = field(default_factory=dict)
    body: dict[str, Any] = field(default_factory=dict)


Transport = Callable[[str, dict[str, str], dict[str, Any], float], HttpResponse]


def httpx_transport(
    url: str, headers: dict[str, str], body: dict[str, Any], timeout: float
) -> HttpResponse:
    """Default transport. ``httpx`` is imported lazily so offline code never needs it."""
    import httpx  # noqa: PLC0415

    try:
        r = httpx.post(url, headers=headers, json=body, timeout=timeout)
    except httpx.HTTPError as exc:  # network error → treat like a 503
        return HttpResponse(status=503, body={"error": {"message": type(exc).__name__}})
    try:
        payload = r.json()
    except ValueError:
        payload = {}
    return HttpResponse(
        status=r.status_code, headers={k.lower(): v for k, v in r.headers.items()}, body=payload
    )


class TokenBucket:
    """Classic token bucket. ``rate_per_min`` tokens refill per minute, burst = one minute."""

    def __init__(
        self,
        rate_per_min: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.capacity = float(rate_per_min)
        self.rate = float(rate_per_min) / 60.0
        self.tokens = self.capacity
        self._clock = clock
        self._sleep = sleep
        self._last = clock()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = self._clock()
        self.tokens = min(self.capacity, self.tokens + (now - self._last) * self.rate)
        self._last = now

    def acquire(self, n: float = 1.0) -> None:
        n = min(float(n), self.capacity)
        with self._lock:
            self._refill()
            if self.tokens < n:
                wait = (n - self.tokens) / self.rate
                self._sleep(wait)
                self._refill()
            self.tokens -= n


class _Cache:
    """Tiny SQLite key/value cache. One connection per operation (thread-safe enough)."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with self._conn() as c:
                c.execute("CREATE TABLE IF NOT EXISTS llm_cache (k TEXT PRIMARY KEY, v TEXT)")
        except sqlite3.Error:
            log.warning("LLM cache disabled: cannot open %s", path)
            self.path = None

    def _conn(self) -> sqlite3.Connection:
        assert self.path is not None
        return sqlite3.connect(self.path, timeout=5)

    def get(self, key: str) -> dict[str, Any] | None:
        if self.path is None:
            return None
        try:
            with contextlib.closing(self._conn()) as c:
                row = c.execute("SELECT v FROM llm_cache WHERE k = ?", (key,)).fetchone()
        except sqlite3.Error:
            return None
        return json.loads(row[0]) if row else None

    def put(self, key: str, value: dict[str, Any]) -> None:
        if self.path is None:
            return
        try:
            with contextlib.closing(self._conn()) as c, c:
                c.execute(
                    "INSERT OR REPLACE INTO llm_cache (k, v) VALUES (?, ?)",
                    (key, json.dumps(value)),
                )
        except sqlite3.Error:
            log.warning("LLM cache write failed")


class GroqClient:
    """``LLMClient`` implementation for Groq's OpenAI-compatible endpoint."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        config: dict[str, Any] | None = None,
        transport: Transport | None = None,
        cache_path: Path | str | None = None,
        on_call: OnCall | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[], float] | None = None,
    ) -> None:
        cfg = config if config is not None else ai_config()["llm"]
        self._cfg = cfg
        self._key = api_key or None
        self._transport = transport or httpx_transport
        self._on_call = on_call
        self._clock = clock
        self._sleep = sleep
        self._jitter = jitter or (lambda: random.uniform(0, 0.25))
        if cache_path is None:
            cache_path = ROOT / cfg.get("cache", "data/llm_cache.sqlite")
        self._cache = _Cache(Path(cache_path))
        self._rpm = TokenBucket(cfg.get("rpm", 25), clock=clock, sleep=sleep)
        self._tpm = TokenBucket(cfg.get("tpm", 5500), clock=clock, sleep=sleep)
        self._failures = 0
        self._open_until = 0.0
        self._lock = threading.Lock()

    # --- state ---------------------------------------------------------------------------------
    @property
    def mode(self) -> str:
        return "online" if self._key else "offline"

    @property
    def breaker_open(self) -> bool:
        return self._clock() < self._open_until

    @property
    def available(self) -> bool:
        return self._key is not None and not self.breaker_open

    def set_on_call(self, on_call: OnCall | None) -> None:
        """Let P3 attach its ``llm_calls`` audit callback after construction."""
        self._on_call = on_call

    # --- helpers -------------------------------------------------------------------------------
    @staticmethod
    def cache_key(model: str, system: str, user: str, max_tokens: int) -> str:
        raw = json.dumps([model, system, user, max_tokens], ensure_ascii=False)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def _report(self, rec: dict[str, Any]) -> None:
        if self._on_call is None:
            return
        try:
            self._on_call(rec)
        except Exception:  # noqa: BLE001 - the audit callback must never break analysis
            log.warning("on_call callback failed for purpose=%s", rec.get("purpose"))

    def _record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= int(self._cfg.get("breaker_failures", 3)):
                self._open_until = self._clock() + float(self._cfg.get("breaker_cooldown_s", 60))
                self._failures = 0
                log.warning("Groq circuit breaker open")

    def _record_success(self) -> None:
        with self._lock:
            self._failures = 0

    def _backoff(self, attempt: int, retry_after: str | None) -> float:
        base = float(self._cfg.get("backoff_base_s", 1.0))
        cap = float(self._cfg.get("backoff_max_s", 8.0))
        delay = min(cap, base * (2**attempt)) + self._jitter()
        if retry_after:
            with contextlib.suppress(ValueError):
                delay = max(delay, float(retry_after))
        return delay

    # --- main entry point ------------------------------------------------------------------------
    def complete_json(
        self,
        *,
        purpose: str,
        system: str,
        user: str,
        model: str | None = None,
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        model = model or self._cfg.get("classify_model", "llama-3.1-8b-instant")
        rec: dict[str, Any] = {
            "purpose": purpose,
            "model": model,
            "cache_hit": False,
            "status": None,
            "tokens": 0,
            "latency_ms": 0,
            "attempts": 0,
            "error": None,
        }
        if self._key is None:
            rec["error"] = "offline"
            self._report(rec)
            raise LLMUnavailableError("no GROQ_API_KEY: offline mode")

        key = self.cache_key(model, system, user, max_tokens)
        cached = self._cache.get(key)
        if cached is not None:
            rec.update(cache_hit=True, status=200)
            self._report(rec)
            return cached

        if self.breaker_open:
            rec["error"] = "circuit_open"
            self._report(rec)
            raise LLMUnavailableError("Groq circuit breaker is open")

        body = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": self._cfg.get("temperature", 0),
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}
        timeout = float(self._cfg.get("timeout_s", 20))
        retries = int(self._cfg.get("max_retries", 2))
        t0 = self._clock()

        error = "unknown"
        for attempt in range(retries + 1):
            self._rpm.acquire()
            self._tpm.acquire((len(system) + len(user)) / 4 + max_tokens)
            rec["attempts"] = attempt + 1
            resp = self._transport(self._cfg["base_url"], headers, body, timeout)
            rec["status"] = resp.status
            if resp.status == 200:
                try:
                    content = resp.body["choices"][0]["message"]["content"]
                    parsed = json.loads(content)
                    if not isinstance(parsed, dict):
                        raise ValueError("JSON root is not an object")
                except (KeyError, IndexError, TypeError, ValueError):
                    error = "invalid_json"
                    break
                rec["tokens"] = int((resp.body.get("usage") or {}).get("total_tokens", 0))
                rec["latency_ms"] = int((self._clock() - t0) * 1000)
                self._record_success()
                self._cache.put(key, parsed)
                self._report(rec)
                return parsed
            error = f"http_{resp.status}"
            if resp.status == 429 or resp.status >= 500:
                if attempt < retries:
                    self._sleep(self._backoff(attempt, resp.headers.get("retry-after")))
                continue
            break  # other 4xx: not retryable

        rec["error"] = error
        rec["latency_ms"] = int((self._clock() - t0) * 1000)
        self._record_failure()
        self._report(rec)
        raise LLMUnavailableError(f"Groq call failed: {error}")


def make_llm_client(
    on_call: OnCall | None = None,
    *,
    cache_path: Path | str | None = None,
    api_key: str | None = None,
) -> GroqClient:
    """Factory used by P3's ``backend/deps.py``. Reads ``GROQ_API_KEY`` from the environment."""
    key = api_key if api_key is not None else os.environ.get("GROQ_API_KEY", "").strip()
    return GroqClient(api_key=key or None, cache_path=cache_path, on_call=on_call)
