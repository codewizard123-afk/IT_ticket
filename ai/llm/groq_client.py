"""Groq chat-completions client (architecture §12).

- JSON mode, temperature 0, configurable timeout
- retries with exponential backoff + jitter on 429/5xx, honouring `retry-after`
- token-bucket limiter below the RPM limit
- SQLite response cache keyed by sha1(model + prompt)
- circuit breaker: skip Groq for `breaker_cooldown_s` after `breaker_failures` consecutive failures
- `on_call` callback for P3's `llm_calls` audit table
- `offline` mode when there is no API key: every call raises `LLMUnavailable`

The API key is never logged or included in any exception or callback payload.
Callers must pass already-redacted text (`privacy.redact`).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from ai.config import REPO_ROOT, ai_config
from ai.types import LLMBadOutput, LLMUnavailable

log = logging.getLogger(__name__)

OnCall = Callable[[dict[str, Any]], None]


class TokenBucket:
    """Simple token bucket: `rate_per_min` tokens refill continuously, capacity = rate."""

    def __init__(
        self,
        rate_per_min: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.capacity = max(1.0, float(rate_per_min))
        self.rate = self.capacity / 60.0
        self.tokens = self.capacity
        self._clock = clock
        self._sleep = sleep
        self._last = clock()
        self._lock = threading.Lock()

    def acquire(self, n: float = 1.0) -> float:
        """Take `n` tokens, sleeping if needed. Returns the seconds waited."""
        waited = 0.0
        with self._lock:
            while True:
                now = self._clock()
                self.tokens = min(
                    self.capacity, self.tokens + (now - self._last) * self.rate
                )
                self._last = now
                if self.tokens >= n:
                    self.tokens -= n
                    return waited
                wait = (n - self.tokens) / self.rate
                self._sleep(wait)
                waited += wait


class CircuitBreaker:
    def __init__(
        self,
        failures: int,
        cooldown_s: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_failures = failures
        self.cooldown_s = cooldown_s
        self._clock = clock
        self.consecutive = 0
        self.opened_at: float | None = None

    @property
    def is_open(self) -> bool:
        if self.opened_at is None:
            return False
        if self._clock() - self.opened_at >= self.cooldown_s:
            # half-open: allow one trial call
            self.opened_at = None
            self.consecutive = self.max_failures - 1
            return False
        return True

    def success(self) -> None:
        self.consecutive = 0
        self.opened_at = None

    def failure(self) -> None:
        self.consecutive += 1
        if self.consecutive >= self.max_failures:
            self.opened_at = self._clock()


class SQLiteCache:
    def __init__(self, path: str | Path | None) -> None:
        self.path = str(path) if path else ":memory:"
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS llm_cache (key TEXT PRIMARY KEY, model TEXT, "
            "response TEXT, tokens_in INTEGER, tokens_out INTEGER, created_at TEXT)"
        )
        self._conn.commit()
        self._lock = threading.Lock()

    def get(self, key: str) -> tuple[str, int, int] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT response, tokens_in, tokens_out FROM llm_cache WHERE key = ?",
                (key,),
            ).fetchone()
        return (row[0], int(row[1] or 0), int(row[2] or 0)) if row else None

    def put(
        self, key: str, model: str, response: str, tokens_in: int, tokens_out: int
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO llm_cache VALUES (?, ?, ?, ?, ?, ?)",
                (
                    key,
                    model,
                    response,
                    tokens_in,
                    tokens_out,
                    datetime.now(UTC).isoformat(),
                ),
            )
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()


@dataclass
class _HttpOutcome:
    content: str
    tokens_in: int
    tokens_out: int


class _RetryableError(Exception):
    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


def cache_key(model: str, messages: list[dict[str, str]]) -> str:
    prompt = json.dumps(messages, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1((model + prompt).encode("utf-8")).hexdigest()


def parse_json_object(content: str) -> dict[str, Any]:
    """Parse the model's JSON answer; tolerate code fences. Raises `LLMBadOutput`."""
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise LLMBadOutput("LLM output is not JSON") from exc
        try:
            obj = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc2:
            raise LLMBadOutput("LLM output is not JSON") from exc2
    if not isinstance(obj, dict):
        raise LLMBadOutput("LLM output is not a JSON object")
    return obj


class GroqClient:
    """Implements the `LLMClient` protocol from `ai.types`."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        config: dict[str, Any] | None = None,
        http: httpx.Client | None = None,
        cache_path: str | Path | None = None,
        on_call: OnCall | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
    ) -> None:
        cfg = dict((config if config is not None else ai_config()).get("llm", {}))
        self._api_key = (
            api_key if api_key is not None else os.environ.get("GROQ_API_KEY", "")
        )
        self.base_url: str = cfg.get(
            "base_url", "https://api.groq.com/openai/v1/chat/completions"
        )
        self.temperature: float = float(cfg.get("temperature", 0))
        self.timeout_s: float = float(cfg.get("timeout_s", 20))
        self.max_retries: int = int(cfg.get("max_retries", 2))
        self.backoff_base_s: float = float(cfg.get("backoff_base_s", 1.0))
        self.backoff_max_s: float = float(cfg.get("backoff_max_s", 8.0))
        self.max_tokens: int = int(cfg.get("max_tokens", 1024))
        self.classify_model: str = cfg.get("classify_model", "llama-3.1-8b-instant")
        self.synth_model: str = cfg.get("synth_model", "llama-3.3-70b-versatile")
        self._sleep = sleep
        self._rng = rng or random.Random()
        self._bucket = TokenBucket(float(cfg.get("rpm", 25)), clock=clock, sleep=sleep)
        self._breaker = CircuitBreaker(
            int(cfg.get("breaker_failures", 3)),
            float(cfg.get("breaker_cooldown_s", 60)),
            clock=clock,
        )
        if cache_path is None and cfg.get("cache"):
            cache_path = REPO_ROOT / cfg["cache"]
        self._cache = SQLiteCache(cache_path)
        self._http = http
        self.on_call = on_call

    # ------------------------------------------------------------------ status

    @property
    def offline(self) -> bool:
        return not self._api_key

    @property
    def available(self) -> bool:
        return not self.offline and not self._breaker.is_open

    def __repr__(self) -> str:  # never expose the key
        return f"GroqClient(offline={self.offline})"

    # ------------------------------------------------------------------ public

    def complete_json(
        self,
        *,
        purpose: str,
        model: str,
        messages: list[dict[str, str]],
        analysis_id: str | None = None,
    ) -> dict[str, Any]:
        key = cache_key(model, messages)
        record: dict[str, Any] = {
            "call_id": str(uuid.uuid4()),
            "analysis_id": analysis_id,
            "purpose": purpose,
            "model": model,
            "prompt_hash": key,
            "tokens_in": 0,
            "tokens_out": 0,
            "latency_ms": 0,
            "status": "ok",
            "cached": 0,
            "created_at": datetime.now(UTC).isoformat(),
        }

        cached = self._cache.get(key)
        if cached is not None:
            content, tin, tout = cached
            record.update(tokens_in=tin, tokens_out=tout, cached=1)
            self._emit(record)
            return parse_json_object(content)

        if self.offline:
            record["status"] = "offline"
            self._emit(record)
            raise LLMUnavailable("offline: GROQ_API_KEY not set")
        if self._breaker.is_open:
            record["status"] = "circuit_open"
            self._emit(record)
            raise LLMUnavailable("circuit breaker open")

        t0 = time.perf_counter()
        try:
            outcome = self._call_with_retries(model, messages)
        except LLMUnavailable as exc:
            self._breaker.failure()
            record.update(
                status="error", latency_ms=int((time.perf_counter() - t0) * 1000)
            )
            self._emit(record)
            raise LLMUnavailable(str(exc)) from None
        record["latency_ms"] = int((time.perf_counter() - t0) * 1000)
        record.update(tokens_in=outcome.tokens_in, tokens_out=outcome.tokens_out)

        try:
            obj = parse_json_object(outcome.content)
        except LLMBadOutput:
            self._breaker.success()  # the service answered; output problem is the caller's
            record["status"] = "bad_output"
            self._emit(record)
            raise
        self._breaker.success()
        self._cache.put(
            key, model, outcome.content, outcome.tokens_in, outcome.tokens_out
        )
        self._emit(record)
        return obj

    # ------------------------------------------------------------------ internals

    def _emit(self, record: dict[str, Any]) -> None:
        if self.on_call is None:
            return
        try:
            self.on_call(dict(record))
        except Exception:  # noqa: BLE001 - an audit callback must never break analysis
            log.warning("llm on_call callback failed", exc_info=False)

    def _client(self) -> httpx.Client:
        if self._http is None:
            self._http = httpx.Client(timeout=self.timeout_s)
        return self._http

    def _backoff(self, attempt: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return min(self.backoff_max_s, max(0.0, retry_after))
        base = min(self.backoff_max_s, self.backoff_base_s * (2**attempt))
        return base * (0.5 + self._rng.random() / 2)

    def _call_with_retries(
        self, model: str, messages: list[dict[str, str]]
    ) -> _HttpOutcome:
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            approx_tokens = sum(len(m.get("content", "")) for m in messages) // 4
            self._bucket.acquire(1.0)
            try:
                return self._post(model, messages)
            except _RetryableError as exc:
                last = exc
                if attempt >= self.max_retries:
                    break
                wait = self._backoff(attempt, exc.retry_after)
                log.info(
                    "groq retry %d after %.1fs (%s, ~%d tokens)",
                    attempt + 1,
                    wait,
                    exc,
                    approx_tokens,
                )
                self._sleep(wait)
        raise LLMUnavailable(f"groq failed after retries: {last}")

    def _post(self, model: str, messages: list[dict[str, str]]) -> _HttpOutcome:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        try:
            resp = self._client().post(
                self.base_url, json=payload, headers=headers, timeout=self.timeout_s
            )
        except httpx.TimeoutException as exc:
            raise _RetryableError("timeout") from exc
        except httpx.HTTPError as exc:
            raise _RetryableError(type(exc).__name__) from exc

        if resp.status_code == 429 or resp.status_code >= 500:
            retry_after: float | None = None
            raw = resp.headers.get("retry-after")
            if raw:
                try:
                    retry_after = float(raw)
                except ValueError:
                    retry_after = None
            raise _RetryableError(f"http {resp.status_code}", retry_after)
        if resp.status_code >= 400:
            # 4xx other than 429: not retryable (bad request, auth, model removed)
            raise LLMUnavailable(f"http {resp.status_code}")

        try:
            body = resp.json()
            content = body["choices"][0]["message"]["content"] or ""
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMBadOutput("unexpected Groq response shape") from exc
        usage = body.get("usage") or {}
        return _HttpOutcome(
            content=content,
            tokens_in=int(usage.get("prompt_tokens", 0) or 0),
            tokens_out=int(usage.get("completion_tokens", 0) or 0),
        )


def make_client(on_call: OnCall | None = None) -> GroqClient:
    """Factory used by the engine/backend: reads `GROQ_API_KEY` from the environment."""
    return GroqClient(on_call=on_call)
