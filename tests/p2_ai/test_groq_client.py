"""P2-T1: Groq client with a fake HTTP layer (no network)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ai.llm.groq_client import GroqClient, HttpResponse, TokenBucket, make_llm_client
from shared.interfaces import LLMClient, LLMUnavailableError


class FakeTransport:
    """Returns queued responses and records every request."""

    def __init__(self, responses: list[HttpResponse]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def __call__(
        self, url: str, headers: dict[str, str], body: dict[str, Any], timeout: float
    ) -> HttpResponse:
        self.requests.append({"url": url, "headers": headers, "body": body, "timeout": timeout})
        if not self.responses:
            raise AssertionError("no more fake responses")
        return self.responses.pop(0)


def ok(content: dict[str, Any], tokens: int = 42) -> HttpResponse:
    return HttpResponse(
        status=200,
        headers={},
        body={
            "choices": [{"message": {"content": json.dumps(content)}}],
            "usage": {"prompt_tokens": tokens - 2, "completion_tokens": 2, "total_tokens": tokens},
        },
    )


def err(status: int, retry_after: str | None = None) -> HttpResponse:
    headers = {"retry-after": retry_after} if retry_after else {}
    return HttpResponse(status=status, headers=headers, body={"error": {"message": "x"}})


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.now += s


def make(
    tmp_path: Path, transport: FakeTransport, *, key: str | None = "test-key", **kw: Any
) -> tuple[GroqClient, Clock, list[dict[str, Any]]]:
    clock = Clock()
    calls: list[dict[str, Any]] = []
    client = GroqClient(
        api_key=key,
        transport=transport,
        cache_path=tmp_path / "cache.sqlite",
        on_call=calls.append,
        clock=clock.time,
        sleep=clock.sleep,
        jitter=lambda: 0.0,
        **kw,
    )
    return client, clock, calls


def call(client: GroqClient, user: str = "hello") -> dict[str, Any]:
    return client.complete_json(purpose="classify", system="sys", user=user)


def test_satisfies_protocol(tmp_path: Path) -> None:
    client, _, _ = make(tmp_path, FakeTransport([]))
    assert isinstance(client, LLMClient)


def test_json_mode_temperature_zero_and_auth(tmp_path: Path) -> None:
    t = FakeTransport([ok({"a": 1})])
    client, _, _ = make(tmp_path, t)
    assert call(client) == {"a": 1}
    req = t.requests[0]
    assert req["body"]["response_format"] == {"type": "json_object"}
    assert req["body"]["temperature"] == 0
    assert req["body"]["model"] == "llama-3.1-8b-instant"
    assert req["headers"]["Authorization"] == "Bearer test-key"
    assert req["timeout"] == 20
    assert [m["role"] for m in req["body"]["messages"]] == ["system", "user"]


def test_cache_hit_on_second_identical_call(tmp_path: Path) -> None:
    t = FakeTransport([ok({"a": 1})])
    client, _, calls = make(tmp_path, t)
    assert call(client) == {"a": 1}
    assert call(client) == {"a": 1}
    assert len(t.requests) == 1
    assert [c["cache_hit"] for c in calls] == [False, True]


def test_cache_persists_across_instances(tmp_path: Path) -> None:
    client, _, _ = make(tmp_path, FakeTransport([ok({"a": 1})]))
    call(client)
    t2 = FakeTransport([])
    client2, _, _ = make(tmp_path, t2)
    assert call(client2) == {"a": 1}
    assert t2.requests == []


def test_different_prompt_is_a_cache_miss(tmp_path: Path) -> None:
    t = FakeTransport([ok({"a": 1}), ok({"a": 2})])
    client, _, _ = make(tmp_path, t)
    assert call(client, "one") == {"a": 1}
    assert call(client, "two") == {"a": 2}


def test_retry_on_429_honours_retry_after(tmp_path: Path) -> None:
    t = FakeTransport([err(429, retry_after="3"), ok({"a": 1})])
    client, clock, _ = make(tmp_path, t)
    assert call(client) == {"a": 1}
    assert 3.0 in clock.sleeps


def test_retry_on_5xx_with_exponential_backoff(tmp_path: Path) -> None:
    t = FakeTransport([err(503), err(502), ok({"a": 1})])
    client, clock, _ = make(tmp_path, t)
    assert call(client) == {"a": 1}
    backoffs = [s for s in clock.sleeps if s > 0]
    assert backoffs[:2] == [1.0, 2.0]


def test_gives_up_after_max_retries(tmp_path: Path) -> None:
    t = FakeTransport([err(429), err(429), err(429)])
    client, _, calls = make(tmp_path, t)
    with pytest.raises(LLMUnavailableError):
        call(client)
    assert len(t.requests) == 3  # 1 try + 2 retries
    assert calls[-1]["error"]


def test_4xx_is_not_retried(tmp_path: Path) -> None:
    t = FakeTransport([err(400)])
    client, _, _ = make(tmp_path, t)
    with pytest.raises(LLMUnavailableError):
        call(client)
    assert len(t.requests) == 1


def test_invalid_json_content_raises(tmp_path: Path) -> None:
    bad = HttpResponse(status=200, headers={}, body={"choices": [{"message": {"content": "nope"}}]})
    client, _, _ = make(tmp_path, FakeTransport([bad]))
    with pytest.raises(LLMUnavailableError):
        call(client)


def test_circuit_breaker_opens_after_three_failures_then_recovers(tmp_path: Path) -> None:
    t = FakeTransport([err(400), err(400), err(400), ok({"a": 1})])
    client, clock, _ = make(tmp_path, t)
    for i in range(3):
        with pytest.raises(LLMUnavailableError):
            call(client, f"q{i}")
    assert client.available is False
    with pytest.raises(LLMUnavailableError):
        call(client, "q-open")
    assert len(t.requests) == 3  # breaker skipped the HTTP call
    clock.now += 61
    assert client.available is True
    assert call(client, "q-later") == {"a": 1}


def test_offline_when_no_key(tmp_path: Path) -> None:
    t = FakeTransport([])
    client, _, _ = make(tmp_path, t, key=None)
    assert client.available is False
    assert client.mode == "offline"
    with pytest.raises(LLMUnavailableError):
        call(client)
    assert t.requests == []


def test_on_call_never_contains_prompt_or_key(tmp_path: Path) -> None:
    client, _, calls = make(tmp_path, FakeTransport([ok({"a": 1})]))
    client.complete_json(purpose="synthesize", system="SYSTEXT", user="SECRET TICKET TEXT")
    rec = calls[0]
    assert rec["purpose"] == "synthesize"
    assert rec["tokens"] == 42
    blob = json.dumps(rec)
    assert "SECRET TICKET TEXT" not in blob and "test-key" not in blob and "SYSTEXT" not in blob


def test_on_call_errors_do_not_break_the_call(tmp_path: Path) -> None:
    def boom(_: dict[str, Any]) -> None:
        raise RuntimeError("audit down")

    client = GroqClient(
        api_key="k",
        transport=FakeTransport([ok({"a": 1})]),
        cache_path=tmp_path / "c.sqlite",
        on_call=boom,
    )
    assert call(client) == {"a": 1}


def test_token_bucket_waits_when_empty() -> None:
    clock = Clock()
    bucket = TokenBucket(rate_per_min=60, clock=clock.time, sleep=clock.sleep)  # 1 per second
    for _ in range(60):
        bucket.acquire()
    assert clock.sleeps == []
    bucket.acquire()
    assert sum(clock.sleeps) == pytest.approx(1.0, abs=0.01)


def test_make_llm_client_offline_without_env(tmp_path: Path) -> None:
    client = make_llm_client(cache_path=tmp_path / "c.sqlite")
    assert client.available is False
