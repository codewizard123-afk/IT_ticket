"""P2-T1: Groq client with a fake HTTP layer (httpx.MockTransport). No network."""

from __future__ import annotations

import json

import httpx
import pytest

from ai.llm.groq_client import (
    CircuitBreaker,
    GroqClient,
    TokenBucket,
    parse_json_object,
)
from ai.types import LLMBadOutput, LLMClient, LLMUnavailable

CFG = {
    "llm": {
        "base_url": "https://groq.test/v1/chat/completions",
        "classify_model": "small",
        "synth_model": "big",
        "temperature": 0,
        "timeout_s": 5,
        "max_retries": 2,
        "backoff_base_s": 1.0,
        "backoff_max_s": 8.0,
        "rpm": 600,
        "breaker_failures": 3,
        "breaker_cooldown_s": 60,
        "cache": None,
    }
}
MSGS = [
    {"role": "system", "content": "Return JSON."},
    {"role": "user", "content": "hi"},
]


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0
        self.slept: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.slept.append(s)
        self.t += s


def ok_response(content: dict | str, tin: int = 10, tout: int = 5) -> httpx.Response:
    text = content if isinstance(content, str) else json.dumps(content)
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": tin, "completion_tokens": tout},
        },
    )


def make(handler, key: str = "gsk_test_secret", calls: list | None = None, clock=None):
    clock = clock or FakeClock()
    http = httpx.Client(transport=httpx.MockTransport(handler))
    client = GroqClient(
        api_key=key,
        config=CFG,
        http=http,
        cache_path=":memory:",
        on_call=(calls.append if calls is not None else None),
        clock=clock.now,
        sleep=clock.sleep,
    )
    return client, clock


def test_implements_protocol():
    client, _ = make(lambda r: ok_response({"a": 1}))
    assert isinstance(client, LLMClient)


def test_json_mode_and_temperature_zero_sent():
    seen: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(json.loads(req.content))
        assert req.headers["authorization"] == "Bearer gsk_test_secret"
        return ok_response({"label": "VPN"})

    client, _ = make(handler)
    out = client.complete_json(purpose="classify", model="small", messages=MSGS)
    assert out == {"label": "VPN"}
    assert seen[0]["temperature"] == 0
    assert seen[0]["response_format"] == {"type": "json_object"}
    assert seen[0]["model"] == "small"


def test_cache_hit_on_second_identical_call():
    n = {"calls": 0}

    def handler(req):
        n["calls"] += 1
        return ok_response({"x": 1})

    calls: list[dict] = []
    client, _ = make(handler, calls=calls)
    a = client.complete_json(purpose="classify", model="small", messages=MSGS)
    b = client.complete_json(purpose="classify", model="small", messages=MSGS)
    assert a == b == {"x": 1}
    assert n["calls"] == 1
    assert [c["cached"] for c in calls] == [0, 1]
    # different model -> different key -> new call
    client.complete_json(purpose="classify", model="big", messages=MSGS)
    assert n["calls"] == 2


def test_retry_on_429_honours_retry_after():
    seq = [httpx.Response(429, headers={"retry-after": "3"}), ok_response({"ok": True})]

    client, clock = make(lambda r: seq.pop(0))
    assert client.complete_json(purpose="p", model="small", messages=MSGS) == {
        "ok": True
    }
    assert 3.0 in clock.slept


def test_retry_on_5xx_with_backoff_then_success():
    seq = [httpx.Response(503), httpx.Response(502), ok_response({"ok": 1})]
    client, clock = make(lambda r: seq.pop(0))
    assert client.complete_json(purpose="p", model="small", messages=MSGS) == {"ok": 1}
    backoffs = [s for s in clock.slept if s > 0]
    assert len(backoffs) == 2
    assert backoffs[0] <= 1.0 and backoffs[1] <= 2.0


def test_gives_up_after_max_retries():
    n = {"calls": 0}

    def handler(req):
        n["calls"] += 1
        return httpx.Response(500)

    calls: list[dict] = []
    client, _ = make(handler, calls=calls)
    with pytest.raises(LLMUnavailable):
        client.complete_json(purpose="p", model="small", messages=MSGS)
    assert n["calls"] == 3  # 1 + 2 retries
    assert calls[-1]["status"] == "error"


def test_4xx_not_retried():
    n = {"calls": 0}

    def handler(req):
        n["calls"] += 1
        return httpx.Response(401)

    client, _ = make(handler)
    with pytest.raises(LLMUnavailable):
        client.complete_json(purpose="p", model="small", messages=MSGS)
    assert n["calls"] == 1


def test_timeout_is_retryable():
    seq: list = ["timeout", ok_response({"ok": 1})]

    def handler(req):
        item = seq.pop(0)
        if item == "timeout":
            raise httpx.ReadTimeout("slow", request=req)
        return item

    client, _ = make(handler)
    assert client.complete_json(purpose="p", model="small", messages=MSGS) == {"ok": 1}


def test_circuit_breaker_opens_after_three_failures_and_recovers():
    n = {"calls": 0}
    healthy = {"v": False}

    def handler(req):
        n["calls"] += 1
        return ok_response({"ok": 1}) if healthy["v"] else httpx.Response(401)

    client, clock = make(handler)
    for i in range(3):
        with pytest.raises(LLMUnavailable):
            client.complete_json(
                purpose="p",
                model="small",
                messages=[{"role": "user", "content": str(i)}],
            )
    assert not client.available
    before = n["calls"]
    with pytest.raises(LLMUnavailable, match="circuit"):
        client.complete_json(purpose="p", model="small", messages=MSGS)
    assert n["calls"] == before  # skipped Groq entirely
    clock.t += 61
    healthy["v"] = True
    assert client.available
    assert client.complete_json(purpose="p", model="small", messages=MSGS) == {"ok": 1}


def test_offline_mode_without_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    def handler(req):  # pragma: no cover - must never be called
        raise AssertionError("network used in offline mode")

    calls: list[dict] = []
    client, _ = make(handler, key="", calls=calls)
    assert client.offline and not client.available
    with pytest.raises(LLMUnavailable, match="offline"):
        client.complete_json(purpose="p", model="small", messages=MSGS)
    assert calls[0]["status"] == "offline"


def test_offline_from_env(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    client = GroqClient(config=CFG, cache_path=":memory:")
    assert client.offline


def test_bad_json_raises_bad_output_and_is_not_cached():
    n = {"calls": 0}

    def handler(req):
        n["calls"] += 1
        return ok_response("not json at all")

    client, _ = make(handler)
    for _ in range(2):
        with pytest.raises(LLMBadOutput):
            client.complete_json(purpose="p", model="small", messages=MSGS)
    assert n["calls"] == 2
    assert client.available  # bad output does not trip the breaker


def test_on_call_record_fields_and_no_key_leak():
    calls: list[dict] = []
    client, _ = make(lambda r: ok_response({"ok": 1}, 42, 7), calls=calls)
    client.complete_json(
        purpose="classify", model="small", messages=MSGS, analysis_id="A1"
    )
    rec = calls[0]
    for k in (
        "call_id",
        "analysis_id",
        "purpose",
        "model",
        "prompt_hash",
        "tokens_in",
        "tokens_out",
        "latency_ms",
        "status",
        "cached",
        "created_at",
    ):
        assert k in rec
    assert (
        rec["tokens_in"] == 42 and rec["tokens_out"] == 7 and rec["analysis_id"] == "A1"
    )
    assert "gsk_test_secret" not in json.dumps(rec)
    assert "gsk_test_secret" not in repr(client)


def test_on_call_failure_does_not_break_call():
    def boom(_):
        raise RuntimeError("audit down")

    http = httpx.Client(transport=httpx.MockTransport(lambda r: ok_response({"ok": 1})))
    client = GroqClient(
        api_key="k", config=CFG, http=http, cache_path=":memory:", on_call=boom
    )
    assert client.complete_json(purpose="p", model="small", messages=MSGS) == {"ok": 1}


def test_parse_json_object_tolerates_fences():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Here: {"a": 2} done') == {"a": 2}
    with pytest.raises(LLMBadOutput):
        parse_json_object("[1, 2]")


def test_token_bucket_waits_when_empty():
    clock = FakeClock()
    bucket = TokenBucket(
        60, clock=clock.now, sleep=clock.sleep
    )  # 1 token/s, capacity 60
    for _ in range(60):
        assert bucket.acquire() == 0
    waited = bucket.acquire()
    assert waited == pytest.approx(1.0)


def test_circuit_breaker_unit():
    clock = FakeClock()
    cb = CircuitBreaker(3, 60, clock=clock.now)
    cb.failure()
    cb.failure()
    assert not cb.is_open
    cb.success()
    cb.failure()
    cb.failure()
    assert not cb.is_open
    cb.failure()
    assert cb.is_open
