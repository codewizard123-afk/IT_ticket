# ai/ — AI engine (owner: P2)

Architecture §7 and §12. Everything here is pure over its inputs: stores and the LLM client are
injected through `EngineContext`, so it runs on `shared/stubs.py` and works with `GROQ_API_KEY`
unset (every LLM step has a deterministic fallback).

## Wiring (for P3's `backend/deps.py`)

```python
from ai.engine import AIEngine                  # implements shared.interfaces.Engine
from ai.llm.groq_client import make_llm_client  # implements shared.interfaces.LLMClient
from ai.ticket_search import search_tickets     # FR-07: /tickets/search
from ai.kb_search import search_kb              # FR-07: /kb/search

llm = make_llm_client(on_call=audit_llm_call)   # on_call gets {purpose, model, cache_hit, status,
                                                #   tokens, latency_ms, attempts, error}; no prompt text
engine = AIEngine()
result = engine.run(sanitized_ticket, EngineContext(role=..., ticket_store=..., kb_store=...,
                                                    llm=llm, feedback_prior=..., filters=...))
```

## Pipeline (`ai/engine.py`)

| Stage | Module | Notes |
|-------|--------|-------|
| Signals | `signals.py` + `config/signals.yaml` | error codes (regex + KEDB phrase codes), users/scope, urgency, security, tried steps… |
| Classify (Groq #1) | `classify.py` + `config/rules.yaml` + `prompts/classify*.j2` | kNN + LLM + rules fusion, bands, conflict, `llm_classify_fallback` |
| Split | `split.py` | LLM split or `(1)/(2)` / numbered / product-change fallback, max 2 issues |
| Similar tickets | `ticket_search.py` | dense + BM25 + RRF, bonuses, keyword-trap guard, tiers, FR-07 filters, live clustering |
| KEDB | `kb_search.py` | `0.45 sem + 0.15 bm25 + 0.20 code + 0.12 tax + 0.08 fb`, τ_show / τ_kb, ambiguity |
| Priority | `priority.py` | impact × urgency matrix, baseline and KB-hint floors |
| Synthesise (Groq #2) | `synthesize.py` + `prompts/synthesize*.j2` | evidence pack (verified incidents only), cited steps, template fallback |
| Validate | `validate.py` | citation existence, ROUGE-L grounding ≥ 0.30, safety lint, source rule |

`IssueResult` extras the engine adds (open model): `pattern` (subcategory count + top terms),
`signals` (per-issue signals for split tickets). `Recommendation` extras: `safety_violations`,
`cross_link`. `Step` extras: `grounding`, `cross_link`.

## Tuning

All thresholds and weights live in `config/ai.yaml` (code defaults in `ai/config.py`). P1's
evaluation (P1-T7) should update `classifier.weights`, `classifier.bands`, `knn_power`,
`knn_min_rel` and `retrieval.weights_kb` there. The taxonomy map is read from P1's
`config/taxonomy_map.yaml` when it exists; `taxonomy_map_fallback` in `ai.yaml` is used otherwise.

## Tests

`make test-p2` (or `python -m pytest tests/p2_ai`). The demo texts in
`tests/p2_ai/fixtures/demo_tickets.json` are P2 reconstructions from architecture §16
(`source: synthetic_team`) until P1 lands `data/samples/sample_tickets.json`.
