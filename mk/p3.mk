# P3 · Backend targets (owner: P3). Add new P3 targets here only.
.PHONY: api api-stub test-p3

api: ## FastAPI on http://localhost:8000 (OpenAPI docs at /docs); needs `make data` for real stores
	$(PY) -m uvicorn backend.main:app --reload --port 8000

api-stub: export USE_STUB_STORES = 1
api-stub: export USE_STUB_ENGINE = 1
api-stub: export USE_STUB_REDACT = 1
api-stub: export DEMO_AUTH = 1
api-stub: ## FastAPI on stubs (FakeEngine + golden fixtures, fake stores, demo auth); no `make data`
	$(PY) -m uvicorn backend.main:app --reload --port 8000

test-p3: ## P3 tests
	$(call run_pytest,tests/p3_backend)
