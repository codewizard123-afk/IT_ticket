# P3 · Backend targets (owner: P3). Add new P3 targets here only.
.PHONY: api test-p3

api: ## FastAPI on http://localhost:8000 (OpenAPI docs at /docs)
	$(PY) -m uvicorn backend.main:app --reload --port 8000

test-p3: ## P3 tests
	$(call run_pytest,tests/p3_backend)
