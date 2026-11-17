# P4 · UI and delivery targets (owner: P4). Add new P4 targets here only.
.PHONY: ui mock-api demo test-p4

ui: ## Streamlit on http://localhost:8501
	$(PY) -m streamlit run ui/app.py

mock-api: ## serve golden fixtures on port 8000 (UI work without a backend)
	$(PY) -m ui.mock_api

demo: ## analyse DEMO-001…007 through the API and print contract JSON
	$(PY) -m ui.demo

test-p4: ## P4 tests
	$(call run_pytest,tests/p4_ui)
