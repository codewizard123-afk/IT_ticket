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

.PHONY: trial-schedule trial-export
PARTICIPANTS ?= p1 p2 p3 p4
trial-schedule: ## seeded crossover schedule -> trial/results/schedule.csv (PARTICIPANTS="a b c d")
	$(PY) -m trial.schedule $(PARTICIPANTS) --csv trial/results/schedule.csv

trial-export: ## trial events -> trial/results/raw_events.csv (TRIAL_USER=admin TRIAL_PASSWORD=...)
	$(PY) -m trial.export $(if $(TRIAL_USER),--username $(TRIAL_USER) --password "$(TRIAL_PASSWORD)",--demo-role admin)

.PHONY: examples
examples: ## regenerate docs/examples/ from the API (MOCK_API=1 for golden fixtures)
	$(PY) -m ui.examples --out docs/examples
