# Root Makefile (owner: ALL). Owner-specific targets live in mk/p1.mk … mk/p4.mk.
# Activate your venv first, or override the interpreter: make test PY=.venv/Scripts/python
# Windows without GNU make: use `mingw32-make` or `choco install make`.

PY      ?= python
PYTEST  ?= $(PY) -m pytest
RUFF    ?= $(PY) -m ruff
# CI and `make test` never need Groq or a built index.
PYTEST_MARKS ?= -m "not groq and not slow"

# Run pytest on a folder; "no tests collected" (exit 5) counts as green so empty owner
# folders don't break the build before their first test lands.
define run_pytest
	$(PYTEST) $(PYTEST_MARKS) $(1) || [ $$? -eq 5 ]
endef

.DEFAULT_GOAL := help
.PHONY: help setup lint fmt test test-contract test-int

help: ## list targets
	@grep -hE '^[a-zA-Z0-9_-]+:.*## ' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*## "}; {printf "  %-14s %s\n", $$1, $$2}'

setup: ## install all deps + dev tools + pre-commit hooks
	$(PY) -m pip install -r requirements.txt -r requirements/dev.txt
	-$(PY) -m pre_commit install

lint: ## ruff check + format check
	$(RUFF) check .
	$(RUFF) format --check .

fmt: ## ruff format + autofix
	$(RUFF) format .
	$(RUFF) check --fix .

test: ## all tests except groq/slow
	$(call run_pytest,tests)

test-contract: ## validate fixtures/outputs against the JSON Schema
	$(call run_pytest,tests/contract)

test-int: ## end-to-end golden tests (needs `make data`)
	$(PYTEST) -m "not groq" tests/integration || [ $$? -eq 5 ]

include mk/p1.mk mk/p2.mk mk/p3.mk mk/p4.mk
