# Make targets owned by P2 (AI engine). Included by the root Makefile.
PY ?= python

.PHONY: test-p2 lint-p2 golden-p2

test-p2:
	$(PY) -m pytest tests/p2_ai -m "not groq and not slow" -q

lint-p2:
	$(PY) -m ruff check ai tests/p2_ai
	$(PY) -m ruff format --check ai tests/p2_ai

golden-p2:
	$(PY) -m pytest tests/p2_ai/test_engine_golden.py -q
