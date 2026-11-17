# P2 · AI engine targets (owner: P2). Add new P2 targets here only.
.PHONY: test-p2

test-p2: ## P2 tests
	$(call run_pytest,tests/p2_ai)
