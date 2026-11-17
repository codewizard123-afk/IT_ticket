# P1 · Data & KB targets (owner: P1). Add new P1 targets here only.
.PHONY: data data-eval seed test-sets eval mttr-report test-p1

data: ## enrich history + build Chroma, BM25 and error-code indexes
	$(PY) -m ingest.build_all

data-eval: ## build the *_eval collections from the 70% split
	$(PY) -m ingest.build_all --eval-split

seed: ## regenerate data/seed/resolved_incidents.json from curated_incidents.yaml
	$(PY) -m ingest.make_seed

test-sets: ## regenerate data/test/*.json from the seed files
	$(PY) -m ingest.make_test_sets

eval: ## evaluation on the test split -> eval/out/
	$(PY) -m eval.run --split test

mttr-report: ## MTTR trial report -> eval/out/mttr.json + charts
	$(PY) -m eval.mttr_report

test-p1: ## P1 tests
	$(call run_pytest,tests/p1_data)
