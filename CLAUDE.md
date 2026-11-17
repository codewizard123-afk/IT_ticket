# CLAUDE.md — guardrails for every Claude Code session in this repo

Read `README.md` (team split, workflow) and `docs/architecture.md` (low-level design) before writing code.
The user will tell you which person you are working for (P1, P2, P3 or P4) and which task IDs to do.

## Hard rules
1. **Ownership.** Edit only the folders and config files owned by your person (README §4). If you need a change elsewhere, stop and say so.
2. **`shared/` is frozen.** Never edit `shared/`, `config/base.yaml` or `docs/architecture.md` in a feature branch. These change only through a `contract-change` PR (README §8.3).
3. **Output contract.** `POST /api/v1/analyze` must always validate against `shared/contract/analysis_output.schema.json`. The 8 contract keys never change, and contract objects never get extra keys. Extensions go in `details`.
4. **Evidence.** Every recommended step must cite a KB step (`KB-xxxx#res.n` / `#diag.n`) or a *verified* incident. Never cite `synthetic_kb_derived` resolutions.
5. **Privacy.** Call `privacy.redact()` before any Groq call or embedding of ticket text. Never log raw ticket text, secrets or `GROQ_API_KEY`.
6. **Offline-safe.** Every LLM step needs a deterministic fallback. Tests must pass with `GROQ_API_KEY` unset.
7. **No label leakage.** `metadata.expected_*` must never reach the AI engine.
8. **Safety.** Never generate steps that disable security controls, share passwords or grant broad privileges. The system recommends only; it never executes.
9. **Dependencies.** Add packages only to your own `requirements/<area>.txt` and Make targets only to your own `mk/p<n>.mk`.
10. **Never commit** `.env`, `data/chroma/`, `data/app.db`, `data/llm_cache.sqlite`, `eval/out/` or `trial/results/`.

## Workflow per task
1. Restate the task's "Done when" criteria from README §7.
2. `git fetch origin && git checkout -b p<n>/<task-id>-<slug> origin/develop` (or rebase if the branch already exists).
3. Write tests first in `tests/p<n>_*/`, then implement.
4. Run `make lint test-p<n> test-contract`. Everything must be green.
5. Commit with Conventional Commits and the task ID: `feat(p2): ... (P2-T6)`.
6. `git push -u origin HEAD`, then `gh pr create --base develop --fill`. Fill in the PR template, including the status line (done / next / blocked-on).

## Code style
Python 3.11, type hints everywhere, Pydantic v2 models from `shared/schemas.py`, `ruff` for lint and format.
Use pure functions with injected dependencies (stores, LLM client), and no global state except the cached embedder singleton.
Config values come from `config/*.yaml`; don't hard-code thresholds.
