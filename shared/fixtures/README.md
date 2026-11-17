# shared/fixtures

Phase 0 test data (owner: ALL, frozen; change via a `contract-change` PR).

| File | What it is |
|------|------------|
| `golden/DEMO-001…007.json` | Full-view (`view=full`) responses that encode architecture §16. Mock-API data for P4 and acceptance targets for P2/P3. Golden tests assert `category`, `subcategory`, `kedb_match.error_code` (KB ID), `details.decision` and the two booleans; scores are illustrative only. |
| `mini_articles.json` | 30 KEDB articles (KB-0001…KB-0030) used by `FakeKBStore`. |
| `mini_tickets.json` | 50 tickets used by `FakeTicketStore`: 36 `tickets_hist` (2 per subcategory, `synthetic_*` resolutions) + 14 `tickets_verified` (`INC-H0001…`, `curated_verified`). |

**These are stub placeholders, not real data.** The real `data/kb/articles.json` and
`data/samples/sample_tickets.json` were not in the repo at Phase 0, so the article titles, steps and
error codes here were written from architecture §3.3, §7 and §16. When P1 lands the real KEDB:

1. Replace `mini_articles.json` with the real 30 articles (governance fields added).
2. Re-check that every `details.steps[].citations` entry in `golden/` still resolves
   (`tests/contract/test_golden_fixtures.py` enforces this) and update step texts to match.

Every record carries `"source": "stub_fixture"` or `"fixture": "golden-phase0"` so it can never be
mistaken for real data.
