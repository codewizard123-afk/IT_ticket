# Worked examples

Each example follows one ticket end to end: **input → classification → similar cases →
recommendation → evidence**. Each file is rendered from the analysis's audit record
(`GET /audit/{analysis_id}`), the same record the **Audit view** page shows and exports.

| # | File | Path through the decision gate |
|---|------|--------------------------------|
| 1 | [01-recommend-vpn-DEMO-001.md](01-recommend-vpn-DEMO-001.md) | recommend (exact KEDB phrase code, a step already tried) |
| 2 | [02-clarify-then-recommend.md](02-clarify-then-recommend.md) | clarify → one answer → recommend |
| 3 | [03-escalate-oracle-DEMO-002.md](03-escalate-oracle-DEMO-002.md) | escalate (hard rule E4: Critical) to DBA, L3 |
| 4 | [04-multi-issue-DEMO-005.md](04-multi-issue-DEMO-005.md) | multi-issue: VPN recommend + overheating escalate (E6) |

## Status: regenerate against the real API

The current files come from the **mock API** (`MOCK_API=1`). It replays the Phase 0 golden fixtures
for DEMO-001/002/005 and uses a keyword-overlap stand-in for example 2, so **scores are
illustrative** and example 2's wording comes from the stand-in. Once the real API is up (milestone
M3/M4), regenerate them so they match real output:

```bash
make api                      # terminal 1
make examples                 # terminal 2 (MOCK_API=0): rewrites docs/examples/*.md
```

You can also download any single analysis as a worked example from **Audit view → Record →
Worked example (Markdown)**.
