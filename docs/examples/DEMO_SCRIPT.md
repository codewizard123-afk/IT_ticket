# Demo script (about 8 minutes)

The live demo and the recording follow this script. Each step names its page and the requirement it
shows.

## Before you start

```bash
make api                         # terminal 1 (or: MOCK_API=1 for a backend-free rehearsal)
make ui                          # terminal 2 → http://localhost:8501
make demo                        # terminal 3: all 7 demos, schema check, §16 ✓ column
```

- Reset the database for a clean audit trail: stop the API, delete `data/app.db`, start it again.
- Open two browser windows: one for **l2_bob** (NET-OPS) and one private window for **l1_alice**.
- Set the screen recorder to 1080p and turn off notifications.

## Talk track

| # | Time | Page · action | What to say | Shows |
|---|------|---------------|-------------|-------|
| 1 | 0:00 | Terminal: `make demo` | "Seven demo tickets through the real API. Every response validates against the 8-key contract; the last column checks the expected behaviour." | contract |
| 2 | 0:40 | **Login** → l2_bob | "Role-based access: the role decides what you see." | access control |
| 3 | 1:00 | **Analyze** → load DEMO-001 → Analyze | "Network/VPN with high confidence; you can see the components. The KEDB match is KB-0018 through the exact phrase 'Gateway not responding'. Every step cites its source, and the step the user already tried is greyed out. Priority High means an analyst approves first." | FR-01, 03, 04 |
| 4 | 2:00 | Similar incidents card | "Each similar incident shows its tier, its score and why it matched. Synthetic resolutions are badged and never cited as evidence." | FR-02, data rules |
| 5 | 2:40 | 👍 on a step, then **Submit decision** (Resolved) | "Analyst feedback and the final decision are recorded with timings." | FR-06, FR-09 |
| 6 | 3:10 | **Analyze** → DEMO-002 | "Oracle, ORA-01013, 15 users, month-end: Critical, which is a hard escalation rule. It goes to DBA at L3, with a diagnostic bundle to download." | FR-05 |
| 7 | 3:50 | l1_alice window → DEMO-002 | "The same ticket for an L1 analyst: L2 steps become 'Handled by DBA'." | RBAC |
| 8 | 4:20 | **Analyze** → type "VPN broken" | "Too vague: the assistant asks one round of questions. I answer once and it re-analyses; it never loops." | FR-05 clarify |
| 9 | 5:00 | **Analyze** → DEMO-005, Issue 2 tab | "Two problems in one ticket. VPN gets steps; the overheating laptop escalates to EUC-HW." | multi-issue |
| 10 | 5:40 | **Search & filter**: Network + VPN + High | "FR-07 filters combine; restricted tickets never appear for L1." | FR-07 |
| 11 | 6:20 | **Audit view** → the DEMO-001 analysis | "The whole trail: input, evidence, decision and feedback. The hash chain verifies, and the export is masked for non-admins." | FR-08 |
| 12 | 7:00 | **Manual mode**, then **MTTR dashboard** | "The baseline for our timed trial has no AI at all. The dashboard compares time-to-decision, manual against assisted. Historical MTTR is synthetic and labelled that way." | FR-09 |
| 13 | 7:40 | Close | "It recommends, asks or escalates, always with evidence, and an analyst always decides." | |

## Backup plan

- If the API or Groq is down, use `MOCK_API=1 make ui` (golden fixtures; the header shows
  "MOCK API"), or point at the "degraded" banner, which shows the offline fallback working.
- Keep `docs/examples/*.md` open in a tab as a static fallback.
