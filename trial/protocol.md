# Timed MTTR trial protocol (FR-09, team-defined)

Owner: P4 (runs the trial) · Analysis: P1 (`eval/mttr_report.py`) · Design: architecture §9.3

The trial asks one question: **does the assistant shorten the time from opening a ticket to a
correct decision?** It compares *manual* handling (keyword search only) with *assisted* handling
(the Analyze page) on the same 12 tickets, using a crossover design so every participant does both.

## 1. Participants

- **At least 4.** All teammates, plus volunteers if possible (volunteers are better, because they
  did not build the system).
- Each participant needs **their own account** so events can be told apart. With the real API, ask
  P3 to add trial accounts to `config/users.yaml` (for example `trial_p1` … `trial_p6`, role `l1`
  or `l2`). Do not use `DEMO_AUTH` role headers during the trial: every demo-role user shows up as
  the same actor.
- Record a pseudonym per participant. Do not record names in `trial/results/`.

## 2. Materials

| Item | Where |
|------|-------|
| 12 trial tickets T01–T12 (2 per main category + 2 gap/escalation cases) | `GET /trial/tickets` (served from `data/test/trial_tickets.json`) |
| Gold answers | `data/test/trial_gold.json`. **Never shown in the UI.** Only P1's report reads it. |
| Practice ticket | `DEMO-001` on the Analyze page |
| Schedule | `make trial-schedule PARTICIPANTS="p1 p2 p3 p4"` → `trial/results/schedule.csv` |

## 3. Design: seeded crossover

| Group | T01–T06 | T07–T12 |
|-------|---------|---------|
| A | manual | assisted |
| B | assisted | manual |

- Participants are split A/B alternately after a seeded shuffle (`config/ui.yaml → trial.seed`, default 42).
- The order of tickets inside each block is shuffled per participant with the same seed.
- Everyone first does the practice ticket in assisted mode. It is flagged `practice=True` and left
  out of the analysis.
- **Integrity rule:** a participant who built a component should skip trial tickets in that
  component's category if the schedule allows it. Note every skip in the session log; it is a
  limitation in the report.

## 4. Running a session (about 40 minutes per participant)

1. Start the stack: `make api` and `make ui` (or `MOCK_API=1 make ui` for a dry run only).
2. The participant logs in with their trial account on the **Login** page.
3. Briefing (read aloud): *"Handle each ticket as you would at the service desk. Choose a final
   decision, the category, subcategory and priority, the KB article or incident you used, and how
   confident you are (1–5). Work at your normal pace; you are not being graded."*
4. Practice ticket `DEMO-001` on **Analyze**. Answer questions about the UI now, not later.
5. Follow the schedule row by row:
   - **Manual rows** → **Manual mode** page → choose the ticket → **Start ticket** (this posts
     `manual.started` and reveals the text) → search → submit the decision (this posts
     `manual.decision`).
   - **Assisted rows** → **Analyze** page → *Load sample* → the trial ticket → **Analyze** (the
     result posts `ui.viewed`) → review the evidence → submit the decision (`decision.submitted`).
     If the assistant asks a clarifying question, the participant answers it from the ticket text
     or types "not stated".
6. The facilitator does not help with tickets. Note any interruption in the session log with the
   ticket ID; P1 can exclude that ticket.

## 5. Measures

| Measure | Definition | Source |
|---------|------------|--------|
| TTD (time to decision) | assisted: `decision.submitted − ui.viewed`; manual: `manual.decision − manual.started` | events |
| Correctness | category ✓, subcategory ✓, priority within ±1 level, and the correct KB article *or* a correct escalate decision | decision vs `trial_gold.json` |
| Self-reported confidence | 1–5, last line of the decision notes (`Self-reported confidence: n/5`) | decision |

## 6. Export for P1

After the last session, as an admin (the export is masked for everyone else):

```bash
make trial-export TRIAL_USER=admin TRIAL_PASSWORD=...     # -> trial/results/raw_events.csv
```

`raw_events.csv` has one row per event (`participant, group, seq, practice, ticket_id,
analysis_id, mode, event, ts_utc`), and the decision columns (`final_*`, `kb_or_incident_used`,
`confidence`) are filled on the decision event rows. P1's `make mttr-report` turns it into median TTD
(manual vs assisted), % change, correctness both ways, n and caveats for
`docs/evaluation_report.md`.

`trial/results/` is gitignored. Hand the CSV to P1 directly; never commit it.

## 7. Caveats to report

- Small n (≥ 4 × 12 decisions), so the result is **indicative**, not significant.
- Participants are partly the builders, so expect learning and expectation effects. The crossover
  balances order, but it cannot remove bias.
- The trial tickets are team-written (`metadata.source = "synthetic_team"`).
- Historical MTTR on the dashboard comes from **synthetic** timestamps and is context only.

## 8. Session log template

| Participant | Group | Date | Facilitator | Skipped tickets (reason) | Interruptions | Notes |
|-------------|-------|------|-------------|--------------------------|---------------|-------|
| | | | | | | |
