# Worked example 3: escalate (DEMO-002, Oracle month-end slowdown)

> 15 users and a blocked month-end close make it Critical, a hard rule (E4), so the ticket goes to the DBA team at L3 with a diagnostic bundle.
>
> Source: **mock API** (`MOCK_API=1`, Phase 0 golden fixtures; scores are illustrative). Regenerate with `make examples` against the real API. Analysis `mock-00004`, rendered for role `l2`. Audit hash chain: verified.

## 1. Input

**DEMO-002 · Oracle finance database extremely slow, ORA-01013**

```text
The finance reporting queries on the Oracle database are timing out with ORA-01013: user requested cancel of current operation. About 15 users in finance are affected and the month-end close is blocked.
```

## 2. Analysis

### Issue 1 (primary)

**Classification**

- Database / **Oracle**, confidence 0.95 (band high)
- Components: llm 0.96, knn 0.9, rule 1, margin 0.88, agree 3
- Rationale: ORA-01013 is an Oracle error code; slow month-end queries
- Alternatives: Database/DB2 (0.03)

**Priority**

- **Critical** (P1): 15 users affected; month-end close blocked
- Driver: 15 users affected → I1
- Driver: month-end close blocked → U1

**KEDB match**

- **KB-0001**: Database queries slow or timing out (score 0.88)
- Matched code in the ticket: `ORA-01013`
- Matched field: errors; why: exact error code + slow/timeout symptom
- Excerpt: “ORA-01013”
- Alternatives: KB-0002 (0.47)

**Recommended solution** (source: KEDB + Historical Incident)

1. Check the database for long-running or blocked sessions during the slow period `[KB-0001·diag.1]` · L2 · diagnostic
2. Review the execution plan of the slowest query for full table scans `[KB-0001·diag.2]` · L2 · diagnostic
3. Ask the DBA to end the blocking session after confirming with its owner `[KB-0001·res.1]` `[INC-H0006]` · L2 · resolution
4. Ask the DBA to gather fresh optimiser statistics on the affected tables `[KB-0001·res.2]` · L2 · resolution

## 3. Similar cases

| Incident | Score | Tier | Origin | Why | Resolution |
|---|---|---|---|---|---|
| INC-H0006 | 0.86 | SIMILAR | VERIFIED | same subcategory; shared terms: Oracle, slow, month-end | DBA ended a blocking batch session and refreshed table statistics. |
| TKT-0011 | 0.83 | SIMILAR | SYNTHETIC | same subcategory; error code ORA-01013 | Resolved via KB-0001: Ask the DBA to end the blocking session after confirming with its owner |
| TKT-0012 | 0.74 | RELATED | SYNTHETIC | same subcategory; shared terms: Oracle, slow | Resolved via KB-0001: Ask the DBA to end the blocking session after confirming with its owner |

SYNTHETIC resolutions are shown for context only and are never cited as evidence.

## 4. Decision

**ESCALATE**
- Escalate issue 1 to **DBA** (tier L3). Reasons: E4: ticket priority P1 (Critical); E6: KB-0001: a month-end or financial close process is blocked
- `escalation_required`: **true**
- `human_approval_required`: **true** (decision escalate; priority Critical; step role L2; KB-0001 requires admin)

## 5. Evidence trail (audit)

| Kind | Ref | Rank | Score | Tier | Why |
|---|---|---|---|---|---|
| similar | INC-H0006 | 1 | 0.86 | SIMILAR | same subcategory; shared terms: Oracle, slow, month-end |
| similar | TKT-0011 | 2 | 0.83 | SIMILAR | same subcategory; error code ORA-01013 |
| similar | TKT-0012 | 3 | 0.74 | RELATED | same subcategory; shared terms: Oracle, slow |
| kedb | KB-0001 | 1 | 0.88 | - | exact error code + slow/timeout symptom |

**Events**

- `2026-11-17T09:45:20Z` ticket.submitted (assisted)
- `2026-11-17T09:45:20Z` analysis.completed (assisted)

## Contract output

```json
{
  "ticket_id": "DEMO-002",
  "classification": {
    "category": "Database",
    "subcategory": "Oracle",
    "confidence": 0.95
  },
  "priority": {
    "level": "Critical",
    "reason": "15 users affected; month-end close blocked"
  },
  "similar_incidents": [
    {
      "incident_id": "INC-H0006",
      "similarity_score": 0.86,
      "resolution": "DBA ended a blocking batch session and refreshed table statistics."
    },
    {
      "incident_id": "TKT-0011",
      "similarity_score": 0.83,
      "resolution": "Resolved via KB-0001: Ask the DBA to end the blocking session after confirming with its owner"
    },
    {
      "incident_id": "TKT-0012",
      "similarity_score": 0.74,
      "resolution": "Resolved via KB-0001: Ask the DBA to end the blocking session after confirming with its owner"
    }
  ],
  "kedb_match": {
    "found": true,
    "error_code": "KB-0001",
    "solution": "Ask the DBA to end the blocking session after confirming with its owner"
  },
  "recommended_solution": {
    "steps": [
      "Check the database for long-running or blocked sessions during the slow period",
      "Review the execution plan of the slowest query for full table scans",
      "Ask the DBA to end the blocking session after confirming with its owner",
      "Ask the DBA to gather fresh optimiser statistics on the affected tables"
    ],
    "source": "KEDB + Historical Incident"
  },
  "escalation_required": true,
  "human_approval_required": true
}
```
