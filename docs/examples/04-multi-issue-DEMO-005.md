# Worked example 4: multi-issue (DEMO-005, VPN drops + overheating)

> The splitter separates two issues: the VPN issue gets a recommendation, while the overheating issue hits a KB escalation clause (E6) and escalates to EUC-HW.
>
> Source: **mock API** (`MOCK_API=1`, Phase 0 golden fixtures; scores are illustrative). Regenerate with `make examples` against the real API. Analysis `mock-00005`, rendered for role `l2`. Audit hash chain: verified.

## 1. Input

**DEMO-005 · VPN keeps dropping and laptop overheating**

```text
My VPN disconnects every few minutes and I am losing unsaved work. Also the laptop gets very hot and the fan is loud even when I only have Outlook open.
```

## 2. Analysis

### Issue 1 (primary)

**Classification**

- Network / **VPN**, confidence 0.86 (band high)
- Components: llm 0.88, knn 0.8, rule 1, margin 0.55, agree 3
- Rationale: two issues: VPN drops (primary) and laptop overheating (secondary)
- Alternatives: Hardware/CPU (0.38)

**Priority**

- **High** (P2): losing unsaved work; floor: VPN baseline
- Driver: losing unsaved work → U2
- Driver: floor: VPN baseline P2

**KEDB match**

- **KB-0018**: VPN will not connect or repeatedly disconnects (score 0.79)
- Matched field: symptom; why: symptom match + VPN mapping
- Excerpt: “VPN client cannot connect”
- Alternatives: KB-0019 (0.62)

**Recommended solution** (source: KEDB + Historical Incident)

1. Confirm the device has working internet access without the VPN `[KB-0018·diag.1]` · L1 · diagnostic
2. Check endpoint CPU and memory usage while the VPN is connected `[KB-0019·diag.1]` · L1 · diagnostic
3. Restart the VPN client and reconnect `[KB-0018·res.1]` · L1 · resolution
4. Clear the saved VPN credentials and sign in again `[KB-0018·res.3]` `[INC-H0002]` · L1 · resolution

### Issue 2 (secondary, decision escalate)

**Classification**

- Hardware / **CPU**, confidence 0.82 (band high)
- Components: llm 0.85, knn 0.78, rule 1, margin 0.6, agree 3
- Rationale: laptop overheating and fan noise
- Alternatives: Hardware/Memory (0.12)

**Priority**

- **Medium** (P3): Baseline for CPU

**KEDB match**

- **KB-0030**: Application slow or high CPU/memory usage (score 0.71)
- Matched field: symptom; why: symptom match + CPU mapping
- Excerpt: “Application is very slow”
- Alternatives: KB-0006 (0.66)

**Recommended solution** (source: KEDB)

1. Check Task Manager for the process using the most CPU and memory `[KB-0030·diag.1]` · L1 · diagnostic
2. Check that the air vents are not blocked and the laptop is on a hard surface `[KB-0006·diag.1]` · L1 · diagnostic
3. Check the system event log for thermal or Kernel-Power 41 events `[KB-0006·diag.3]` · L1 · diagnostic
4. Close the runaway process and restart the application `[KB-0030·res.1]` · L1 · resolution

## 3. Similar cases

| Incident | Score | Tier | Origin | Why | Resolution |
|---|---|---|---|---|---|
| INC-H0002 | 0.87 | SIMILAR | VERIFIED | same subcategory; shared terms: VPN, disconnects | Cleared saved VPN credentials and re-authenticated; no further drops. |
| TKT-0002 | 0.76 | SIMILAR | SYNTHETIC | same subcategory; shared terms: VPN, connect | Resolved via KB-0018: Restart the VPN client and reconnect |

SYNTHETIC resolutions are shown for context only and are never cited as evidence.

## 4. Decision

**ESCALATE**
- Escalate issue 2 to **EUC-HW** (tier L2). Reasons: E6: KB-0006: the laptop overheats
- `escalation_required`: **true**
- `human_approval_required`: **true** (decision escalate; priority High)

## 5. Evidence trail (audit)

| Kind | Ref | Rank | Score | Tier | Why |
|---|---|---|---|---|---|
| similar | INC-H0002 | 1 | 0.87 | SIMILAR | same subcategory; shared terms: VPN, disconnects |
| similar | TKT-0002 | 2 | 0.76 | SIMILAR | same subcategory; shared terms: VPN, connect |
| kedb | KB-0018 | 1 | 0.79 | - | symptom match + VPN mapping |

**Events**

- `2026-11-17T09:45:20Z` ticket.submitted (assisted)
- `2026-11-17T09:45:20Z` analysis.completed (assisted)

## Contract output

```json
{
  "ticket_id": "DEMO-005",
  "classification": {
    "category": "Network",
    "subcategory": "VPN",
    "confidence": 0.86
  },
  "priority": {
    "level": "High",
    "reason": "losing unsaved work; floor: VPN baseline"
  },
  "similar_incidents": [
    {
      "incident_id": "INC-H0002",
      "similarity_score": 0.87,
      "resolution": "Cleared saved VPN credentials and re-authenticated; no further drops."
    },
    {
      "incident_id": "TKT-0002",
      "similarity_score": 0.76,
      "resolution": "Resolved via KB-0018: Restart the VPN client and reconnect"
    }
  ],
  "kedb_match": {
    "found": true,
    "error_code": "KB-0018",
    "solution": "Restart the VPN client and reconnect"
  },
  "recommended_solution": {
    "steps": [
      "Confirm the device has working internet access without the VPN",
      "Check endpoint CPU and memory usage while the VPN is connected",
      "Restart the VPN client and reconnect",
      "Clear the saved VPN credentials and sign in again"
    ],
    "source": "KEDB + Historical Incident"
  },
  "escalation_required": true,
  "human_approval_required": true
}
```
