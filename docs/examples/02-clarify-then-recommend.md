# Worked example 2: clarify → recommend (vague VPN ticket)

> A two-word ticket is too sparse to act on, so the assistant asks one round of questions; the answer brings a KEDB match and a cited recommendation.
>
> Source: **mock API** (`MOCK_API=1`, Phase 0 golden fixtures; scores are illustrative). Regenerate with `make examples` against the real API. Analysis `mock-00003`, rendered for role `l2`. Audit hash chain: verified.

## 1. Input

**INC-20261117-0002 · VPN broken**

```text
VPN broken
[Clarification] VPN client says gateway not responding when working from home
```

## 2. First pass: the assistant asks

Decision **CLARIFY** (analysis `mock-00002`, band medium). Questions:

- What is the exact error message?
- How many users are affected?
- When did it start, and did anything change?

The analyst's answer is appended to the description as `[Clarification] …` and the ticket is re-analysed once (a second low-confidence result would escalate).

## 3. Analysis

### Issue 1 (primary)

**Classification**

- Network / **VPN**, confidence 0.69 (band medium)
- Components: knn 0.78, llm 0, rule 0
- Rationale: mock API: keyword overlap with historical tickets

**Priority**

- **High** (P2): Baseline for VPN

**KEDB match**

- **KB-0018**: VPN will not connect or repeatedly disconnects (score 0.57)
- Matched field: diagnostic; why: keyword overlap (mock API)
- Excerpt: “Check the VPN client log for 'Gateway not responding' and note the gateway name”
- Alternatives: KB-0019 (0.29), KB-0012 (0.14)

**Recommended solution** (source: KEDB)

1. Confirm the device has working internet access without the VPN `[KB-0018·diag.1]` · L1 · diagnostic
2. Check the VPN client log for 'Gateway not responding' and note the gateway name `[KB-0018·diag.2]` · L1 · diagnostic
3. Restart the VPN client and reconnect `[KB-0018·res.1]` · L1 · resolution
4. Connect to the alternate gateway listed in the VPN client profile `[KB-0018·res.2]` · L1 · resolution

## 4. Similar cases

| Incident | Score | Tier | Origin | Why | Resolution |
|---|---|---|---|---|---|
| TKT-0001 | 0.78 | LIKELY DUPLICATE | SYNTHETIC | keyword overlap (mock API) | Resolved via KB-0018: Restart the VPN client and reconnect |
| INC-H0001 | 0.33 | RELATED | VERIFIED | keyword overlap (mock API) | Switched the user to the alternate regional gateway in the VPN profile; connection stable. |
| INC-H0002 | 0.11 | RELATED | VERIFIED | keyword overlap (mock API) | Cleared saved VPN credentials and re-authenticated; no further drops. |
| TKT-0002 | 0.11 | RELATED | SYNTHETIC | keyword overlap (mock API) | Resolved via KB-0018: Restart the VPN client and reconnect |

SYNTHETIC resolutions are shown for context only and are never cited as evidence.

## 5. Decision

**RECOMMEND**
- `escalation_required`: **false**
- `human_approval_required`: **true** (priority High; confidence medium)

## 6. Evidence trail (audit)

| Kind | Ref | Rank | Score | Tier | Why |
|---|---|---|---|---|---|
| similar | TKT-0001 | 1 | 0.78 | LIKELY DUPLICATE | keyword overlap (mock API) |
| similar | INC-H0001 | 2 | 0.33 | RELATED | keyword overlap (mock API) |
| similar | INC-H0002 | 3 | 0.11 | RELATED | keyword overlap (mock API) |
| similar | TKT-0002 | 4 | 0.11 | RELATED | keyword overlap (mock API) |
| kedb | KB-0018 | 1 | 0.57 | - | keyword overlap (mock API) |

**Events**

- `2026-11-17T09:45:20Z` analysis.completed (assisted)

## Contract output

```json
{
  "ticket_id": "INC-20261117-0002",
  "classification": {
    "category": "Network",
    "subcategory": "VPN",
    "confidence": 0.69
  },
  "priority": {
    "level": "High",
    "reason": "Baseline for VPN"
  },
  "similar_incidents": [
    {
      "incident_id": "TKT-0001",
      "similarity_score": 0.78,
      "resolution": "Resolved via KB-0018: Restart the VPN client and reconnect"
    },
    {
      "incident_id": "INC-H0001",
      "similarity_score": 0.33,
      "resolution": "Switched the user to the alternate regional gateway in the VPN profile; connection stable."
    },
    {
      "incident_id": "INC-H0002",
      "similarity_score": 0.11,
      "resolution": "Cleared saved VPN credentials and re-authenticated; no further drops."
    },
    {
      "incident_id": "TKT-0002",
      "similarity_score": 0.11,
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
      "Check the VPN client log for 'Gateway not responding' and note the gateway name",
      "Restart the VPN client and reconnect",
      "Connect to the alternate gateway listed in the VPN client profile"
    ],
    "source": "KEDB"
  },
  "escalation_required": false,
  "human_approval_required": true
}
```
