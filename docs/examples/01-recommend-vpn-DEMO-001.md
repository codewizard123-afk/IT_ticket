# Worked example 1: recommend (DEMO-001, VPN gateway not responding)

> An exact KEDB phrase code gives a confident match; the step the user already tried is greyed out; High priority means an analyst approves before applying.
>
> Source: **mock API** (`MOCK_API=1`, Phase 0 golden fixtures; scores are illustrative). Regenerate with `make examples` against the real API. Analysis `mock-00001`, rendered for role `l2`. Audit hash chain: verified.

## 1. Input

**DEMO-001 · VPN not connecting - Gateway not responding**

```text
Since this morning I cannot connect to the VPN from home. The client shows 'Gateway not responding'. I already restarted the VPN client twice. I have a deadline today and need access to the file shares.
```

## 2. Analysis

### Issue 1 (primary)

**Classification**

- Network / **VPN**, confidence 0.93 (band high)
- Components: llm 0.95, knn 0.86, rule 1, margin 0.82, agree 3
- Rationale: mentions 'Gateway not responding' and the VPN client
- Alternatives: Network/Wireless (0.04)

**Priority**

- **High** (P2): deadline today; floor: VPN baseline
- Driver: deadline today → U1
- Driver: floor: VPN baseline P2

**KEDB match**

- **KB-0018**: VPN will not connect or repeatedly disconnects (score 0.86)
- Matched code in the ticket: `Gateway not responding`
- Matched field: errors; why: exact error code + VPN mapping
- Excerpt: “Gateway not responding”
- Alternatives: KB-0019 (0.51), KB-0023 (0.38)

**Recommended solution** (source: KEDB + Historical Incident)

1. Confirm the device has working internet access without the VPN `[KB-0018·diag.1]` · L1 · diagnostic
2. Check the VPN client log for 'Gateway not responding' and note the gateway name `[KB-0018·diag.2]` · L1 · diagnostic
3. Restart the VPN client and reconnect `[KB-0018·res.1]` · L1 · resolution ~~already tried~~
4. Connect to the alternate gateway listed in the VPN client profile `[KB-0018·res.2]` `[INC-H0001]` · L1 · resolution
5. Clear the saved VPN credentials and sign in again `[KB-0018·res.3]` `[INC-H0002]` · L1 · resolution

## 3. Similar cases

| Incident | Score | Tier | Origin | Why | Resolution |
|---|---|---|---|---|---|
| INC-H0001 | 0.88 | SIMILAR | VERIFIED | same subcategory; shared terms: VPN, gateway, not responding | Switched the user to the alternate regional gateway in the VPN profile; connection stable. |
| TKT-0001 | 0.84 | SIMILAR | SYNTHETIC | same subcategory; shared terms: VPN, gateway | Resolved via KB-0018: Restart the VPN client and reconnect |
| TKT-0002 | 0.71 | RELATED | SYNTHETIC | same subcategory; shared terms: VPN, connect | Resolved via KB-0018: Restart the VPN client and reconnect |

SYNTHETIC resolutions are shown for context only and are never cited as evidence.

## 4. Decision

**RECOMMEND**
- `escalation_required`: **false**
- `human_approval_required`: **true** (priority High)

## 5. Evidence trail (audit)

| Kind | Ref | Rank | Score | Tier | Why |
|---|---|---|---|---|---|
| similar | INC-H0001 | 1 | 0.88 | SIMILAR | same subcategory; shared terms: VPN, gateway, not responding |
| similar | TKT-0001 | 2 | 0.84 | SIMILAR | same subcategory; shared terms: VPN, gateway |
| similar | TKT-0002 | 3 | 0.71 | RELATED | same subcategory; shared terms: VPN, connect |
| kedb | KB-0018 | 1 | 0.86 | - | exact error code + VPN mapping |

**Events**

- `2026-11-17T09:45:20Z` ticket.submitted (assisted)
- `2026-11-17T09:45:20Z` analysis.completed (assisted)

## Contract output

```json
{
  "ticket_id": "DEMO-001",
  "classification": {
    "category": "Network",
    "subcategory": "VPN",
    "confidence": 0.93
  },
  "priority": {
    "level": "High",
    "reason": "deadline today; floor: VPN baseline"
  },
  "similar_incidents": [
    {
      "incident_id": "INC-H0001",
      "similarity_score": 0.88,
      "resolution": "Switched the user to the alternate regional gateway in the VPN profile; connection stable."
    },
    {
      "incident_id": "TKT-0001",
      "similarity_score": 0.84,
      "resolution": "Resolved via KB-0018: Restart the VPN client and reconnect"
    },
    {
      "incident_id": "TKT-0002",
      "similarity_score": 0.71,
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
      "Connect to the alternate gateway listed in the VPN client profile",
      "Clear the saved VPN credentials and sign in again"
    ],
    "source": "KEDB + Historical Incident"
  },
  "escalation_required": false,
  "human_approval_required": true
}
```
