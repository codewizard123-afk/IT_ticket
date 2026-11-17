# Intelligent IT Ticket Analysis & Resolution Assistant — Low-Level Architecture (v2)

> HCLTech Future Ready AI Engineer Hackathon · Use Case 1
> Stack: Python 3.11 · FastAPI · Streamlit · Groq (free tier) · ChromaDB · sentence-transformers · SQLite
> Team: P1 Data & KB · P2 AI Engine/RAG · P3 Backend & Control · P4 Analyst UI & Delivery

**What changed from v1.** v2 adds:
- the fixed output contract (§4)
- an owner for every module
- FastAPI as the integration layer, with the UI calling it over HTTP
- a three-way decision gate: recommend, clarify or escalate (§8.3)
- synthetic resolved history, because the historical corpus has no resolutions (§6.3)
- access control
- MTTR timing (FR-09, a team-defined requirement)
- a manual-mode baseline and a timed trial (§9.3)

---

## 0. Contents

1. Scope and design principles
2. Data assets and findings
3. Taxonomy, priority scale and the ticket→KEDB mapping
4. **Output contract** (the required response format)
5. System overview: components, owners, deployment, sequence
6. Offline pipeline (P1): data, synthetic history, privacy and indexing
7. AI engine (P2): signals, split, classify, prioritise, retrieve, synthesise, validate
8. Control layer (P3): orchestration, decision gate, escalation, approval, access, audit, feedback, MTTR
9. Analyst interface (P4): screens, manual mode, timed trial
10. HTTP API specification
11. Persistence (ChromaDB and SQLite)
12. Groq usage, rate limits and degraded modes
13. Security and privacy
14. Evaluation design, including the MTTR analysis (P1)
15. Requirement traceability (FR-01 to FR-09, with owners)
16. Walkthrough of the seven demo tickets with expected outputs
17. Repository layout and configuration
18. Assumptions, trade-offs and known limitations

---

## 1. Scope and design principles

**In scope.** The assistant analyses a ticket. It classifies it, finds similar or duplicate incidents, matches the Known Error Database (KEDB), recommends an evidence-backed resolution, and then recommends, asks for more details or escalates. It also captures analyst feedback, a full audit trail and MTTR timings.

**Out of scope** (brief §5): auto-closing tickets, running privileged fixes, and writing to a production ITSM system. The system recommends; a human acts.

| # | Principle | Consequence |
|---|-----------|-------------|
| P1 | Evidence first | Every recommended step cites a KEDB step (`KB-xxxx#res.n` or `#diag.n`) or a verified historical incident ID. Steps without a citation are removed (§7.8). |
| P2 | Deterministic core, LLM at the edges | Signals, retrieval, scoring, priority, the decision gate and escalation are rule- or math-based. The LLM is used only for classification votes, issue splitting and wording the resolution, and each of these has a non-LLM fallback. |
| P3 | Explainable confidence | Confidence comes from a weighted formula over visible components, not from the LLM's own estimate. |
| P4 | Fail safe, not silent | Weak, conflicting or vague evidence leads to *clarify*. High impact or security leads to *escalate*. The system never guesses. |
| P5 | Only approved knowledge reaches users | Only KEDB articles with `status=approved` and verified resolutions count as recommendation sources. |
| P6 | Free-tier aware | At most 2 Groq calls per ticket, response caching, 429 backoff and a fully working offline mode. |
| P7 | Contract first | The output contract, Pydantic models and module interfaces are frozen in `shared/` before parallel work starts. Each person codes against stubs. |

---

## 2. Data assets and findings

### 2.1 Inventory

| File (repo path) | Role | Records | Facts verified on inspection |
|------------------|------|---------|------------------------------|
| `data/raw/Dataset_ticket_4.xlsx` | Raw historical tickets (Italian) | 1,800 | Columns `Category, Subcategory, Text`. 5 categories, 18 subcategories, exactly 100 per subcategory. |
| `data/processed/tickets.json` | Groq-translated corpus → **ChromaDB** | 1,800 | `status=closed` for all. **`resolution=null` for all.** Title = description in 1,463 (81%). Median description length 13 words. 9 exact-duplicate groups (18 rows). Metadata keys: `source, dataset, language, original_it`. |
| `data/kb/articles.json` | **KEDB** | 30 (KB-0001…KB-0030) | symptoms, common_error_codes, root_cause, diagnostic_steps, resolution{steps, estimated_time, requires_admin}, escalation_criteria, related_kb_ids, tags, priority_hint, last_updated |
| `data/samples/sample_tickets.json` | **Input contract** and demo set | 7 (DEMO-001…007) | `category/subcategory/priority=null`, `status=new`. Gold labels in `metadata.expected_*`. DEMO-005 is multi-category. |
| `docs/brief.docx` | Requirements | — | FR-01 to FR-08, evaluation dimensions, scope rules |

### 2.2 Input ticket schema (same as `sample_tickets.json`)

```json
{
  "id": "DEMO-001",
  "title": "string",
  "description": "string",
  "category": null, "subcategory": null, "priority": null,
  "status": "new",
  "resolution": null,
  "metadata": {"source": "demo", "expected_category": "...", "...": "..."}
}
```

Optional extra fields the API accepts for FR-07 and access control: `application`, `environment`, `location`, `reporter`, `assignment_group`, `restricted` (bool).

### 2.3 KEDB schema

This is `articles.json` as supplied. At ingest P1 adds these governance fields: `status` (`approved|draft|retired`), `version` (int), `approved_by`, `source` (`seed|analyst`).

### 2.4 Findings that drive the design

| Finding | Design response | Owner |
|---------|-----------------|-------|
| No historical ticket has a resolution, but the contract needs `similar_incidents[].resolution` | **Synthetic history** (§6.3). (a) Every historical ticket gets a *KB-derived synthetic* resolution, labelled as such and **never counted as independent evidence**. (b) A curated set of ~60 *verified* resolved incidents, which are allowed to be a "Historical Incident" source. (c) Tickets resolved in the app become verified incidents. | P1 |
| Priority in `tickets.json` depends only on subcategory (Network/Database → P2, Hardware/Software → P3, Inquiry/Help → P4; P1 never appears) | Treat it as a *baseline prior*. Final priority comes from an impact × urgency engine (§7.4). The evaluation report states that these labels are synthetic. | P2 / P1 |
| The ticket taxonomy and the KEDB taxonomy differ (e.g. `Wireless` vs `WiFi`; DB engines vs DB symptom types) | A versioned mapping table (§3.3) used as a soft boost, never as a filter | P1 data, P2 use |
| KEDB gaps: DHCP, IP Address, Disk and DB2-specific issues | Curated verified incidents fill some gaps. The rest give low KB relevance, which triggers clarify or escalate. Gaps are listed in the evaluation report. | P1 |
| Historical text is short and translated; demo tickets are long and full of signals | Hybrid retrieval (dense + BM25 + exact error-code index), with signal extraction before retrieval | P2 |
| DEMO-005 contains two issues | An issue splitter. The primary issue fills the contract and the secondary goes in `details.secondary_issues` | P2 |
| FR-07 asks for application, assignment group, environment and date filters, but none exist in the data | Rule-derived synthetic metadata (§6.3.1), flagged in `metadata.derived_fields` | P1 |

---

## 3. Taxonomy, priority scale and mapping

### 3.1 Classification taxonomy (FR-01)

| Category | Subcategories | Baseline priority | Assignment group |
|----------|---------------|-------------------|------------------|
| Network | VPN, Wireless, DNS, DHCP, IP Address | P2 | `NET-OPS` |
| Database | Oracle, MS SQL Server, DB2 | P2 | `DBA` |
| Hardware | CPU, Disk, Keyboard, Memory, Monitor, Mouse | P3 | `EUC-HW` |
| Software | Email, Operating System | P3 | `EUC-SW` |
| Inquiry/Help | Antivirus, Internal Application | P4 | `SERVICE-DESK` (Antivirus with a threat → `SEC-OPS`) |

Each issue also gets an **intent** (`incident | service_request | how_to | access_request`). Subcategory `Other` is allowed when no label scores high enough; it forces confidence band `low`.

### 3.2 Priority scale (contract `priority.level`)

| Code (internal) | Contract `level` | Meaning |
|-----------------|------------------|---------|
| P1 | `Critical` | Major impact and urgent: many users, production, security, or month-end blocked |
| P2 | `High` | Significant impact or a deadline at risk |
| P3 | `Medium` | Single user, degraded, workaround possible |
| P4 | `Low` | How-to, request, cosmetic |

`shared/constants.py: PRIORITY_LEVEL = {"P1":"Critical","P2":"High","P3":"Medium","P4":"Low"}`

### 3.3 Ticket subcategory → KEDB mapping (`config/taxonomy_map.yaml`, owned by P1)

| Ticket category / subcategory | Primary KB (boost 1.0) | Secondary KB (boost 0.5) | Coverage |
|-------------------------------|------------------------|--------------------------|----------|
| Network / VPN | KB-0018, KB-0019 | KB-0023 | Full |
| Network / Wireless | KB-0020, KB-0021 | KB-0013 | Full |
| Network / DNS | KB-0023 | KB-0024 | Full |
| Network / DHCP | — | KB-0023, KB-0020 | **Gap** |
| Network / IP Address | — | KB-0023, KB-0022 | **Gap** |
| Database / Oracle, MS SQL Server | KB-0001…KB-0005 | — | Full (by symptom) |
| Database / DB2 | KB-0001…KB-0005 | — | Partial |
| Hardware / Monitor | KB-0008 | KB-0011 | Full |
| Hardware / Keyboard, Mouse | KB-0012 | KB-0011 | Full |
| Hardware / CPU | KB-0030 | KB-0006, KB-0007 | Partial |
| Hardware / Memory | KB-0030 | KB-0006 | Partial |
| Hardware / Disk | — | KB-0006 | **Gap** |
| Software / Email | KB-0029, KB-0015 | KB-0030 | Full |
| Software / Operating System | KB-0006, KB-0029 | KB-0026, KB-0027 | Partial |
| Inquiry/Help / Antivirus | KB-0028 | KB-0014 | Partial |
| Inquiry/Help / Internal Application | KB-0029, KB-0030, KB-0014 | KB-0026 | Partial |

Within Database, the article is chosen by symptom:

| Symptom | KB |
|---------|----|
| slow, timeout | KB-0001 |
| connection pool | KB-0002 |
| access denied, login failed, locked | KB-0003 |
| password expired | KB-0004 |
| backup or restore | KB-0005 |

KB-0007, 0009, 0010, 0013, 0016, 0017, 0022, 0024–0027 are unmapped but still reachable by semantic search.

---

## 4. Output contract (required format)

### 4.1 Contract

`POST /api/v1/analyze?view=contract` returns **exactly** these keys. The default `view=full` returns the same keys plus one extra top-level key, `details` (§4.3). Contract objects never contain extra keys; all extensions live in `details`.

```json
{
  "ticket_id": "INC1001",
  "classification": {"category": "Network", "subcategory": "VPN", "confidence": 0.94},
  "priority": {"level": "High", "reason": "Multiple users affected"},
  "similar_incidents": [
    {"incident_id": "INC0892", "similarity_score": 0.91, "resolution": "Restart VPN service"}
  ],
  "kedb_match": {"found": true, "error_code": "VPN_001", "solution": "Restart VPN service"},
  "recommended_solution": {
    "steps": ["Check VPN connectivity", "Restart VPN service", "Verify connection"],
    "source": "Historical Incident"
  },
  "escalation_required": false,
  "human_approval_required": true
}
```

JSON Schema: `shared/contract/analysis_output.schema.json`. Contract tests validate every API response against it.

### 4.2 Field-by-field production rules

| Field | Rule | Producer (owner) |
|-------|------|------------------|
| `ticket_id` | Input `id`. If missing, the API assigns `INC-<yyyymmdd>-<seq>`. | `backend/intake.py` (P3) |
| `classification.category / subcategory` | Primary issue, fused classifier top-1 (§7.3) | `ai/classify.py` (P2) |
| `classification.confidence` | Fused confidence, 0–1, rounded to 2 dp | P2 |
| `priority.level` | `PRIORITY_LEVEL[ticket_priority]`, where ticket priority is the most severe across issues | `ai/priority.py` (P2) |
| `priority.reason` | The top 1–2 priority drivers joined with "; " (e.g. `"15 users affected; month-end close blocked"`). If there are no drivers: `"Baseline for <subcategory>"`. | P2 |
| `similar_incidents[]` | Top 5 after the duplicate guard (§7.5), sorted by `similarity_score` (cosine, 2 dp). `resolution` = the stored resolution text, or `null` if there is none. | `ai/ticket_search.py` (P2) using P1's stores |
| `kedb_match.found` | `kb_relevance ≥ τ_kb (0.50)` | `ai/kb_search.py` (P2) |
| `kedb_match.error_code` | **The KEDB known-error ID** of the top article (e.g. `"KB-0018"`), or `null` if `found=false`. The literal code matched in the ticket (e.g. `"Gateway not responding"`) is in `details.kedb.matched_error_code`. | P2 |
| `kedb_match.solution` | The first *resolution* step of the top article, truncated to 200 chars, or `null` | P2 |
| `recommended_solution.steps` | The validated, ordered step texts (diagnostics first). Each step's citations are in `details.steps[i].citations`. | `ai/synthesize.py` + `ai/validate.py` (P2) |
| `recommended_solution.source` | `"KEDB"` if all citations are KB · `"Historical Incident"` if all are verified incidents · `"KEDB + Historical Incident"` if mixed · `"None"` if no grounded steps | P2 |
| `escalation_required` | `true` if the decision of **any** issue is `escalate` (§8.3) | `backend/decision.py` (P3) |
| `human_approval_required` | §8.5 rule | P3 |

### 4.3 `details` extension (full view)

```json
"details": {
  "analysis_id": "uuid",
  "decision": "recommend | clarify | escalate",
  "clarifying_questions": ["Is the issue slowness, or a login failure?"],
  "intent": "incident",
  "classification": {"band": "high", "components": {"llm":0.9,"knn":0.82,"rule":1.0,"margin":0.6,"agree":3},
                     "alternatives": [{"category":"Network","subcategory":"Wireless","score":0.08}],
                     "rationale": "mentions 'Gateway not responding' and VPN client"},
  "priority": {"code": "P2", "impact": 3, "urgency": 1, "drivers": ["deadline today → U1", "floor: KB-0018 hint P2"]},
  "similar": [{"incident_id":"TKT-0312","tier":"SIMILAR","excerpt":"...","why":"shared terms: VPN, gateway; same subcategory",
               "resolution_origin":"synthetic_kb_derived|synthetic_escalated|curated_verified|analyst_verified|none",
               "category":"Network","subcategory":"VPN"}],
  "kedb": {"kb_id":"KB-0018","title":"VPN will not connect or repeatedly disconnects","score":0.83,
           "matched_error_code":"Gateway not responding","matched_field":"errors","excerpt":"Gateway not responding",
           "why":"exact error code + VPN mapping","alternatives":[{"kb_id":"KB-0019","score":0.51}]},
  "steps": [{"text":"...","type":"diagnostic|resolution","citations":["KB-0018#diag.4"],"role":"L1|L2","already_attempted":false}],
  "escalation": {"reasons":[{"rule":"E6","text":"KB-0018: many users are affected"}],"assignment_group":"NET-OPS","tier":"L2","bundle_id":"uuid"},
  "approval_reasons": ["priority High", "step requires admin"],
  "secondary_issues": [{"classification":{...}, "priority":{...}, "kedb_match":{...}, "recommended_solution":{...}, "decision":"escalate", "escalation":{...}}],
  "redactions": {"count": 1, "types": ["EMAIL"]},
  "degraded": [],
  "timings_ms": {"sanitize":2,"signals":3,"classify":640,"retrieve":85,"synthesize":1100,"total":1900},
  "versions": {"pipeline":"2.0.0","embed_model":"BAAI/bge-small-en-v1.5","classify_model":"...","synth_model":"...","kb_index":"sha1..."}
}
```

The role filter (§8.6) is applied **after** assembly. For example, for `l1` users, L2 step texts in `details.steps` are replaced with `"Handled by <group>"`.

---

## 5. System overview

### 5.1 HLD box → module → owner

| HLD box | Module(s) | Owner |
|---------|-----------|-------|
| Ticket form (Streamlit) | `ui/pages/1_Analyze.py`, `ui/client.py` | P4 |
| (privacy, implicit) | `privacy/redact.py` | P1 (called by P3) |
| Signal extractor | `ai/signals.py` | P2 |
| Classifier (LLM + fallback) | `ai/split.py`, `ai/classify.py`, `ai/priority.py` | P2 |
| Ticket vector DB | `store/chroma_store.py`, `ingest/index_tickets.py` (P1) · `ai/ticket_search.py` (P2) | P1 + P2 |
| Curated runbooks (KEDB) | `data/kb/articles.json`, `ingest/index_kedb.py` (P1) · `ai/kb_search.py` (P2) | P1 + P2 |
| Resolution synthesizer | `ai/synthesize.py`, `ai/validate.py` | P2 |
| Escalation check | `backend/decision.py`, `backend/escalate.py` | P3 |
| L2/L3 escalation + bundle | `backend/escalate.py` (bundle), UI download (P4) | P3 + P4 |
| Resolution + sources | `backend/assemble.py` (contract), UI evidence cards (P4) | P3 + P4 |
| Analyst feedback + MTTR log | `backend/feedback.py`, `backend/mttr.py`, `backend/audit.py` · UI (P4) | P3 + P4 |
| Feeds KB (loop arrow) | `backend/kb_drafts.py` (P3) → `ingest/index_kedb.reindex_article()` (P1) | P3 + P1 |

### 5.2 Component and deployment diagram

```mermaid
flowchart LR
  subgraph P4["P4 · Streamlit UI (:8501)"]
    UI1[Analyze] --- UI2[Manual mode] --- UI3[Search & filter] --- UI4[Review queue] --- UI5[KB browser] --- UI6[Audit] --- UI7[MTTR dashboard]
  end
  subgraph P3["P3 · FastAPI backend (:8000)"]
    AUTH[auth + RBAC] --> PIPE[pipeline.analyze]
    PIPE --> DEC[decision gate] --> ESC[escalation + bundle] --> ASM[assemble contract]
    FB[feedback] ; AUD[audit log] ; MT[mttr events] ; KBD[kb drafts]
  end
  subgraph P2["P2 · AI engine (python package)"]
    SIG[signals] --> SPL[split] --> CLS[classify] --> PRI[priority]
    PRI --> TS[ticket_search] & KS[kb_search] --> SYN[synthesize] --> VAL[validate]
    LLM[groq_client]
  end
  subgraph P1["P1 · Data & stores"]
    RED[privacy.redact]
    CH[(ChromaDB)] ; BM[(BM25 + error-code index)]
    ING[ingest/* builders] ; EV[eval/*]
  end
  DB[(SQLite app.db)]
  UI1 & UI2 & UI3 & UI4 & UI5 & UI6 & UI7 -- HTTP/JSON --> AUTH
  PIPE --> RED
  PIPE --> SIG
  TS --> CH & BM
  KS --> CH & BM
  CLS & SYN --> LLM
  FB & AUD & MT & KBD --> DB
  KBD --> ING
  EV --> PIPE
```

**Process model.** Two processes: `uvicorn backend.main:app` and `streamlit run ui/app.py`. The AI engine and stores are imported in-process by the backend. The UI never imports `ai/`, `store/` or `backend/` directly; it only uses `ui/client.py`, which talks to the API over HTTP.

### 5.3 Request sequence (assisted path)

```mermaid
sequenceDiagram
  participant U as Analyst (UI)
  participant B as Backend (P3)
  participant R as Privacy (P1)
  participant E as AI engine (P2)
  participant C as Chroma/BM25 (P1)
  participant G as Groq
  participant S as SQLite
  U->>B: POST /analyze (ticket) + token
  B->>S: event ticket.submitted
  B->>R: redact(title, description)
  B->>E: engine.run(sanitized, ctx{role, feedback_prior})
  E->>E: signals → (split)
  E->>C: kNN tickets (votes)
  E->>G: #1 classify+split (JSON)
  E->>E: fuse confidence, priority
  par per issue
    E->>C: ticket hybrid search
    E->>C: KB chunks + error codes + BM25
  end
  E->>G: #2 synthesize (cited JSON)
  E->>E: validate citations
  E-->>B: EngineResult
  B->>B: decision gate → escalation → approval → assemble → role filter
  B->>S: analyses, evidence, event analysis.completed, audit row
  B-->>U: contract + details
  U->>B: POST /events ui.viewed
  U->>B: POST /feedback (per item) · POST /clarify (if asked)
  U->>B: POST /decisions (resolved|escalated, notes)
  B->>S: decision, event decision.submitted, MTTR
  B->>C: upsert verified incident (if resolved with notes)
```

---

## 6. Offline pipeline (P1)

### 6.1 Translation (already done: `tickets.json`)

`ingest/preprocess.py` is kept for reproducibility: xlsx → Groq translate (`temperature=0`, batches of 20, cached by `sha1(text)`) → `tickets.json`. **Do not re-run it unless needed.** The committed `tickets.json` is the source of truth. QA checks: length ratio 0.5–2.0, non-empty, `language=en`.

### 6.2 Privacy step (`privacy/redact.py`)

This is used offline (on all corpora before indexing) **and** online (P3 calls it before any LLM or embedding call).

```python
def redact(text: str, keep_map: bool = True) -> RedactionResult:
    # RedactionResult(text: str, items: list[Redaction], secret_detected: bool)
    # Redaction(type, placeholder, span, original_hash)   # original stored only as a hash
```

| Pattern | Placeholder |
|---------|-------------|
| Email | `<EMAIL_n>` |
| Phone (intl/IN/US formats) | `<PHONE_n>` |
| IPv4/IPv6 | `<IP_n>` (the original is kept in an in-memory map only, for the L2 bundle) |
| Person name after "my name is / I am / user:", plus a names list from `config/privacy.yaml` | `<PERSON_n>` |
| Employee ID / asset tag (configurable regex) | `<ID_n>` |
| `password[:=]`, `pwd=`, `Password=` in connection strings | `<SECRET>`, and sets `secret_detected` |
| API tokens (`sk-…`, `gsk_…`, long hex/base64, Bearer) | `<TOKEN>`, and sets `secret_detected` |

These are **stable placeholders**: the same value gets the same placeholder within one ticket. Tests live in `tests/p1_data/test_redact.py`, with ≥ 30 positive and negative cases, including false-positive guards (e.g. `ORA-01013` and `0xc0000005` must **not** be redacted).

### 6.3 Synthetic history (`ingest/enrich.py`, `data/seed/`)

#### 6.3.1 Enriched historical corpus → `data/processed/tickets_enriched.json`

`tickets.json` is never modified. Fields added to each record (seeded RNG `seed=42`, so the result is reproducible):

| Field | Rule | Flag |
|-------|------|------|
| `opened_at` | Uniform over the 180 days before `2026-10-31`, business-hours-weighted | `derived_fields` |
| `resolved_at` | `opened_at` + log-normal duration by baseline priority (median: P2 4 h, P3 8 h, P4 24 h; σ=0.6) | `derived_fields` |
| `assignment_group` | From the category table (§3.1) | `derived_fields` |
| `application` | Keyword dictionary (`outlook→Outlook`, `oracle→Oracle DB`, `vpn→VPN client`, `antivirus→Endpoint AV`, …); otherwise `"<Category> general"` | `derived_fields` |
| `environment` | Keyword rules (`home/remote→remote`, `office/floor/building→office`, `production/server→production`); otherwise `"unspecified"` | `derived_fields` |
| `restricted` | `false`. 2% of Inquiry/Help/Antivirus tickets are set to `true` to demonstrate access filtering. | `derived_fields` |
| `resolution` | **KB-derived synthetic**: take the top KB article by dense similarity + taxonomy boost. If its score is ≥ 0.45, `resolution` = a template like `"Resolved via KB-0018: <first resolution step>; <verification step>."`. Otherwise `"Escalated to <group>; resolved by L2 (no KEDB article)."` | `resolution_origin = "synthetic_kb_derived"` or `"synthetic_escalated"`, plus `resolution_kb_id` |

**Rule:** `synthetic_*` resolutions appear in `similar_incidents[].resolution` (with `details.similar[].resolution_origin`) but are **never** used as citations for the recommendation. Citing them would be circular, because they were derived from the KB.

#### 6.3.2 Curated verified incidents → `data/seed/resolved_incidents.json`

- ~60 records, `INC-H0001…`, in the same schema as enriched tickets. Coverage: 3 per subcategory (54) plus 6 that fill KB gaps (DHCP ×2, IP Address ×2, Disk ×1, DB2 ×1).
- Written with Groq assistance, then **reviewed and edited by a person** (checklist in `data/seed/README.md`). Resolutions are concrete, realistic workarounds that are consistent with KB safety rules (no "disable antivirus").
- `resolution_origin = "curated_verified"`, `verified = true`, `verified_by = "<team member>"`.
- These **can** be cited as `"Historical Incident"`.

#### 6.3.3 Test and trial tickets → `data/test/`

| File | Contents |
|------|----------|
| `test_tickets.json` | The 7 demos + ~25 candidate-created tickets in the `sample_tickets.json` format, with gold `metadata.expected_category/subcategory/priority/kb_ids/decision`. Includes: gap subcategories (4), out-of-taxonomy (printer, MFA: 3), vague (3), contradictory (2), security (3), keyword traps (3), PII-bearing (3), prompt injection (2), multi-issue (2). |
| `trial_tickets.json` | 12 tickets T01–T12 for the timed trial (§9.3). They do not overlap with the demos. 2 per main category plus 2 gap/escalation cases. Gold answers are in a separate `trial_gold.json`, which is never sent to the UI. |

All candidate-created data has `metadata.source = "synthetic_team"`.

### 6.4 KEDB ingest (`ingest/index_kedb.py`)

1. Validate `articles.json` against `shared/schemas.py:KBArticle`. Add governance fields (`status=approved, version=1, source=seed`). Write `data/kb/articles_governed.json`.
2. Build **field-level chunks**:

| chunk_type | Text | id |
|------------|------|----|
| `title_root` | title + root_cause | `KB-0018#title` |
| `symptom` | each symptom | `KB-0018#sym.n` |
| `errors` | joined error codes | `KB-0018#err` |
| `diagnostic` | each diagnostic step | `KB-0018#diag.n` |
| `resolution` | each resolution step (stored for citation, **not searched**) | `KB-0018#res.n` |
| `escalation` | escalation_criteria | `KB-0018#esc` |

3. **Error-code index** `indexes/error_codes.json`: normalised code → `[kb_id]` (lowercased, whitespace collapsed, punctuation stripped).
4. BM25 over the searchable chunks → `indexes/bm25_kedb.pkl`.
5. `reindex_article(kb_id)` re-chunks and upserts one article. It is used by the KB-draft approval flow.

### 6.5 Ticket indexing (`ingest/index_tickets.py`)

- Embedding text: `f"{title}. {description}"` if title ≠ description, otherwise `description` (redacted). Labels are **not** included.
- Model: `BAAI/bge-small-en-v1.5` via sentence-transformers, normalised. Embeddings are **always passed explicitly** to Chroma. Never use Chroma's default embedding function.
- Collections are listed in §11.1. Exact duplicates share a `dup_group_id`.
- Chroma metadata rules: scalar values only. Lists are stored as CSV strings. `null` is stored as `""`. Dates are stored as int epoch seconds (`opened_ts`) so `$gte/$lte` filters work.
- BM25 → `indexes/bm25_tickets.pkl`.
- `indexes/manifest.json` holds `{source_sha1, model, built_at, counts}`. The backend `/health` endpoint reports `index_stale` if the hashes differ.

### 6.6 Store API provided by P1 (`store/`)

```python
class TicketStore:
    def query(self, text: str, k: int, where: dict | None = None,
              collections: tuple[str, ...] = ("tickets_hist", "tickets_verified", "tickets_live")) -> list[TicketHit]
    def bm25(self, text: str, k: int) -> list[tuple[str, float]]
    def get(self, ids: list[str]) -> list[TicketRecord]
    def upsert_live(self, rec: TicketRecord) -> None          # used by P3
    def mark_verified(self, ticket_id: str, resolution: str, by: str) -> None   # moves to tickets_verified

class KBStore:
    def query_chunks(self, text: str, k: int, where: dict | None = None) -> list[ChunkHit]
    def bm25(self, text: str, k: int) -> list[tuple[str, float]]        # chunk ids
    def lookup_codes(self, codes: list[str]) -> dict[str, list[str]]    # code → kb_ids
    def get_article(self, kb_id: str) -> KBArticle
    def chunk_text(self, chunk_id: str) -> str                          # e.g. "KB-0018#res.1"
    def all_error_phrases(self) -> list[str]                            # used by P2 signals

def embed(texts: list[str]) -> np.ndarray        # one shared embedder (singleton)
def build_all(eval_split: bool = False) -> None  # python -m ingest.build_all
```

---

## 7. AI engine (P2)

Entry point: `ai/engine.py`

```python
def run(ticket: SanitizedTicket, ctx: EngineContext) -> EngineResult
# EngineContext(role, feedback_prior: dict[kb_id, float], filters: dict, clarification: str | None,
#               ticket_store: TicketStore, kb_store: KBStore, llm: LLMClient)
# EngineResult(signals, issues: list[IssueResult], degraded: list[str], timings_ms: dict)
# IssueResult(text_span, classification, priority, similar, kb_matches, recommendation)
```

Every function is pure over its inputs. Stores and the LLM client are injected, so P2 can test with `shared/stubs.py`.

### 7.1 Signal extractor (`ai/signals.py`): deterministic

```python
@dataclass
class Signals:
    error_codes: list[str]; products: list[str]; affected_users: int | None
    scope: Literal["single","multiple","team","site","org"]
    urgency_terms: list[str]; business_critical: bool
    environment: str | None; location: str | None; onset: str | None; change_trigger: str | None
    multi_issue_markers: bool; security_flags: list[str]; tried_steps: list[str]; keywords: list[str]
```

| Family | Patterns (`config/signals.yaml`) |
|--------|----------------------------------|
| Oracle | `ORA-\d{5}` |
| SQL Server | `(SQL Server )?error \d{3,5}`, `SQLSTATE \w{5}` |
| Windows exception | `0x[0-9a-fA-F]{8}` |
| MSI | `(MSI )?error 16\d\d` |
| Stop codes | `\b[A-Z]+(_[A-Z]+){2,}\b` |
| Phrase codes | All KEDB `common_error_codes`, via `KBStore.all_error_phrases()` |
| Users | `(\d+)\s+(users|people|staff|colleagues)`, "approximately N", "whole team", "multiple colleagues" |
| Urgency | "deadline today", "end of day", "month-end", "blocking", "next week" |
| Business critical | "production", "month-end close", "finance team", "customer-facing", "critical system" (a deadline alone is **not** business-critical) |
| Security | malware, phishing, ransomware, breach, quarantined, suspicious, data loss |
| Tried steps | "restarted", "tried a different cable", "reinstalled", "DBA checked" |

### 7.2 Issue splitter (`ai/split.py`)

- **Trigger:** `multi_issue_markers`, or the top-2 categories differ and both score ≥ 0.35.
- **Method:** the split is requested inside Groq call #1. Fallback: split on `(1)/(2)`, numbered lists, or sentence boundaries between product mentions.
- At most 2 issues. Primary = the issue with the higher priority, with ties going to the first mentioned.
- **Causal cross-link:** added only if it is cited (e.g. KB-0019 diag "Check endpoint CPU…" for DEMO-005).

### 7.3 Classifier (`ai/classify.py`)

Three voters are fused:

- **A. kNN** (`tickets_hist`, k=15): `v_knn(s) = Σ sim_i·[label_i=s] / Σ sim_i`
- **B. LLM** (Groq #1, JSON mode, `temperature=0`): taxonomy + intents + 2 few-shot examples per category + top-5 kNN neighbours + signals. The output is validated with Pydantic. Retry once, then fall back.
- **C. Rules** (`config/rules.yaml`): `ORA-*→Oracle`, `18456→MS SQL Server`, `DB2|SQL\d{4}N→DB2`, `outlook|exchange→Email`, `vpn|anyconnect|globalprotect→VPN`, `wifi|wireless|ssid|access point→Wireless`, `monitor|display|displayport|hdmi→Monitor`, `cpu|fan|overheat→CPU`, `antivirus|quarantine→Antivirus`, …

```
score(s) = 0.45·v_llm + 0.35·v_knn + 0.20·v_rule
conf     = clip(0.6·score(pred) + 0.25·margin + 0.15·agree/3, 0, 1)
bands    : high ≥ 0.75 · medium 0.55–0.75 · low < 0.55
conflict : LLM top-1 and kNN top-1 in different categories, both ≥ 0.4
LLM down : weights knn 0.65 / rule 0.35, conf capped at 0.80, degraded += llm_classify_fallback
```

The weights and bands are tuned by P1's evaluation on the validation split, so that observed accuracy in the "high" band is ≥ 90%.

### 7.4 Priority engine (`ai/priority.py`): rules

**Impact:**
- I1: ≥ 10 users, site/org scope, production down, data loss or security
- I2: multiple/team scope, 2–9 users, or a shared resource
- I3: single user

**Urgency:**
- U1: deadline today, month-end, fully blocked, or security
- U2: degraded, or a deadline within days
- U3: how-to/request

| | U1 | U2 | U3 |
|-|----|----|----|
| I1 | P1 | P2 | P3 |
| I2 | P2 | P3 | P3 |
| I3 | P3 | P3 | P4 |

```
if intent in {how_to, service_request}: p = p_matrix
elif no impact/urgency signal:          p = p_baseline
else:                                   p = most_severe(p_matrix, p_baseline, p_kb_hint if kb_relevance ≥ τ_kb)
```

Output: `code`, `impact`, `urgency`, `drivers[]`. The ticket priority is the most severe across issues. The analyst can override it, and the override is audited by P3.

### 7.5 Ticket retrieval and duplicates (`ai/ticket_search.py`)

1. Dense search over `tickets_hist + tickets_verified + tickets_live` (top 30 each) plus BM25 top 30, combined with RRF (`k=60`).
2. Bonuses: `+0.10·same_category +0.05·same_subcategory +0.05·verified_resolution`.
3. `where` filters (FR-07): category, subcategory, priority, status, source, `opened_ts` range, assignment_group, application, environment, and `restricted` (set by role, §8.6).
4. **Keyword-trap guard:** a hit counts as "related" only if `cosine ≥ 0.55` **and** (same predicted category **or** cross-encoder ≥ 0.5, if rerank is on). BM25-only hits are returned in `keyword_only[]` (shown collapsed in the UI and counted in evaluation).
5. **Tiers:**

| Tier | Condition |
|------|-----------|
| DUPLICATE | same `text_hash` |
| LIKELY DUPLICATE | cosine ≥ 0.92 and same subcategory |
| SIMILAR | cosine ≥ 0.75 |
| RELATED | passes the guard |

6. **Live clustering:** ≥ 3 near-duplicates in `tickets_live` within 72 h with status new or in review sets `possible_major_incident`.
7. **Pattern summary:** the count of historical tickets in the predicted subcategory and the top 3 recurring terms.
8. Also exposed for FR-07: `search_tickets(q, filters, k, role) -> list[TicketHit]` (`q` may be empty, which returns a filter-only listing sorted by `opened_ts` desc).

### 7.6 KEDB retrieval (`ai/kb_search.py`)

```
kb_score = 0.45·sem + 0.15·bm25 + 0.20·code_hit + 0.12·tax + 0.08·fb
sem      = max chunk cosine per article (best chunk = matched excerpt)
code_hit = 1 if any ticket error code maps to the article
tax      = 1.0 primary / 0.5 secondary mapping
fb       = feedback_prior[kb_id] ∈ [-0.5, 0.5] (passed in by P3)
τ_show = 0.35 · τ_kb = 0.50 · ambiguity: top1 − top2 < 0.05 and different subcategory → kb_ambiguous
```

This returns the top 3 articles, each with `kb_id, title, score, matched_field, excerpt, why, matched_error_code`. Also exposed: `search_kb(q, filters, k)`.

### 7.7 Resolution synthesiser (`ai/synthesize.py`)

**Evidence pack** (≤ 2.5k tokens):
- issue summary, signals and `tried_steps`
- the top 1–2 KB articles with numbered `diag.n` / `res.n` steps, `escalation_criteria`, `requires_admin` and `estimated_time`
- up to 3 **verified** incidents (`curated_verified`/`analyst_verified`, cosine ≥ 0.75, same subcategory) with IDs and resolutions
- synthetic resolutions are excluded

**Groq #2 system rules:**
- use only the pack; every step needs `citations[]` from the pack
- diagnostics come before resolution steps
- mark steps the user has already tried as `already_attempted`
- tag admin/DBA/network-ops steps as `role: L2`
- never advise bypassing security controls
- return `{"insufficient_evidence": true}` rather than guess
- ticket text goes in a `<ticket>` block and is treated as data

**Output:** `{summary, probable_cause{text,citations}, steps[{n,type,text,citations,role,already_attempted}], alternatives[], estimated_time, requires_admin, insufficient_evidence}`

**Template fallback** (LLM down or two validation failures): render the KB diag and res steps verbatim, in order, with citations, and add `degraded += llm_synth_fallback`.

### 7.8 Citation validator (`ai/validate.py`)

- Unknown citation IDs are removed. A step left with no citation is **dropped** (`unsupported_steps_removed++`).
- **Grounding:** ROUGE-L recall between the step and its cited text must be ≥ 0.30, otherwise the step is dropped.
- **Safety lint:** steps matching `disable (antivirus|firewall|UAC|defender)`, `share (your )?password`, `format (the )?(drive|disk)`, `delete .*registry`, or `grant .*(dba|owner|full control|admin)` are dropped and `safety_violation` is logged.
- If fewer than 2 steps remain, set `insufficient_evidence = true`.
- Sets `source` per §4.2.

---

## 8. Control layer (P3)

### 8.1 Orchestrator (`backend/pipeline.py`)

```python
def analyze(ticket_in: TicketIn, user: User, clarification: str | None = None) -> AnalysisResponse:
    t0 = now(); audit.event(ticket_id, "ticket.submitted", user)
    ticket = intake.validate(ticket_in)            # strips metadata.expected_* (label-leak guard)
    red = privacy.redact(ticket.title + "\n" + ticket.description)
    ctx = EngineContext(role=user.role, feedback_prior=feedback.kb_prior(), filters=role_filters(user), ...)
    er = engine.run(SanitizedTicket(...), ctx)
    per_issue = [decision.decide(issue, er.signals, red, clarification_round) for issue in er.issues]
    esc = [escalate.route(issue, d) for issue, d in zip(er.issues, per_issue)]
    approval = approval.required(er, per_issue)
    resp = assemble.build(ticket, er, per_issue, esc, approval)    # contract + details
    resp = access.filter_response(resp, user)
    store.save_analysis(resp, er, red); audit.event(ticket_id, "analysis.completed", user)
    return resp
```

Intake limits: title 5–200 chars, description 10–5,000 chars. Incoming `category/priority` are kept as `reporter_*` fields and are not used for prediction.

### 8.2 Escalation rules (`backend/escalate.py`, `config/escalation.yaml`)

| ID | Rule | Class |
|----|------|-------|
| E1 | `kb_relevance < τ_kb` **and** no verified incident with cosine ≥ 0.75 | soft |
| E2 | classification band `low` or `conflict` | soft |
| E3 | `insufficient_evidence` after validation | soft |
| E4 | ticket priority P1 or `possible_major_incident` | **hard** |
| E5 | `security_flags` or `secret_detected` | **hard** |
| E6 | KB escalation-criteria hit: the clauses of the top-2 articles (above τ_show) in `config/kb_escalation_clauses.yaml` (hand-mapped by P3 for all 30 articles to signal predicates, e.g. `KB-0018: "many users are affected" → scope in [multiple,team,site,org]`) | soft→escalate |
| E7 | Only L2/requires_admin steps remain **and** the user has already attempted all the L1 steps | soft→escalate |

**Routing:** the category's assignment group, with these overrides:
- E5 → `SEC-OPS`
- Database + requires_admin → `DBA`
- Network + E4 → `NET-OPS-MAJOR`

**Tier:** L3 if E4 or E5 fired, otherwise L2.

**Diagnostic bundle** (`GET /escalations/{bundle_id}` → JSON or Markdown): the redacted ticket, signals, classification and its components, priority drivers, reasons, top-5 similar incidents, top-3 KB matches with excerpts, attempted steps, remaining KB diagnostics, group and tier, and timestamps.

### 8.3 Decision gate (`backend/decision.py`): recommend, clarify or escalate (FR-05)

Evaluated per issue, in this order:

```
1. any HARD rule (E4, E5)                                   → escalate
2. clarification_round == 0 and any of:                     → clarify
     kb_ambiguous
     conflict and not hard
     sparse ticket: < 12 words and no product/error code
     band == low and top-2 subcategory gap < 0.10
3. any SOFT rule (E1, E2, E3, E6, E7)                       → escalate
4. otherwise                                                → recommend
```

**Clarifying questions** (≤ 3, deterministic, `config/clarify_templates.yaml`):
- For a `kb_ambiguous` pair, a template keyed by the subcategory pair (e.g. DB perf vs DB access: "Is the problem slowness/timeouts, or an access/login error?").
- Otherwise, "What is the exact error message?" if there are no error codes; "How many users are affected?" if scope is unknown; "When did it start, and did anything change?" if there is no onset.

`POST /tickets/{id}/clarify {answer}` appends `"\n[Clarification] " + answer` to the description and re-runs with `clarification_round=1`. Step 2 is then skipped, so a second low-confidence result **escalates**. This guarantees termination.

### 8.4 Contract booleans

- `escalation_required = any(decision == "escalate" for issue in issues)`
- Top-level `details.decision` = `escalate` if any issue escalates, else `clarify` if any issue needs clarification, else `recommend`.

### 8.5 Human approval rule (`backend/approval.py`)

`human_approval_required = true` if **any** of the following holds (each one is recorded in `details.approval_reasons`):
- decision ≠ recommend
- priority is Critical or High
- any step has role L2, or the KB article has `requires_admin`
- the confidence band is not high
- `source == "None"`
- `degraded` is non-empty

Otherwise it is `false`, meaning an L1 analyst may apply the steps without supervisor sign-off. Either way the system never executes anything.

### 8.6 Access control (`backend/auth.py`, `backend/access.py`)

- **Users** live in `config/users.yaml` (`username, role, assignment_group, password_hash` with bcrypt).
- `POST /auth/login` returns an HS256 JWT (`APP_SECRET`, 8 h expiry).
- **Demo mode** (`DEMO_AUTH=1`) lets you choose the role from a dropdown, and the UI shows a "DEMO AUTH" badge.

| Capability | l1 | l2 | kb_owner | admin |
|------------|----|----|----------|-------|
| Analyze, feedback, clarify, decide | ✓ | ✓ | ✓ | ✓ |
| See L2 step text | – (masked to "Handled by <group>") | ✓ | ✓ | ✓ |
| See `restricted` tickets in search/similar | – | own group only | – | ✓ |
| See original IPs in bundle | – | ✓ | – | ✓ |
| Audit view | own analyses | group | – | all |
| Export audit | masked | masked | – | unmasked |
| Approve KB drafts | – | – | ✓ | ✓ |
| Manual-mode trial | ✓ | ✓ | – | ✓ |

Enforcement happens in three places:
1. **Retrieval:** `where` filters via `ctx.filters`
2. **Display:** `access.filter_response()`
3. **Export:** masking

Every denial is logged as an `access.denied` event.

### 8.7 Audit log (FR-08) (`backend/audit.py`)

- Append-only tables (§11.2). Each row of `audit_log` stores `prev_hash` and `row_hash = sha256(prev_hash + payload)`, which gives a tamper-evident chain. `GET /audit/verify` checks the chain.
- Each analysis records: the redacted input, signals, classification and its components, priority drivers, every evidence item (`kind, ref_id, rank, score, tier, excerpt, why`), the steps and citations, the decision, escalation reasons, approval reasons, model versions, timings, and the role the response was filtered for.
- Reviewer actions recorded: feedback, overrides, clarifications and the final decision.
- Prompts are stored only as hashes plus token counts, unless `audit.store_prompts=true`.
- **Export:** `GET /audit/export?format=csv|jsonl&from&to` (role-masked, with a watermark event).

### 8.8 Feedback (FR-06) (`backend/feedback.py`)

`POST /feedback`:

```json
{"analysis_id": "...", "issue_idx": 0,
 "target_type": "classification|similar_incident|kedb_match|step|recommendation",
 "target_ref": "...", "verdict": "useful|incorrect|unresolved",
 "corrected_value": "...", "comment": "..."}
```

Effects:
- `kb_prior()`: `fb(kb) = (useful − incorrect)/(useful + incorrect + 5)`, clipped to ±0.5, and passed to P2 on the next request.
- Classification corrections are stored, and P2 can read them via `GET /internal/corrections` to use as few-shot examples.
- A decision of `resolved` with notes calls `TicketStore.mark_verified()` (P1). The incident becomes an `analyst_verified` historical incident.
- `unresolved` or `incorrect` together with notes enables **Propose KB draft**.

### 8.9 KB draft loop (`backend/kb_drafts.py`)

1. `POST /kb/drafts {analysis_id}`: Groq drafts an article in the exact KEDB schema **from the analyst's notes only**. It is saved with `status=draft` and is not searchable.
2. `POST /kb/drafts/{id}/approve` (kb_owner): sets `status=approved`, `version+1`, `approved_by`, writes to `data/kb/articles_governed.json`, and calls P1's `reindex_article()`.

### 8.10 MTTR timing (FR-09, team-defined) (`backend/mttr.py`)

Event types, stored in `ticket_events(ticket_id, analysis_id, event, ts_utc, actor, mode)` with `mode ∈ {assisted, manual}`:
- `ticket.submitted`, `analysis.completed`, `ui.viewed`
- `clarify.requested`, `clarify.answered`
- `decision.submitted` (resolved, escalated or needs_more_info)
- `ticket.resolved`, `ticket.reopened`
- `manual.started`, `manual.decision`

| Metric | Definition |
|--------|------------|
| TTR (time to recommendation) | `analysis.completed − ticket.submitted` |
| TTD (time to decision, the trial measure) | `decision.submitted − ui.viewed` (assisted) / `manual.decision − manual.started` (manual) |
| MTTR | mean and median of `ticket.resolved − ticket.submitted`, grouped by category, priority, decision and mode |
| Historical MTTR (synthetic) | from enriched `opened_at/resolved_at`, labelled synthetic, used only as context |

`GET /metrics/mttr?group_by=category|priority|mode|decision&from&to` returns `{groups:[{key, n, mean_s, median_s, p90_s}]}`.

---

## 9. Analyst interface (P4)

### 9.1 Screens (`ui/`)

| Page | FR | Content |
|------|----|---------|
| Login / role | access | Login, or a role picker in demo mode |
| **1 Analyze** (assisted) | 01–06 | Ticket form or "Load sample" (DEMO-001…007, test tickets). Cards: **Classification** (confidence bar, band chip, components, alternatives), **Priority** (level, reason, drivers), **Similar incidents** (tier badge, score, excerpt, resolution with an origin badge `SYNTHETIC`/`VERIFIED`, why, collapsed keyword-only list), **KEDB match** (KB ID, title, matched code/field, excerpt), **Recommended solution** (numbered steps with source badges `[KB-0018·res.1]` / `[INC-H0007]`, L1/L2 tags, greyed "already tried" steps, source label), **Decision banner** (green recommend / amber clarify with an answer box / red escalate with reasons, group, tier and bundle download), **Approval badge**. Feedback buttons 👍 ✗ ? on every item. Final decision form (resolved/escalated/needs more info, notes). A "View JSON" toggle shows the contract and the full view. Secondary issue shown in a second tab. |
| **2 Manual mode** | 09 | Shows the trial ticket text only, plus embedded keyword search (tickets + KB) with **no AI fields**. The same decision form (category, subcategory, priority, KB/incident used, decision, notes). Hidden timer events. |
| **3 Search & filter** | 07 | Query plus filters: category, subcategory, priority, status, assignment group, application, environment, source, date range. Paginated table, row → detail drawer. |
| 4 Review queue | 05 | Clarify/escalate/needs-approval items, sorted by priority and age. Override labels and priority. |
| 5 KB browser | 03/06 | Article list and detail, coverage gaps, drafts and approval (kb_owner) |
| **6 Audit view** | 08 | Per-ticket timeline, evidence, decisions, hash-chain status, export button |
| **7 MTTR dashboard** | 09 | TTR/TTD/MTTR by category, priority, mode and decision. Trial results (manual vs assisted). Feedback rates. Degraded-mode counts. Historical synthetic baseline (labelled). |

`ui/client.py` wraps every endpoint in §10. It takes `API_BASE_URL` from the environment. `MOCK_API=1` serves `shared/fixtures/golden/*.json`, so the UI can be built before the backend exists.

### 9.2 UI rules

- Never show a step without its source badge.
- Synthetic data is always badged.
- The `ui.viewed` event is posted when the result renders.
- Session state holds `analysis_id`.
- Accessibility: colour plus text label on every chip.

### 9.3 Timed trial protocol (`trial/`, run by P4, analysed by P1)

1. **Participants:** ≥ 4 (all teammates, plus any volunteers, who are preferable). Each is assigned to group A or B.
2. **Crossover:** A does T01–T06 manual and T07–T12 assisted. B does the reverse. Order within each block is randomised (`trial/schedule.py`, seeded), and practice is done on 1 demo ticket first.
3. **Measures:**
   - TTD per ticket from events
   - **Decision correctness** vs `trial_gold.json`: category ✓, subcategory ✓, priority within ±1, correct KB or escalate decision
   - Self-reported confidence (1–5)
4. **Output:** `trial/results/raw_events.csv` → P1's `eval/mttr_report.py` → `docs/evaluation_report.md` §MTTR (median TTD manual vs assisted, % change, correctness both ways, n, caveats).
5. **Integrity:** participants who built a component skip tickets in its category if possible. This is noted as a limitation.

---

## 10. HTTP API (P3, `backend/main.py`, OpenAPI at `/docs`)

Base: `/api/v1`. Auth: `Authorization: Bearer <jwt>` (or `X-Demo-Role` when `DEMO_AUTH=1`).

| Method | Path | Role | Request | Response |
|--------|------|------|---------|----------|
| GET | `/health` | any | — | `{status, index_stale, llm_available, versions}` |
| POST | `/auth/login` | — | `{username,password}` | `{token, role, assignment_group}` |
| POST | `/analyze?view=full|contract` | l1+ | `TicketIn` | `AnalysisResponse` (§4) |
| POST | `/tickets/{ticket_id}/clarify` | l1+ | `{analysis_id, answer}` | `AnalysisResponse` (round 1) |
| GET | `/tickets/search` | l1+ | `q, category, subcategory, priority, status, assignment_group, application, environment, source, date_from, date_to, page, size` | `{items:[TicketHit], total}` |
| GET | `/tickets/{id}` | l1+ | — | `TicketRecord` (role-filtered) |
| GET | `/kb/search` | l1+ | `q, category, subcategory` | `{items:[KBHit]}` |
| GET | `/kb/{kb_id}` | l1+ | — | `KBArticle` |
| POST | `/feedback` | l1+ | §8.8 | `{feedback_id}` |
| POST | `/decisions` | l1+ | `{analysis_id, final_decision, final_category?, final_subcategory?, final_priority?, resolution_notes?, mode}` | `{ok}` |
| POST | `/events` | l1+ | `{ticket_id, analysis_id?, event, mode}` (UI events only: `ui.viewed`, `manual.started`, `manual.decision`) | `{ok}` |
| GET | `/review-queue` | l1+ | `status, page` | items |
| GET | `/escalations/{bundle_id}?fmt=json|md` | l2+ (l1 masked) | — | bundle |
| GET | `/audit/{analysis_id}` | per §8.6 | — | full audit record |
| GET | `/audit/export` | per §8.6 | `format, from, to` | file stream |
| GET | `/audit/verify` | admin | — | `{ok, broken_at?}` |
| GET | `/metrics/mttr` | l1+ | `group_by, from, to, mode` | §8.10 |
| GET | `/metrics/summary` | l1+ | — | counts, feedback rates, degraded rates |
| POST | `/kb/drafts` | l2+ | `{analysis_id}` | draft |
| POST | `/kb/drafts/{id}/approve` | kb_owner | `{edits?}` | article |
| GET | `/trial/tickets` | l1+ | — | trial tickets **without gold** |

Errors use `{"error": {"code": "...", "message": "...", "request_id": "..."}}` with these codes:
- 400 validation
- 401 unauthenticated
- 403 forbidden (logged)
- 404 not found
- 429 rate limited (internal)
- 503 index missing

---

## 11. Persistence

### 11.1 ChromaDB (`data/chroma/`, persistent, gitignored)

| Collection | Contents | Key metadata |
|------------|----------|--------------|
| `tickets_hist` | 1,800 enriched historical tickets | ticket_id, category, subcategory, priority, status, source, opened_ts, resolved_ts, assignment_group, application, environment, restricted, dup_group_id, text_hash, resolution_origin, resolution_kb_id |
| `tickets_verified` | ~60 curated + analyst-verified incidents | same + verified_by, verified_ts |
| `tickets_live` | Tickets analysed in the app | same + analysis_id, created_ts, live_status |
| `kedb_chunks` | ~300 KB chunks (approved only are searchable) | kb_id, chunk_type, category, subcategory, priority_hint, requires_admin, status, version, tags |
| `*_eval` | Copies built from the 70% split, for evaluation only | same |

The `resolution` text is stored in the document metadata (`resolution` field). The full record is also in `data/processed/*.json`.

### 11.2 SQLite (`data/app.db`, owned by P3; schema in `backend/db/schema.sql`)

```sql
CREATE TABLE tickets (ticket_id TEXT PRIMARY KEY, title TEXT, description_redacted TEXT,
  reporter_category TEXT, reporter_priority TEXT, application TEXT, environment TEXT, location TEXT,
  assignment_group TEXT, restricted INTEGER DEFAULT 0, status TEXT, created_at TEXT, updated_at TEXT, raw_hash TEXT);
CREATE TABLE analyses (analysis_id TEXT PRIMARY KEY, ticket_id TEXT, created_at TEXT, user TEXT, role TEXT,
  clarification_round INTEGER, contract_json TEXT, details_json TEXT, signals_json TEXT,
  decision TEXT, escalation_required INTEGER, human_approval_required INTEGER,
  pipeline_version TEXT, embed_model TEXT, llm_models TEXT, degraded TEXT, timings_json TEXT);
CREATE TABLE evidence (analysis_id TEXT, issue_idx INTEGER, kind TEXT, ref_id TEXT, rank INTEGER,
  score REAL, tier TEXT, excerpt TEXT, matched_field TEXT, why TEXT, origin TEXT);
CREATE TABLE feedback (feedback_id TEXT PRIMARY KEY, analysis_id TEXT, issue_idx INTEGER, target_type TEXT,
  target_ref TEXT, verdict TEXT, corrected_value TEXT, comment TEXT, user TEXT, created_at TEXT);
CREATE TABLE decisions (analysis_id TEXT PRIMARY KEY, mode TEXT, final_decision TEXT, final_category TEXT,
  final_subcategory TEXT, final_priority TEXT, kb_or_incident_used TEXT, resolution_notes TEXT,
  user TEXT, decided_at TEXT);
CREATE TABLE ticket_events (id INTEGER PRIMARY KEY AUTOINCREMENT, ticket_id TEXT, analysis_id TEXT,
  event TEXT, ts_utc TEXT, actor TEXT, mode TEXT);
CREATE TABLE audit_log (id INTEGER PRIMARY KEY AUTOINCREMENT, ts_utc TEXT, actor TEXT, action TEXT,
  entity TEXT, entity_id TEXT, payload_json TEXT, prev_hash TEXT, row_hash TEXT);
CREATE TABLE llm_calls (call_id TEXT, analysis_id TEXT, purpose TEXT, model TEXT, prompt_hash TEXT,
  tokens_in INTEGER, tokens_out INTEGER, latency_ms INTEGER, status TEXT, cached INTEGER, created_at TEXT);
CREATE TABLE kb_drafts (draft_id TEXT PRIMARY KEY, article_json TEXT, status TEXT, proposed_by TEXT,
  source_analysis_id TEXT, approved_by TEXT, created_at TEXT, approved_at TEXT);
CREATE TABLE overrides (analysis_id TEXT, field TEXT, old_value TEXT, new_value TEXT, user TEXT, ts_utc TEXT);
```

---

## 12. Groq usage, rate limits and degraded modes

| Call | When | Config key | Approx. tokens |
|------|------|-----------|----------------|
| Translate | offline only (already done) | `llm.translate_model` | batches of 20 |
| #1 Classify + split | per ticket | `llm.classify_model` (a small/fast model, e.g. `llama-3.1-8b-instant`) | ~1.2k in / 200 out |
| #2 Synthesise | per issue | `llm.synth_model` (a larger model, e.g. `llama-3.3-70b-versatile`) | ~2.5k in / 500 out |
| Curated history drafting | offline, P1 | `llm.synth_model` | — |
| KB draft | on request | `llm.synth_model` | ~1.5k / 700 |

> The model IDs above are examples. Groq's model list changes, so check the Groq console before the demo and set the IDs in `config/ai.yaml`.

**`ai/llm/groq_client.py`** (owned by P2, shared by everyone):
- JSON mode, `temperature=0`, 20 s timeout
- 2 retries on 429/5xx with exponential backoff and jitter, honouring `retry-after`
- a token-bucket limiter below the RPM/TPM limits
- a SQLite cache keyed by `sha1(model+prompt)` (`data/llm_cache.sqlite`)
- a circuit breaker that skips Groq for 60 s after 3 consecutive failures
- every call is logged to `llm_calls` through a callback injected by P3

| Degraded mode | Behaviour |
|---------------|-----------|
| `llm_classify_fallback` | kNN + rules, confidence capped at 0.80 |
| `llm_split_fallback` | regex split |
| `llm_synth_fallback` | KB steps rendered verbatim |
| `offline` (no key) | all of the above. The full flow and the demo still work. |

---

## 13. Security and privacy

| Concern | Control | Owner |
|---------|---------|-------|
| PII/secrets sent to the external LLM | `privacy.redact` runs before every Groq call and every embedding. Embeddings are local. | P1 + P3 |
| Secrets in storage | Only redacted text is persisted. `raw_hash` is used for integrity. Keys come from `.env` and are never logged. | P3 |
| Prompt injection | A delimited `<ticket>` block, schema validation, citations must exist, safety lint. Injection tickets are in the test set. | P2 |
| Unsafe advice | Safety blocklist. L2 tagging. No executable actions anywhere. | P2 |
| Access restrictions | RBAC (§8.6) at retrieval, display and export | P3 |
| Tamper evidence | Hash-chained audit log | P3 |
| Synthetic data disclosure | `source`, `resolution_origin` and `derived_fields` flags, UI badges, README and evaluation report statements | P1 + P4 |

---

## 14. Evaluation (P1, `eval/`)

### 14.1 Data

- **Split:** `tickets_enriched.json` is split 70/15/15, stratified by subcategory. Exact-duplicate groups stay in one split. The `*_eval` collections are built from the 70% split only.
- **Demo set:** 7 tickets with gold labels.
- **Test set:** ~25 candidate-created tickets (§6.3.3).
- **Gold KB:** demos + test set + 40 sampled test-split tickets (≈ 2 per subcategory), hand-labelled. Gap subcategories have gold = "escalate or verified incident".
- **Trial data:** §9.3.

### 14.2 Metrics

| Dimension (brief §4) | Metric | Target |
|----------------------|--------|--------|
| Classification | Category accuracy and macro-F1, subcategory macro-F1, confusion matrix (focus pairs: Oracle↔DB2↔MS SQL, VPN↔Wireless, CPU↔Memory, Email↔Internal App) | cat ≥ 0.90, subcat F1 ≥ 0.80 |
| Priority | Agreement with demo/test gold (±1 level). Sanity check against the synthetic baseline. | demo 7/7 within ±1 |
| Duplicate retrieval | Recall@5, MRR@10, nDCG@10 (relevance = same subcategory, plus a hand-labelled subset). Keyword-trap rate. | R@5 ≥ 0.85 |
| KB recommendation | Hit@1 and Hit@3 against gold KB | Hit@3 ≥ 0.85 |
| Resolution usefulness | Analyst rubric 1–5 (relevance, actionability, grounding) on ≥ 10 cases. Feedback `useful` rate. | mean ≥ 4 |
| Confidence & fallback | Decision accuracy (recommend/clarify/escalate) against the gold decision. Escalation recall on adversarial and gap tickets. Calibration (reliability curve, ECE). | escalation recall ≥ 0.9 |
| Traceability | % of displayed steps with a valid citation (must be 100%), unsupported steps removed, grounding distribution | 100% |
| Contract | % of responses valid against the JSON Schema | 100% |
| Privacy | Redaction recall on PII test cases. False-positive rate on error codes. | recall ≥ 0.95, FP = 0 on codes |
| **MTTR (FR-09)** | Trial: median TTD manual vs assisted, % change, decision correctness both ways. Live: TTR p50/p95. | report as measured |
| Ops | Latency per stage (p50/p95), Groq calls and tokens per ticket, degraded rate | p95 < 8 s |

### 14.3 Ablations

kNN-only, LLM-only and fused classifier; with and without BM25; with and without the error-code index; with and without the taxonomy boost; with and without verified incidents.

### 14.4 Outputs

`python -m eval.run --split test` writes:
- `eval/out/metrics.json`
- `eval/out/confusion_*.png`
- `eval/out/errors.jsonl`

`python -m eval.mttr_report` writes `eval/out/mttr.json` and its charts. P1 writes `docs/evaluation_report.md` (dataset, method, results, observed errors, limitations, the synthetic-data statement).

---

## 15. Requirement traceability

| Req | Requirement | Where it is satisfied | Owner |
|-----|-------------|-----------------------|-------|
| FR-01 | Category, subcategory, priority, intent + confidence | §3, §7.3, §7.4; contract `classification`, `priority` | P2 |
| FR-02 | Similar/duplicate ranking with title, description, symptoms, service, metadata | §6.5, §7.1, §7.5; contract `similar_incidents` | P2 (stores P1) |
| FR-03 | Resolution paths from resolved cases or approved knowledge | §6.3.2, §7.6, §7.7; contract `kedb_match`, `recommended_solution` | P2 |
| FR-04 | Source, excerpt, reason for each item | §6.4 chunks, `details.similar/kedb/steps`, §7.8 | P2 (+ P4 display) |
| FR-05 | Low-confidence flag, no unsupported conclusions, review/escalation path | §8.2–8.5, `escalation_required`, `human_approval_required`, clarify loop, bundle, review queue | P3 |
| FR-06 | Useful / incorrect / unresolved feedback | §8.8, UI buttons | P3 (+ P4 UI) |
| FR-07 | Filters: application, category, date, status, assignment group, environment | §6.3.1 derived metadata, §7.5 filters, `/tickets/search`, Search page | P4 (P2 search fn, P1 metadata) |
| FR-08 | Capture input, prediction, evidence, recommendation, decision | §8.7, §11.2, Audit view, export | P3 (+ P4 UI) |
| FR-09 *(team-defined)* | Measure handling time, assisted vs manual | §8.10 events, §9.3 trial, §14.2 | P3 timing · P4 trial · P1 report |
| Access control (brief §5) | Respect restrictions in retrieval, display, export | §8.6 | P3 |
| Privacy (brief §5) | Sensitive content | §6.2, §13 | P1 |
| Data rules (brief §5) | Generated labels and assumptions identified | §6.3 flags, §18 | P1 |
| Submission: prototype | API + UI | §9, §10 | P3 + P4 |
| Submission: evaluation report | `docs/evaluation_report.md` | §14 | P1 |
| Submission: ≥ 3 worked examples | `docs/examples/*.md` exported from the audit view | §16 | P4 |
| Submission: README | setup, run, architecture summary, assumptions, trade-offs | `README.md` | P4 |

---

## 16. Demo walkthrough (expected behaviour; actual scores are measured in evaluation)

| Ticket | Classification | Priority (level · reason) | KEDB / similar | Decision → contract booleans |
|--------|----------------|---------------------------|----------------|------------------------------|
| DEMO-001 VPN "Gateway not responding" | Network/VPN, high | **High** · "deadline today; floor: VPN baseline" | KB-0018 via exact phrase code. "Restart client" is marked already attempted. | recommend → esc **false**, approval **true** (High) |
| DEMO-002 Oracle slow, ORA-01013, 15 users, month-end | Database/Oracle, high | **Critical** · "15 users affected; month-end close blocked" | KB-0001 via code hit + symptom; alternative KB-0002 | E4 hard → **escalate** to DBA, L3 → esc **true**, approval **true** |
| DEMO-003 Monitor blacks out after update | Hardware/Monitor, high | **Medium** · "Baseline for Monitor" | KB-0008. The cable step is marked attempted. Driver/refresh-rate steps follow. | recommend → esc **false**, approval **false** |
| DEMO-004 Outlook crash, 0xc0000005 | Software/Email, high | **High** · "end-of-day deadline; floor: KB-0029 hint P2" | KB-0029 via "Faulting application name"; alternative KB-0030 | recommend → esc **false**, approval **true** (requires_admin) |
| DEMO-005 VPN drops + overheating | Primary Network/VPN; secondary Hardware/CPU | **High** · "losing unsaved work; floor: VPN baseline" | Issue 1 → KB-0018. Issue 2 → KB-0030, with KB-0006 2nd. Cross-link cites KB-0019 diag. | Issue 2: E6 (KB-0006 "the laptop overheats") → escalate EUC-HW. Ticket → esc **true**, approval **true**. |
| DEMO-006 Wi-Fi drops, Building B 2nd floor | Network/Wireless, high | **High** · "multiple colleagues affected; floor: Wireless baseline" | KB-0021 via symptom match | E6 ("multiple managed devices disconnect") → escalate NET-OPS → esc **true**, approval **true** |
| DEMO-007 MS SQL 18456, new joiner | Database/MS SQL Server, high | **High** · "Baseline for MS SQL Server" | KB-0003 ranks above KB-0004 (shared code 18456; "expired" is absent) | recommend → esc **false**, approval **true** (L2 role grant, DBA) |

The golden fixtures `shared/fixtures/golden/DEMO-00x.json` encode these expectations. Golden tests assert `category`, `subcategory`, `kb_id`, `decision` and the two booleans; scores are not asserted.

---

## 17. Repository layout and configuration

```
it-ticket-assistant/
├── shared/                    # ALL · contracts (frozen after Phase 0)
│   ├── schemas.py             # Pydantic: TicketIn, TicketRecord, KBArticle, Signals, Classification, Priority,
│   │                          #   SimilarIncident, KedbMatch, RecommendedSolution, AnalysisContract, Details, AnalysisResponse,
│   │                          #   TicketHit, ChunkHit, KBHit, Feedback, Decision, Event, User
│   ├── interfaces.py          # Protocols: TicketStore, KBStore, LLMClient, Engine, Redactor
│   ├── constants.py           # taxonomy, PRIORITY_LEVEL, roles, thresholds defaults
│   ├── stubs.py               # FakeTicketStore, FakeKBStore, FakeLLM, FakeEngine, fake_redact
│   ├── contract/analysis_output.schema.json
│   └── fixtures/{golden/, mini_tickets.json, mini_articles.json}
├── privacy/      (P1) redact.py
├── ingest/       (P1) preprocess.py enrich.py index_tickets.py index_kedb.py build_all.py make_test_sets.py
├── store/        (P1) chroma_store.py bm25_store.py embedder.py
├── eval/         (P1) run.py metrics.py mttr_report.py gold/ out/
├── ai/           (P2) engine.py signals.py split.py classify.py priority.py ticket_search.py kb_search.py
│                      synthesize.py validate.py llm/groq_client.py prompts/*.j2
├── backend/      (P3) main.py routers/ pipeline.py intake.py decision.py escalate.py approval.py assemble.py
│                      access.py auth.py audit.py feedback.py mttr.py kb_drafts.py db/schema.sql db/sqlite.py
├── ui/           (P4) app.py client.py mock_api.py pages/ components/
├── trial/        (P4) schedule.py protocol.md results/
├── config/       base.yaml(ALL) data.yaml privacy.yaml taxonomy_map.yaml(P1) ai.yaml rules.yaml signals.yaml(P2)
│                 backend.yaml escalation.yaml kb_escalation_clauses.yaml clarify_templates.yaml users.yaml(P3) ui.yaml(P4)
├── data/         raw/ processed/ kb/ samples/ seed/(P1) test/(P1) chroma/ app.db llm_cache.sqlite (last three gitignored)
├── docs/         architecture.md evaluation_report.md(P1) examples/(P4) brief.docx
├── tests/        p1_data/ p2_ai/ p3_backend/ p4_ui/ contract/ integration/
├── requirements/ base.txt data.txt ai.txt backend.txt ui.txt dev.txt
├── mk/           p1.mk p2.mk p3.mk p4.mk     # included by Makefile
├── .github/      workflows/ci.yml  CODEOWNERS  pull_request_template.md
├── Makefile  .env.example  pyproject.toml  CLAUDE.md  README.md
```

**Key config defaults** (`config/ai.yaml` and `config/backend.yaml`):

```yaml
embedding: {model: BAAI/bge-small-en-v1.5, normalize: true}
llm: {classify_model: llama-3.1-8b-instant, synth_model: llama-3.3-70b-versatile, temperature: 0,
      timeout_s: 20, max_retries: 2, rpm: 25, cache: data/llm_cache.sqlite}
classifier: {weights: {llm: 0.45, knn: 0.35, rule: 0.20}, knn_k: 15, bands: {high: 0.75, medium: 0.55}}
retrieval:
  tickets: {k_dense: 30, k_bm25: 30, rrf_k: 60, top_n: 5, rerank: false, related_min: 0.55}
  kb: {k_chunks: 20, top_n: 3, tau_show: 0.35, tau_kb: 0.50, ambiguity_gap: 0.05}
  weights_kb: {sem: 0.45, bm25: 0.15, code: 0.20, tax: 0.12, fb: 0.08}
duplicates: {near: 0.92, similar: 0.75, live_window_h: 72, cluster_min: 3}
validator: {grounding_min: 0.30, min_steps: 2}
decision: {sparse_words: 12, low_gap: 0.10, max_clarify_rounds: 1, verified_min_cos: 0.75}
audit: {store_prompts: false}
```

---

## 18. Assumptions, trade-offs and limitations

**Assumptions** (repeated in the README and evaluation report)
1. Categories and subcategories in `tickets.json` are human labels. The English text is machine-translated by Groq. Priority is a synthetic rule-based label.
2. The 30 KEDB articles are approved knowledge.
3. All resolutions, timestamps, application, environment and assignment group values on historical tickets are **synthetic** (§6.3) and flagged as such. Curated incidents are team-written and team-reviewed.
4. `kedb_match.error_code` is the KEDB known-error ID (e.g. `KB-0018`). The literal error code found in the ticket is in `details.kedb.matched_error_code`.
5. FR-09 (MTTR) is a team-defined requirement, not part of the brief.
6. Trial participants are few and partly the builders, so the MTTR result is indicative, not statistically conclusive.

**Trade-offs**

| Decision | Alternative | Why |
|----------|-------------|-----|
| FastAPI + Streamlit over HTTP | Streamlit calling the core directly | Parallel development against a stable contract; API-level audit and RBAC; a headless API for judges |
| Fused kNN + LLM + rules classifier | LLM only / fine-tuning | Works offline, explainable confidence, survives free-tier outages; 1,800 short translated texts would overfit a fine-tune |
| KB-derived synthetic resolutions are shown but never cited | Cite them | Citing them would be circular evidence and would inflate the "Historical Incident" source |
| Curated verified incidents | No history source | Fills KB gaps and makes the "Historical Incident" source real and reviewable |
| Three-way decision with one clarify round | Binary escalate | Matches FR-05 (clarification request); bounded loop |
| Field-level KB chunks | Whole-article embedding | Precise excerpts (FR-04) |
| Mapping as a soft boost | Hard filter | Different taxonomies and classifier errors would hide correct articles |
| SQLite + Chroma | Postgres + pgvector | Zero ops; store interfaces allow a later swap |
| 2 LLM calls per ticket | Agent loop | Predictable latency and quota |

**Limitations**
- KB gaps (DHCP, IP Address, Disk, DB2) escalate often. This is by design, and they are reported as a KB backlog.
- Historical tickets are short and uniform, so retrieval metrics on them are optimistic. The test and adversarial sets partly correct for this.
- Priority accuracy against synthetic labels is only a sanity check.
- Translation artefacts affect kNN. `original_it` is kept for audit.
- Thresholds come from a small validation set and need recalibration with real feedback.
