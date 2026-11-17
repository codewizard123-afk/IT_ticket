-- SQLite schema for data/app.db (owner: P3). Source of truth: docs/architecture.md §11.2.
-- Idempotent: every statement uses IF NOT EXISTS so init_db() can run on every start-up.
-- Only redacted ticket text is stored (architecture §13); raw_hash keeps integrity of the original.

CREATE TABLE IF NOT EXISTS tickets (
  ticket_id TEXT PRIMARY KEY, title TEXT, description_redacted TEXT,
  reporter_category TEXT, reporter_priority TEXT, application TEXT, environment TEXT,
  location TEXT, assignment_group TEXT, restricted INTEGER DEFAULT 0, status TEXT,
  created_at TEXT, updated_at TEXT, raw_hash TEXT
);

CREATE TABLE IF NOT EXISTS analyses (
  analysis_id TEXT PRIMARY KEY, ticket_id TEXT, created_at TEXT, user TEXT, role TEXT,
  clarification_round INTEGER, contract_json TEXT, details_json TEXT, signals_json TEXT,
  decision TEXT, escalation_required INTEGER, human_approval_required INTEGER,
  pipeline_version TEXT, embed_model TEXT, llm_models TEXT, degraded TEXT, timings_json TEXT
);

CREATE TABLE IF NOT EXISTS evidence (
  analysis_id TEXT, issue_idx INTEGER, kind TEXT, ref_id TEXT, rank INTEGER,
  score REAL, tier TEXT, excerpt TEXT, matched_field TEXT, why TEXT, origin TEXT
);

CREATE TABLE IF NOT EXISTS feedback (
  feedback_id TEXT PRIMARY KEY, analysis_id TEXT, issue_idx INTEGER, target_type TEXT,
  target_ref TEXT, verdict TEXT, corrected_value TEXT, comment TEXT, user TEXT, created_at TEXT
);

CREATE TABLE IF NOT EXISTS decisions (
  analysis_id TEXT PRIMARY KEY, mode TEXT, final_decision TEXT, final_category TEXT,
  final_subcategory TEXT, final_priority TEXT, kb_or_incident_used TEXT, resolution_notes TEXT,
  user TEXT, decided_at TEXT
);

CREATE TABLE IF NOT EXISTS ticket_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ticket_id TEXT, analysis_id TEXT,
  event TEXT, ts_utc TEXT, actor TEXT, mode TEXT
);

CREATE TABLE IF NOT EXISTS audit_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts_utc TEXT, actor TEXT, action TEXT,
  entity TEXT, entity_id TEXT, payload_json TEXT, prev_hash TEXT, row_hash TEXT
);

CREATE TABLE IF NOT EXISTS llm_calls (
  call_id TEXT, analysis_id TEXT, purpose TEXT, model TEXT, prompt_hash TEXT,
  tokens_in INTEGER, tokens_out INTEGER, latency_ms INTEGER, status TEXT, cached INTEGER,
  created_at TEXT
);

CREATE TABLE IF NOT EXISTS kb_drafts (
  draft_id TEXT PRIMARY KEY, article_json TEXT, status TEXT, proposed_by TEXT,
  source_analysis_id TEXT, approved_by TEXT, created_at TEXT, approved_at TEXT
);

CREATE TABLE IF NOT EXISTS overrides (
  analysis_id TEXT, field TEXT, old_value TEXT, new_value TEXT, user TEXT, ts_utc TEXT
);

-- Lookup indexes (not in §11.2's DDL, purely additive for the read paths in §10).
CREATE INDEX IF NOT EXISTS ix_analyses_ticket ON analyses (ticket_id);
CREATE INDEX IF NOT EXISTS ix_evidence_analysis ON evidence (analysis_id);
CREATE INDEX IF NOT EXISTS ix_feedback_analysis ON feedback (analysis_id);
CREATE INDEX IF NOT EXISTS ix_events_ticket ON ticket_events (ticket_id);
CREATE INDEX IF NOT EXISTS ix_events_analysis ON ticket_events (analysis_id);
CREATE INDEX IF NOT EXISTS ix_audit_entity ON audit_log (entity, entity_id);
CREATE INDEX IF NOT EXISTS ix_llm_calls_analysis ON llm_calls (analysis_id);

-- audit_log is append-only (architecture §8.7). These triggers stop accidental edits from app
-- code; deliberate tampering with the file is caught by the hash chain (backend/audit.py).
CREATE TRIGGER IF NOT EXISTS audit_log_no_update BEFORE UPDATE ON audit_log
BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END;

CREATE TRIGGER IF NOT EXISTS audit_log_no_delete BEFORE DELETE ON audit_log
BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END;
