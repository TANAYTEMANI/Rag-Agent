-- Run once: psql -U rag -d ragdb -f db/migrations.sql

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ── Documents ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS documents (
    doc_id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    file_name       TEXT NOT NULL,
    file_hash       VARCHAR(64) UNIQUE NOT NULL,
    doc_type        VARCHAR(50),
    department      VARCHAR(100),
    language        VARCHAR(10),
    page_count      INTEGER,
    tags            TEXT[],
    superseded_by   UUID REFERENCES documents(doc_id),
    ingestion_ts    TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_doc_type    ON documents(doc_type);
CREATE INDEX IF NOT EXISTS idx_department  ON documents(department);
CREATE INDEX IF NOT EXISTS idx_tags        ON documents USING GIN(tags);

-- ── Ingestion jobs ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ingestion_jobs (
    doc_id          UUID PRIMARY KEY REFERENCES documents(doc_id),
    file_name       TEXT NOT NULL,
    file_hash       VARCHAR(64),
    status          VARCHAR(20) NOT NULL DEFAULT 'queued',
    page_count      INTEGER,
    chunk_count     INTEGER DEFAULT 0,
    error_message   TEXT,
    queued_at       TIMESTAMP DEFAULT NOW(),
    started_at      TIMESTAMP,
    completed_at    TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_job_status ON ingestion_jobs(status);

-- ── Tabular tables (pandas engine) ───────────────────────────────────────
CREATE TABLE IF NOT EXISTS tabular_tables (
    table_id        TEXT PRIMARY KEY,
    doc_id          UUID REFERENCES documents(doc_id),
    file_name       TEXT,
    sheet_name      TEXT,
    parquet_path    TEXT NOT NULL,
    row_count       INTEGER,
    col_count       INTEGER,
    columns         TEXT[],
    description     TEXT,
    created_at      TIMESTAMP DEFAULT NOW()
);
