-- GAMI Enhancement: Procedure Extraction (Hermes-style)
-- Stores learned procedures from successful multi-step interactions
-- Run: psql -h 127.0.0.1 -p 5433 -U gami -d gami -f 017_procedures.sql

BEGIN;

-- Procedures table for learned workflows
CREATE TABLE IF NOT EXISTS procedures (
    id BIGSERIAL PRIMARY KEY,
    procedure_id TEXT UNIQUE DEFAULT gen_random_uuid()::text,
    owner_tenant_id TEXT NOT NULL,

    -- Identity
    name TEXT NOT NULL,
    description TEXT,
    category TEXT,  -- deployment, debugging, configuration, maintenance, etc.

    -- Structured content
    trigger_patterns JSONB DEFAULT '[]',   -- Array of regex/semantic triggers
    preconditions JSONB DEFAULT '[]',      -- Required state before execution
    steps JSONB NOT NULL,                  -- Array of {order, action, params, expected}
    postconditions JSONB DEFAULT '[]',     -- Expected state after execution

    -- Parameters (Hermes-style templating)
    parameters JSONB DEFAULT '[]',  -- Array of {name, type, description, default, required}

    -- Embedding for semantic matching
    embedding vector(768),
    canonical_text TEXT,  -- Full procedure text for embedding

    -- Learning/Trust
    execution_count INTEGER DEFAULT 0,
    success_count INTEGER DEFAULT 0,
    failure_count INTEGER DEFAULT 0,
    success_rate FLOAT GENERATED ALWAYS AS (
        CASE WHEN execution_count > 0
        THEN success_count::FLOAT / execution_count
        ELSE 0.5 END
    ) STORED,

    -- Provenance
    source_session_ids TEXT[],  -- Sessions where this was observed
    source_segment_ids TEXT[],  -- Segments that contributed
    confidence FLOAT DEFAULT 0.5,

    status TEXT DEFAULT 'active',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_procedures_tenant
    ON procedures(owner_tenant_id) WHERE status = 'active';
CREATE INDEX IF NOT EXISTS idx_procedures_category
    ON procedures(category) WHERE status = 'active';
CREATE INDEX IF NOT EXISTS idx_procedures_success
    ON procedures(success_rate DESC) WHERE status = 'active';
CREATE UNIQUE INDEX IF NOT EXISTS uq_procedures_tenant_name
    ON procedures(owner_tenant_id, LOWER(name));

COMMIT;
