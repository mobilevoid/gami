-- Migration 014: Multi-Manifold Embeddings
-- Creates infrastructure for true multi-dimensional manifold architecture.
--
-- To apply: psql -h 127.0.0.1 -p 5433 -U gami -d gami -f 014_multi_manifold_embeddings.sql

BEGIN;

-- Enable vector extension if not already enabled
CREATE EXTENSION IF NOT EXISTS vector;

-- ===========================================================================
-- 1. MANIFOLD EMBEDDINGS TABLE
-- ===========================================================================
CREATE TABLE IF NOT EXISTS manifold_embeddings (
    id BIGSERIAL PRIMARY KEY,
    target_id TEXT NOT NULL,
    target_type TEXT NOT NULL,  -- segment, claim, entity, memory
    manifold_type TEXT NOT NULL,  -- TOPIC, CLAIM, PROCEDURE, RELATION, TIME, EVIDENCE

    -- Primary embedding (768d for topic/claim/procedure)
    embedding vector(768),

    -- The text that was embedded
    canonical_form TEXT,

    -- Embedding metadata
    embedding_model TEXT DEFAULT 'nomic-embed-text',
    embedding_version INTEGER DEFAULT 1,

    -- Quality score
    confidence_score FLOAT DEFAULT 0.5,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT uq_manifold_embeddings_target_manifold
        UNIQUE(target_id, target_type, manifold_type)
);

CREATE INDEX IF NOT EXISTS idx_me_target ON manifold_embeddings(target_id, target_type);
CREATE INDEX IF NOT EXISTS idx_me_manifold_type ON manifold_embeddings(manifold_type);

-- ===========================================================================
-- 2. CANONICAL CLAIMS TABLE
-- ===========================================================================
CREATE TABLE IF NOT EXISTS canonical_claims (
    id BIGSERIAL PRIMARY KEY,
    claim_id TEXT NOT NULL UNIQUE,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT,
    modality TEXT DEFAULT 'factual',
    temporal_scope TEXT,
    canonical_text TEXT NOT NULL,
    confidence FLOAT DEFAULT 0.5,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_canonical_claims_subject ON canonical_claims(subject);

-- ===========================================================================
-- 3. CANONICAL PROCEDURES TABLE
-- ===========================================================================
CREATE TABLE IF NOT EXISTS canonical_procedures (
    id BIGSERIAL PRIMARY KEY,
    procedure_id TEXT NOT NULL UNIQUE,
    action TEXT NOT NULL,
    target TEXT,
    steps_json JSONB DEFAULT '[]',
    canonical_text TEXT NOT NULL,
    confidence FLOAT DEFAULT 0.5,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_canonical_procedures_action ON canonical_procedures(action);

-- ===========================================================================
-- 4. TEMPORAL FEATURES TABLE
-- ===========================================================================
CREATE TABLE IF NOT EXISTS temporal_features (
    id BIGSERIAL PRIMARY KEY,
    target_id TEXT NOT NULL,
    target_type TEXT NOT NULL,
    features FLOAT[] NOT NULL,  -- 12-dimensional feature vector
    has_timestamp BOOLEAN DEFAULT FALSE,
    timestamp_value TIMESTAMPTZ,
    temporal_keywords TEXT[],
    created_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT uq_temporal_features_target UNIQUE(target_id, target_type)
);

-- ===========================================================================
-- 5. GRAPH FINGERPRINTS TABLE
-- ===========================================================================
CREATE TABLE IF NOT EXISTS graph_fingerprints (
    id BIGSERIAL PRIMARY KEY,
    target_id TEXT NOT NULL,
    target_type TEXT NOT NULL,
    fingerprint FLOAT[] NOT NULL,  -- 64-dimensional fingerprint
    in_degree INTEGER DEFAULT 0,
    out_degree INTEGER DEFAULT 0,
    edge_types_json JSONB DEFAULT '{}',
    computed_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT uq_graph_fingerprints_target UNIQUE(target_id, target_type)
);

-- ===========================================================================
-- 6. ADD COLUMNS TO SEGMENTS (requires superuser or table owner)
-- ===========================================================================
-- NOTE: Run these as postgres user if needed:
-- ALTER TABLE segments ADD COLUMN IF NOT EXISTS manifold_status TEXT DEFAULT 'pending';
-- ALTER TABLE segments ADD COLUMN IF NOT EXISTS manifolds_populated TEXT[] DEFAULT '{}';

COMMIT;

-- ===========================================================================
-- 7. CREATE IVFFLAT INDEXES FOR MANIFOLD EMBEDDINGS
-- ===========================================================================
-- These are created outside the transaction for CONCURRENTLY support
-- Run after initial data population for better index quality

-- Topic manifold index
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_me_topic_vec
    ON manifold_embeddings USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100) WHERE manifold_type = 'TOPIC';

-- Claim manifold index
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_me_claim_vec
    ON manifold_embeddings USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100) WHERE manifold_type = 'CLAIM';

-- Procedure manifold index
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_me_procedure_vec
    ON manifold_embeddings USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100) WHERE manifold_type = 'PROCEDURE';

-- Relation manifold index
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_me_relation_vec
    ON manifold_embeddings USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100) WHERE manifold_type = 'RELATION';

-- Time manifold index
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_me_time_vec
    ON manifold_embeddings USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100) WHERE manifold_type = 'TIME';

-- Evidence manifold index
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_me_evidence_vec
    ON manifold_embeddings USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100) WHERE manifold_type = 'EVIDENCE';
