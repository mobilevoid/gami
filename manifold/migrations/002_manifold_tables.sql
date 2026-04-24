-- Migration 002: Multi-Manifold Memory System Tables
--
-- WARNING: This migration is NOT in the Alembic chain.
-- It must be run explicitly during manifold system activation.
--
-- To apply: psql -p 5433 -U gami -d gami -f 002_manifold_tables.sql
-- To rollback: psql -p 5433 -U gami -d gami -f rollback_002.sql
--
-- Prerequisites:
-- - PostgreSQL 16 with pgvector extension
-- - GAMI base tables already exist (sources, segments, entities, claims, etc.)

BEGIN;

-- Enable vector extension if not already enabled
CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------------------
-- Manifold Embeddings Table
-- ---------------------------------------------------------------------------
-- Stores multi-manifold embeddings for objects.
-- Each object can have multiple embeddings, one per manifold type.

CREATE TABLE IF NOT EXISTS manifold_embeddings (
    id BIGSERIAL PRIMARY KEY,
    target_id TEXT NOT NULL,
    target_type TEXT NOT NULL,  -- 'segment', 'claim', 'entity', 'summary', 'memory'
    manifold_type TEXT NOT NULL,  -- 'topic', 'claim', 'procedure'

    -- The embedding vector (768d for nomic-embed-text)
    embedding vector(768),

    -- Embedding metadata
    embedding_model TEXT NOT NULL DEFAULT 'nomic-embed-text',
    embedding_version INTEGER NOT NULL DEFAULT 1,

    -- The text that was actually embedded (for debugging/verification)
    canonical_form TEXT,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT uq_manifold_embeddings_target_manifold
        UNIQUE(target_id, target_type, manifold_type)
);

-- Indexes for manifold_embeddings
CREATE INDEX IF NOT EXISTS idx_manifold_embeddings_target
    ON manifold_embeddings(target_id, target_type);
CREATE INDEX IF NOT EXISTS idx_manifold_embeddings_type
    ON manifold_embeddings(manifold_type);

-- Vector indexes per manifold (IVFFlat for approximate nearest neighbor)
-- Note: These require sufficient data to build. Run AFTER backfill.
-- CREATE INDEX idx_manifold_topic_vec ON manifold_embeddings
--     USING ivfflat (embedding vector_cosine_ops)
--     WHERE manifold_type = 'topic';
-- CREATE INDEX idx_manifold_claim_vec ON manifold_embeddings
--     USING ivfflat (embedding vector_cosine_ops)
--     WHERE manifold_type = 'claim';
-- CREATE INDEX idx_manifold_procedure_vec ON manifold_embeddings
--     USING ivfflat (embedding vector_cosine_ops)
--     WHERE manifold_type = 'procedure';


-- ---------------------------------------------------------------------------
-- Canonical Claims Table
-- ---------------------------------------------------------------------------
-- Stores structured SPO form of claims for the claim manifold.

CREATE TABLE IF NOT EXISTS canonical_claims (
    id BIGSERIAL PRIMARY KEY,
    claim_id TEXT NOT NULL UNIQUE,

    -- SPO structure
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT,

    -- Modifiers
    modality TEXT DEFAULT 'factual',  -- factual, possible, negated
    qualifiers JSONB DEFAULT '[]'::jsonb,
    temporal_scope TEXT,

    -- Full canonical text for embedding
    canonical_text TEXT NOT NULL,

    confidence FLOAT DEFAULT 0.5,
    extraction_method TEXT DEFAULT 'llm',

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_canonical_claims_subject
    ON canonical_claims(subject);
CREATE INDEX IF NOT EXISTS idx_canonical_claims_predicate
    ON canonical_claims(predicate);


-- ---------------------------------------------------------------------------
-- Canonical Procedures Table
-- ---------------------------------------------------------------------------
-- Stores structured form of procedures for the procedure manifold.

CREATE TABLE IF NOT EXISTS canonical_procedures (
    id BIGSERIAL PRIMARY KEY,
    source_id TEXT,
    segment_id TEXT,

    title TEXT NOT NULL,
    prerequisites JSONB DEFAULT '[]'::jsonb,
    steps JSONB NOT NULL,  -- Array of {order, text, optional}
    expected_outcome TEXT,

    -- Full canonical text for embedding
    canonical_text TEXT NOT NULL,

    owner_tenant_id TEXT NOT NULL,
    confidence FLOAT DEFAULT 0.5,

    created_at TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT uq_canonical_procedures_source_title
        UNIQUE(source_id, segment_id, title)
);

CREATE INDEX IF NOT EXISTS idx_canonical_procedures_tenant
    ON canonical_procedures(owner_tenant_id);
CREATE INDEX IF NOT EXISTS idx_canonical_procedures_title
    ON canonical_procedures(title);


-- ---------------------------------------------------------------------------
-- Temporal Features Table
-- ---------------------------------------------------------------------------
-- Stores structured temporal features for the time manifold.

CREATE TABLE IF NOT EXISTS temporal_features (
    id BIGSERIAL PRIMARY KEY,
    target_id TEXT NOT NULL,
    target_type TEXT NOT NULL,

    -- 12-dimensional feature vector
    features FLOAT[12] NOT NULL,

    -- Original text that was parsed
    raw_temporal_text TEXT,

    -- Parsed dates
    start_date DATE,
    end_date DATE,

    is_range BOOLEAN DEFAULT FALSE,
    precision TEXT DEFAULT 'day',  -- year, month, day, hour
    is_relative BOOLEAN DEFAULT FALSE,

    created_at TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT uq_temporal_features_target
        UNIQUE(target_id, target_type)
);

CREATE INDEX IF NOT EXISTS idx_temporal_features_target
    ON temporal_features(target_id, target_type);
CREATE INDEX IF NOT EXISTS idx_temporal_features_dates
    ON temporal_features(start_date, end_date);


-- ---------------------------------------------------------------------------
-- Promotion Scores Table
-- ---------------------------------------------------------------------------
-- Stores promotion scores for deciding manifold treatment.

CREATE TABLE IF NOT EXISTS promotion_scores (
    id BIGSERIAL PRIMARY KEY,
    target_id TEXT NOT NULL,
    target_type TEXT NOT NULL,

    -- Individual score components
    importance FLOAT DEFAULT 0.5,
    retrieval_frequency FLOAT DEFAULT 0.0,
    source_diversity FLOAT DEFAULT 0.0,
    confidence FLOAT DEFAULT 0.5,
    novelty FLOAT DEFAULT 0.5,
    graph_centrality FLOAT DEFAULT 0.0,
    user_relevance FLOAT DEFAULT 0.0,

    -- Computed total score
    total_score FLOAT GENERATED ALWAYS AS (
        0.25 * importance +
        0.20 * retrieval_frequency +
        0.15 * source_diversity +
        0.15 * confidence +
        0.10 * novelty +
        0.10 * graph_centrality +
        0.05 * user_relevance
    ) STORED,

    promotion_status TEXT DEFAULT 'raw',  -- raw, provisional, promoted, manifold

    computed_at TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT uq_promotion_scores_target
        UNIQUE(target_id, target_type)
);

CREATE INDEX IF NOT EXISTS idx_promotion_scores_status
    ON promotion_scores(promotion_status);
CREATE INDEX IF NOT EXISTS idx_promotion_scores_total
    ON promotion_scores(total_score DESC);


-- ---------------------------------------------------------------------------
-- Query Logs Table
-- ---------------------------------------------------------------------------
-- Stores query logs for retrieval frequency tracking.

CREATE TABLE IF NOT EXISTS query_logs (
    id BIGSERIAL PRIMARY KEY,

    query_text TEXT NOT NULL,
    query_hash TEXT NOT NULL,

    query_mode TEXT,
    tenant_ids TEXT[],

    -- Results returned
    result_ids TEXT[],
    result_scores FLOAT[],

    latency_ms INTEGER,

    -- Manifold weights used
    manifold_weights JSONB,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_query_logs_hash
    ON query_logs(query_hash);
CREATE INDEX IF NOT EXISTS idx_query_logs_time
    ON query_logs(created_at DESC);


-- ---------------------------------------------------------------------------
-- Manifold Config Table
-- ---------------------------------------------------------------------------
-- Per-tenant manifold configuration.

CREATE TABLE IF NOT EXISTS manifold_config (
    id SERIAL PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    config_key TEXT NOT NULL,
    config_value JSONB NOT NULL,
    description TEXT,

    updated_at TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT uq_manifold_config_tenant_key
        UNIQUE(tenant_id, config_key)
);


-- ---------------------------------------------------------------------------
-- Shadow Comparisons Table
-- ---------------------------------------------------------------------------
-- Stores shadow comparison results for A/B testing.

CREATE TABLE IF NOT EXISTS shadow_comparisons (
    id BIGSERIAL PRIMARY KEY,

    query_hash TEXT NOT NULL,
    query_text TEXT,

    -- Old (v1) results
    old_result_ids TEXT[],
    old_scores FLOAT[],

    -- New (v2) results
    new_result_ids TEXT[],
    new_scores FLOAT[],

    -- Comparison metrics
    overlap_count INTEGER,
    rank_correlation FLOAT,

    latency_old_ms INTEGER,
    latency_new_ms INTEGER,

    winner TEXT,  -- 'old', 'new', 'tie', 'unknown'
    notes TEXT,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_shadow_comparisons_time
    ON shadow_comparisons(created_at DESC);


-- ---------------------------------------------------------------------------
-- Insert default configuration
-- ---------------------------------------------------------------------------

INSERT INTO manifold_config (tenant_id, config_key, config_value, description) VALUES
    ('*', 'manifold_enabled', 'false', 'Global manifold system enable flag'),
    ('*', 'shadow_mode', 'false', 'Enable shadow comparison mode'),
    ('*', 'v2_retrieval', 'false', 'Use v2 manifold retrieval as primary'),
    ('*', 'promotion_threshold_provisional', '0.45', 'Threshold for provisional promotion'),
    ('*', 'promotion_threshold_promoted', '0.70', 'Threshold for full promotion'),
    ('*', 'promotion_threshold_manifold', '0.85', 'Threshold for all-manifold treatment')
ON CONFLICT (tenant_id, config_key) DO NOTHING;


COMMIT;

-- ---------------------------------------------------------------------------
-- Post-migration notes
-- ---------------------------------------------------------------------------
--
-- After running this migration:
-- 1. Run migrate_topic_embeddings.py to copy existing embeddings
-- 2. Run migrate_canonical_claims.py to canonicalize existing claims
-- 3. Run compute_promotion_scores.py to calculate promotion scores
-- 4. Build IVFFlat indexes after sufficient data exists
-- 5. Enable manifold_enabled in manifold_config when ready
