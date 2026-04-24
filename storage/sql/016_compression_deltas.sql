-- GAMI Enhancement: Lossless Compression Support
-- Stores unique facts (deltas) not captured in cluster abstractions
-- Run: psql -h 127.0.0.1 -p 5433 -U gami -d gami -f 016_compression_deltas.sql

BEGIN;

-- Delta storage for lossless compression
CREATE TABLE IF NOT EXISTS compression_deltas (
    id BIGSERIAL PRIMARY KEY,
    delta_id TEXT UNIQUE DEFAULT gen_random_uuid()::text,

    -- Link to source segment and cluster
    segment_id TEXT NOT NULL,
    cluster_id TEXT NOT NULL,

    -- Unique facts not captured in abstraction
    unique_facts JSONB NOT NULL,  -- [{fact, importance, category}]
    unique_facts_text TEXT,       -- Concatenated for search
    embedding vector(768),

    -- Metrics
    fact_count INTEGER DEFAULT 0,
    total_importance FLOAT DEFAULT 0.0,
    compression_ratio FLOAT,

    -- Provenance
    extracted_by TEXT DEFAULT 'dream_compress',
    extraction_confidence FLOAT DEFAULT 0.7,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for compression_deltas
CREATE INDEX IF NOT EXISTS idx_deltas_segment
    ON compression_deltas(segment_id);
CREATE INDEX IF NOT EXISTS idx_deltas_cluster
    ON compression_deltas(cluster_id);
CREATE INDEX IF NOT EXISTS idx_deltas_importance
    ON compression_deltas(total_importance DESC);

-- Compression tracking table (since we can't modify segments owned by postgres)
-- This tracks which segments have been processed for compression
CREATE TABLE IF NOT EXISTS segment_compression_status (
    segment_id TEXT PRIMARY KEY,
    compression_status TEXT DEFAULT 'raw',  -- raw, compressed, abstracted
    delta_id TEXT,                          -- Link to compression_deltas
    cluster_id TEXT,                        -- Which cluster this belongs to
    processed_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_seg_comp_status
    ON segment_compression_status(compression_status);
CREATE INDEX IF NOT EXISTS idx_seg_comp_cluster
    ON segment_compression_status(cluster_id);

COMMIT;
