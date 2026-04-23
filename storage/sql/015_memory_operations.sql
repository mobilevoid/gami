-- GAMI Enhancement: Memory Operations Tracking
-- Tracks ADD/UPDATE/DELETE/NOOP decisions for Mem0-style consolidation
-- Run: psql -h 127.0.0.1 -p 5433 -U gami -d gami -f 015_memory_operations.sql

BEGIN;

-- Memory operation tracking table
CREATE TABLE IF NOT EXISTS memory_operations (
    id BIGSERIAL PRIMARY KEY,
    operation_id TEXT UNIQUE DEFAULT gen_random_uuid()::text,
    operation_type TEXT NOT NULL,  -- ADD, UPDATE, DELETE, NOOP

    -- New memory details
    new_memory_text TEXT NOT NULL,
    new_memory_embedding vector(768),

    -- Target memory (for UPDATE/DELETE/NOOP)
    target_memory_id TEXT,
    target_memory_text TEXT,
    similarity_score FLOAT,

    -- Decision details
    llm_decision TEXT,       -- Raw LLM response
    llm_confidence FLOAT,
    decision_reason TEXT,

    -- Execution tracking
    executed BOOLEAN DEFAULT FALSE,
    executed_at TIMESTAMPTZ,
    result_memory_id TEXT,   -- ID of created/updated memory

    -- Tenant and timing
    tenant_id TEXT NOT NULL,
    agent_id TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_memory_ops_tenant
    ON memory_operations(tenant_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_memory_ops_type
    ON memory_operations(operation_type);
CREATE INDEX IF NOT EXISTS idx_memory_ops_unexecuted
    ON memory_operations(executed) WHERE NOT executed;

-- Statistics view
CREATE OR REPLACE VIEW memory_operation_stats AS
SELECT
    tenant_id,
    operation_type,
    count(*) as count,
    avg(similarity_score) as avg_similarity,
    avg(llm_confidence) as avg_confidence,
    count(*) FILTER (WHERE executed) as executed_count,
    max(created_at) as last_operation
FROM memory_operations
GROUP BY tenant_id, operation_type;

COMMIT;
