-- GAMI Innovation Extensions - Direct SQL Migration
-- Run with: psql -p 5433 -U gami -d gami -f 013_innovation_tables.sql
-- Creates all tables for Phase 13 Innovation Extensions

BEGIN;

-- =============================================================================
-- 1. RETRIEVAL LOGS - Learning Signal Collection
-- =============================================================================
CREATE TABLE IF NOT EXISTS retrieval_logs (
    id BIGSERIAL PRIMARY KEY,
    log_id TEXT NOT NULL UNIQUE DEFAULT gen_random_uuid()::text,
    session_id TEXT,
    query_text TEXT NOT NULL,
    query_embedding vector(768),
    query_mode TEXT,

    segments_returned TEXT[],
    scores_returned FLOAT[],

    outcome_type TEXT,
    outcome_signal FLOAT,
    correction_text TEXT,

    tenant_id TEXT NOT NULL,
    agent_id TEXT,
    user_id TEXT,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    processed_in_dream BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_retrieval_logs_session ON retrieval_logs(session_id);
CREATE INDEX IF NOT EXISTS idx_retrieval_logs_unprocessed ON retrieval_logs(processed_in_dream)
    WHERE NOT processed_in_dream;
CREATE INDEX IF NOT EXISTS idx_retrieval_logs_tenant ON retrieval_logs(tenant_id, created_at DESC);

-- =============================================================================
-- 2. AGENT CONFIGS - Per-Agent Credentials & Settings
-- =============================================================================
CREATE TABLE IF NOT EXISTS agent_configs (
    id SERIAL PRIMARY KEY,
    agent_id TEXT NOT NULL UNIQUE,
    owner_tenant_id TEXT NOT NULL,

    agent_name TEXT NOT NULL,
    agent_type TEXT DEFAULT 'assistant',
    personality_json JSONB DEFAULT '{}',

    default_model TEXT,
    endpoint_url TEXT,
    credentials_encrypted BYTEA,
    credential_key_id TEXT,

    default_temperature FLOAT DEFAULT 0.7,
    default_max_tokens INTEGER DEFAULT 2048,
    context_window_size INTEGER DEFAULT 8192,

    -- Provider Configuration
    llm_provider TEXT DEFAULT 'vllm',  -- vllm, ollama, openai, anthropic
    embedding_provider TEXT DEFAULT 'sentence_transformers',  -- sentence_transformers, ollama, openai
    embedding_model TEXT DEFAULT 'nomic-ai/nomic-embed-text-v1.5',
    embedding_device TEXT DEFAULT 'auto',  -- auto, cpu, cuda, mps

    system_prompt_override TEXT,
    extraction_prompt_overrides JSONB DEFAULT '{}',
    scoring_overrides JSONB DEFAULT '{}',

    rate_limit_rpm INTEGER DEFAULT 60,
    token_budget_daily INTEGER DEFAULT 1000000,
    tokens_used_today INTEGER DEFAULT 0,
    budget_reset_at TIMESTAMPTZ,

    accuracy_score FLOAT DEFAULT 0.5,
    verified_claims INTEGER DEFAULT 0,
    disputed_claims INTEGER DEFAULT 0,
    total_claims INTEGER DEFAULT 0,

    status TEXT DEFAULT 'active',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_configs_tenant ON agent_configs(owner_tenant_id);
CREATE INDEX IF NOT EXISTS idx_agent_configs_status ON agent_configs(status) WHERE status = 'active';

-- =============================================================================
-- 3. PROMPT TEMPLATES - Configurable Prompts
-- =============================================================================
CREATE TABLE IF NOT EXISTS prompt_templates (
    id SERIAL PRIMARY KEY,
    template_id TEXT NOT NULL,
    tenant_id TEXT DEFAULT '*',
    agent_id TEXT,

    template_name TEXT NOT NULL,
    template_type TEXT NOT NULL,

    system_prompt TEXT,
    user_prompt_template TEXT,

    temperature FLOAT,
    max_tokens INTEGER,
    model_override TEXT,

    output_format TEXT DEFAULT 'json',
    output_schema JSONB,

    version INTEGER DEFAULT 1,
    is_active BOOLEAN DEFAULT TRUE,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT uq_prompt_template UNIQUE(template_id, tenant_id, agent_id, version)
);

CREATE INDEX IF NOT EXISTS idx_prompt_templates_lookup ON prompt_templates(template_type, tenant_id, agent_id)
    WHERE is_active;

-- =============================================================================
-- 4. CAUSAL RELATIONS - Cause-Effect Links
-- =============================================================================
CREATE TABLE IF NOT EXISTS causal_relations (
    id BIGSERIAL PRIMARY KEY,
    causal_id TEXT NOT NULL UNIQUE DEFAULT gen_random_uuid()::text,
    owner_tenant_id TEXT NOT NULL,

    cause_entity_id TEXT,
    cause_text TEXT NOT NULL,
    effect_entity_id TEXT,
    effect_text TEXT NOT NULL,

    causal_type TEXT NOT NULL,

    cause_timestamp TIMESTAMPTZ,
    effect_timestamp TIMESTAMPTZ,
    temporal_valid BOOLEAN,
    temporal_gap_hours FLOAT,

    explicitness_score FLOAT DEFAULT 0.5,
    authority_score FLOAT DEFAULT 0.5,
    corroboration_count INTEGER DEFAULT 0,
    strength_score FLOAT GENERATED ALWAYS AS (
        0.4 * COALESCE(explicitness_score, 0.5) +
        0.3 * COALESCE(authority_score, 0.5) +
        0.3 * LEAST(1.0, COALESCE(corroboration_count, 0) / 3.0)
    ) STORED,

    source_segment_id TEXT,
    extraction_pattern TEXT,
    extraction_method TEXT DEFAULT 'pattern',

    created_by_agent_id TEXT,

    status TEXT DEFAULT 'active',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_causal_cause ON causal_relations(cause_entity_id) WHERE status = 'active';
CREATE INDEX IF NOT EXISTS idx_causal_effect ON causal_relations(effect_entity_id) WHERE status = 'active';
CREATE INDEX IF NOT EXISTS idx_causal_tenant ON causal_relations(owner_tenant_id);
CREATE INDEX IF NOT EXISTS idx_causal_strength ON causal_relations(strength_score DESC) WHERE status = 'active';

-- =============================================================================
-- 5. MEMORY CLUSTERS - Consolidation Groups
-- =============================================================================
CREATE TABLE IF NOT EXISTS memory_clusters (
    id BIGSERIAL PRIMARY KEY,
    cluster_id TEXT NOT NULL UNIQUE DEFAULT gen_random_uuid()::text,
    owner_tenant_id TEXT NOT NULL,

    cluster_name TEXT,
    cluster_embedding vector(768),
    centroid_segment_id TEXT,

    member_ids TEXT[] NOT NULL,
    member_count INTEGER DEFAULT 0,

    abstraction_text TEXT,
    abstraction_embedding vector(768),

    stability_score FLOAT DEFAULT 0.5,
    repetition_count INTEGER DEFAULT 1,
    last_reinforced_at TIMESTAMPTZ,

    last_accessed_at TIMESTAMPTZ DEFAULT NOW(),
    access_count INTEGER DEFAULT 0,
    decay_rate FLOAT DEFAULT 0.01,
    current_decay FLOAT DEFAULT 1.0,

    inference_ids TEXT[],

    status TEXT DEFAULT 'active',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_memory_clusters_tenant ON memory_clusters(owner_tenant_id);
CREATE INDEX IF NOT EXISTS idx_memory_clusters_stability ON memory_clusters(stability_score DESC)
    WHERE status = 'active';

-- =============================================================================
-- 6. SESSIONS - Conversation State Tracking
-- =============================================================================
CREATE TABLE IF NOT EXISTS sessions (
    id BIGSERIAL PRIMARY KEY,
    session_id TEXT NOT NULL UNIQUE,
    tenant_id TEXT NOT NULL,
    agent_id TEXT,
    user_id TEXT,

    conversation_state TEXT DEFAULT 'idle',
    state_confidence FLOAT DEFAULT 0.5,
    state_history JSONB DEFAULT '[]',

    active_entities TEXT[],
    active_topics TEXT[],
    hot_context_ids TEXT[],

    message_count INTEGER DEFAULT 0,
    retrieval_count INTEGER DEFAULT 0,
    learning_signals_positive INTEGER DEFAULT 0,
    learning_signals_negative INTEGER DEFAULT 0,

    started_at TIMESTAMPTZ DEFAULT NOW(),
    last_activity_at TIMESTAMPTZ DEFAULT NOW(),
    ended_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_sessions_tenant ON sessions(tenant_id, last_activity_at DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_agent ON sessions(agent_id) WHERE agent_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_sessions_active ON sessions(last_activity_at DESC)
    WHERE ended_at IS NULL;

-- =============================================================================
-- 7. SUBCONSCIOUS EVENTS - Daemon Audit Trail
-- =============================================================================
CREATE TABLE IF NOT EXISTS subconscious_events (
    id BIGSERIAL PRIMARY KEY,
    event_id TEXT NOT NULL UNIQUE DEFAULT gen_random_uuid()::text,
    session_id TEXT,

    event_type TEXT NOT NULL,

    detected_state TEXT,
    state_confidence FLOAT,
    previous_state TEXT,

    predicted_entities TEXT[],
    predicted_topics TEXT[],
    preloaded_segment_ids TEXT[],
    prediction_confidence FLOAT,

    injected_context TEXT,
    injection_token_count INTEGER,
    injection_accepted BOOLEAN,

    latency_ms INTEGER,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_subconscious_session ON subconscious_events(session_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_subconscious_type ON subconscious_events(event_type, created_at DESC);

-- =============================================================================
-- 8. AGENT TRUST HISTORY
-- =============================================================================
CREATE TABLE IF NOT EXISTS agent_trust_history (
    id BIGSERIAL PRIMARY KEY,
    agent_id TEXT NOT NULL,
    accuracy_score FLOAT,
    verified_claims INTEGER,
    disputed_claims INTEGER,
    recorded_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_trust_agent ON agent_trust_history(agent_id, recorded_at DESC);

-- =============================================================================
-- 9. ATTRIBUTION COLUMNS ON EXISTING TABLES
-- =============================================================================
ALTER TABLE segments ADD COLUMN IF NOT EXISTS created_by_agent_id TEXT;
ALTER TABLE segments ADD COLUMN IF NOT EXISTS created_by_user_id TEXT;
ALTER TABLE segments ADD COLUMN IF NOT EXISTS derived_from TEXT[];
ALTER TABLE segments ADD COLUMN IF NOT EXISTS derivation_type TEXT;
ALTER TABLE segments ADD COLUMN IF NOT EXISTS stability_score FLOAT DEFAULT 0.5;
ALTER TABLE segments ADD COLUMN IF NOT EXISTS decay_score FLOAT DEFAULT 1.0;
ALTER TABLE segments ADD COLUMN IF NOT EXISTS cluster_id TEXT;
ALTER TABLE segments ADD COLUMN IF NOT EXISTS last_reinforced_at TIMESTAMPTZ;

ALTER TABLE entities ADD COLUMN IF NOT EXISTS created_by_agent_id TEXT;
ALTER TABLE entities ADD COLUMN IF NOT EXISTS created_by_user_id TEXT;

ALTER TABLE claims ADD COLUMN IF NOT EXISTS created_by_agent_id TEXT;
ALTER TABLE claims ADD COLUMN IF NOT EXISTS created_by_user_id TEXT;
ALTER TABLE claims ADD COLUMN IF NOT EXISTS derived_from TEXT[];
ALTER TABLE claims ADD COLUMN IF NOT EXISTS derivation_type TEXT;

ALTER TABLE relations ADD COLUMN IF NOT EXISTS created_by_agent_id TEXT;

ALTER TABLE assistant_memories ADD COLUMN IF NOT EXISTS created_by_agent_id TEXT;
ALTER TABLE assistant_memories ADD COLUMN IF NOT EXISTS created_by_user_id TEXT;
ALTER TABLE assistant_memories ADD COLUMN IF NOT EXISTS derived_from TEXT[];
ALTER TABLE assistant_memories ADD COLUMN IF NOT EXISTS cluster_id TEXT;

-- =============================================================================
-- 10. VECTOR INDEXES
-- =============================================================================
CREATE INDEX IF NOT EXISTS idx_memory_clusters_embedding
    ON memory_clusters USING ivfflat (cluster_embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_memory_clusters_abstraction
    ON memory_clusters USING ivfflat (abstraction_embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_retrieval_logs_embedding
    ON retrieval_logs USING ivfflat (query_embedding vector_cosine_ops)
    WITH (lists = 100);

-- =============================================================================
-- 11. SEED DEFAULT PROMPTS
-- =============================================================================
INSERT INTO prompt_templates (template_id, tenant_id, template_name, template_type, system_prompt, user_prompt_template, temperature, max_tokens)
VALUES
    ('entity_default', '*', 'Default Entity Extraction', 'entity_extraction',
     'You are a precise entity extraction system. Return ONLY valid JSON arrays.',
     'Extract entities from:\n\n{{ text }}\n\nReturn JSON array of {name, type, description}.',
     0.1, 2048),
    ('claim_default', '*', 'Default Claim Extraction', 'claim_extraction',
     'You are a precise claim extraction system. Return ONLY valid JSON arrays.',
     'Extract factual claims from:\n\n{{ text }}\n\nReturn JSON array of {subject, predicate, object, confidence, modality}.',
     0.1, 2048),
    ('relation_default', '*', 'Default Relation Extraction', 'relation_extraction',
     'You are a precise relation extraction system. Return ONLY valid JSON arrays.',
     'Extract relationships from:\n\n{{ text }}\n\nReturn JSON array of {source, target, relation_type, strength}.',
     0.1, 2048),
    ('causal_default', '*', 'Default Causal Extraction', 'causal_extraction',
     'You are a causal relationship extraction system. Identify cause-effect relationships.',
     'Extract causal relationships from:\n\n{{ text }}\n\nReturn JSON array of {cause, effect, causal_type, confidence}.',
     0.1, 2048),
    ('state_classify', '*', 'State Classification', 'state_classification',
     'Classify the conversation state into one of: debugging, planning, recalling, exploring, executing, idle.',
     'Recent messages:\n{% for msg in messages %}{{ msg }}\n{% endfor %}\n\nClassify the state and confidence (0-1).',
     0.3, 100),
    ('abstraction', '*', 'Memory Abstraction', 'abstraction_generation',
     'You summarize groups of related memories into a single coherent abstraction.',
     'Create a brief abstraction from these related memories:\n\n{% for t in texts %}{{ t }}\n---\n{% endfor %}',
     0.3, 500),
    ('event_default', '*', 'Default Event Extraction', 'event_extraction',
     'You extract temporal events and their timestamps from text.',
     'Extract events from:\n\n{{ text }}\n\nReturn JSON array of {event, timestamp, actors, location}.',
     0.1, 2048),
    ('summary_default', '*', 'Default Summarization', 'summarization',
     'You create concise summaries preserving key information.',
     'Summarize:\n\n{{ text }}\n\nProvide a concise summary.',
     0.3, 500),
    ('procedure_default', '*', 'Default Procedure Extraction', 'procedure_extraction',
     'You extract step-by-step procedures and instructions.',
     'Extract procedures from:\n\n{{ text }}\n\nReturn JSON array of {title, steps: [{step, action}], prerequisites}.',
     0.1, 2048),
    ('importance_default', '*', 'Default Importance Scoring', 'importance_scoring',
     'You assess the importance and relevance of information segments.',
     'Rate importance of:\n\n{{ text }}\n\nProvide score 0-1 and reasoning.',
     0.2, 200)
ON CONFLICT (template_id, tenant_id, agent_id, version) DO NOTHING;

COMMIT;

-- Verify
SELECT 'Tables created: ' || COUNT(*)::text FROM information_schema.tables
WHERE table_schema = 'public'
AND table_name IN ('retrieval_logs', 'agent_configs', 'prompt_templates', 'causal_relations',
                   'memory_clusters', 'sessions', 'subconscious_events', 'agent_trust_history');

SELECT 'Prompts seeded: ' || COUNT(*)::text FROM prompt_templates;
