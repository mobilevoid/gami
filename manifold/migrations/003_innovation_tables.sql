-- Migration 003: Innovation Extension Tables
--
-- Adds tables for:
-- 1. Learning retrieval (retrieval_logs)
-- 2. Per-agent configuration (agent_configs)
-- 3. Configurable prompts (prompt_templates)
-- 4. Causal extraction (causal_relations)
-- 5. Memory consolidation (memory_clusters)
-- 6. Session tracking (sessions)
-- 7. Subconscious daemon audit (subconscious_events)
--
-- To apply: psql -p 5433 -U gami -d gami -f 003_innovation_tables.sql
-- To rollback: psql -p 5433 -U gami -d gami -f rollback_003.sql
--
-- NOTE: Column additions to existing tables (section 8) require postgres user
-- Run as postgres: psql -p 5433 -U postgres -d gami -f 003_innovation_tables_alter.sql

-- No transaction wrapper - allows partial success

-- ===========================================================================
-- 1. RETRIEVAL LOGS - Learning Signal Collection
-- ===========================================================================

CREATE TABLE IF NOT EXISTS retrieval_logs (
    id BIGSERIAL PRIMARY KEY,
    log_id TEXT NOT NULL UNIQUE DEFAULT gen_random_uuid()::text,
    session_id TEXT,
    query_text TEXT NOT NULL,
    query_embedding vector(768),
    query_mode TEXT,

    -- Results returned
    segments_returned TEXT[],
    scores_returned FLOAT[],
    latency_ms INTEGER,

    -- Outcome signals (populated by subsequent user actions)
    outcome_type TEXT,  -- 'confirmed', 'continued', 'corrected', 'rephrased', 'ignored'
    outcome_signal FLOAT,  -- -1.0 to 1.0
    correction_text TEXT,
    outcome_recorded_at TIMESTAMPTZ,

    -- Attribution
    tenant_id TEXT NOT NULL,
    agent_id TEXT,
    user_id TEXT,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    processed_in_dream BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_retrieval_logs_session
    ON retrieval_logs(session_id) WHERE session_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_retrieval_logs_unprocessed
    ON retrieval_logs(created_at DESC) WHERE NOT processed_in_dream AND outcome_signal IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_retrieval_logs_tenant
    ON retrieval_logs(tenant_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_retrieval_logs_agent
    ON retrieval_logs(agent_id) WHERE agent_id IS NOT NULL;


-- ===========================================================================
-- 2. AGENT CONFIGS - Per-Agent Credentials & Settings
-- ===========================================================================

CREATE TABLE IF NOT EXISTS agent_configs (
    id SERIAL PRIMARY KEY,
    agent_id TEXT NOT NULL UNIQUE,
    owner_tenant_id TEXT NOT NULL,

    -- Identity
    agent_name TEXT NOT NULL,
    agent_type TEXT DEFAULT 'assistant',  -- assistant, tool, background, human
    personality_json JSONB DEFAULT '{}'::jsonb,

    -- LLM Configuration
    default_model TEXT,
    endpoint_url TEXT,
    credentials_encrypted BYTEA,  -- AES-256-GCM encrypted
    credential_key_id TEXT,

    -- Default Parameters
    default_temperature FLOAT DEFAULT 0.7,
    default_max_tokens INTEGER DEFAULT 2048,
    context_window_size INTEGER DEFAULT 8192,

    -- Prompt Overrides
    system_prompt_override TEXT,
    extraction_prompt_overrides JSONB DEFAULT '{}'::jsonb,

    -- Scoring Overrides (merged with defaults)
    scoring_overrides JSONB DEFAULT '{}'::jsonb,

    -- Rate Limits
    rate_limit_rpm INTEGER DEFAULT 60,
    token_budget_daily INTEGER DEFAULT 1000000,
    tokens_used_today INTEGER DEFAULT 0,
    budget_reset_at TIMESTAMPTZ DEFAULT NOW() + INTERVAL '1 day',

    -- Trust Metrics (updated by dream cycle)
    accuracy_score FLOAT DEFAULT 0.5,
    verified_claims INTEGER DEFAULT 0,
    disputed_claims INTEGER DEFAULT 0,
    total_claims INTEGER DEFAULT 0,

    status TEXT DEFAULT 'active',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_configs_tenant
    ON agent_configs(owner_tenant_id);
CREATE INDEX IF NOT EXISTS idx_agent_configs_status
    ON agent_configs(status) WHERE status = 'active';


-- Agent trust history for trending
CREATE TABLE IF NOT EXISTS agent_trust_history (
    id BIGSERIAL PRIMARY KEY,
    agent_id TEXT NOT NULL,
    accuracy_score FLOAT,
    verified_claims INTEGER,
    disputed_claims INTEGER,
    recorded_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_trust_history_agent
    ON agent_trust_history(agent_id, recorded_at DESC);


-- ===========================================================================
-- 3. PROMPT TEMPLATES - Configurable Prompts
-- ===========================================================================

CREATE TABLE IF NOT EXISTS prompt_templates (
    id SERIAL PRIMARY KEY,
    template_id TEXT NOT NULL,
    tenant_id TEXT DEFAULT '*',  -- '*' = global default
    agent_id TEXT,  -- NULL = tenant default

    template_name TEXT NOT NULL,
    template_type TEXT NOT NULL,  -- entity_extraction, claim_extraction,
                                   -- relation_extraction, event_extraction,
                                   -- summarization, causal_extraction,
                                   -- state_classification, abstraction, etc.

    -- Template Content (Jinja2)
    system_prompt TEXT,
    user_prompt_template TEXT,

    -- Output Configuration
    output_format TEXT DEFAULT 'json',  -- json, text, structured
    output_schema JSONB,  -- JSON Schema for validation

    -- LLM Parameters
    temperature FLOAT,
    max_tokens INTEGER,
    model_override TEXT,

    -- Versioning
    version INTEGER DEFAULT 1,
    is_active BOOLEAN DEFAULT TRUE,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Unique constraint using expression index
CREATE UNIQUE INDEX IF NOT EXISTS uq_prompt_template
    ON prompt_templates(template_id, tenant_id, COALESCE(agent_id, ''), version);

CREATE INDEX IF NOT EXISTS idx_prompt_templates_lookup
    ON prompt_templates(template_type, tenant_id, agent_id) WHERE is_active;
CREATE INDEX IF NOT EXISTS idx_prompt_templates_type
    ON prompt_templates(template_type) WHERE is_active;


-- ===========================================================================
-- 4. CAUSAL RELATIONS - Cause-Effect Links
-- ===========================================================================

CREATE TABLE IF NOT EXISTS causal_relations (
    id BIGSERIAL PRIMARY KEY,
    causal_id TEXT NOT NULL UNIQUE DEFAULT gen_random_uuid()::text,
    owner_tenant_id TEXT NOT NULL,

    -- Causal Structure
    cause_entity_id TEXT,
    cause_text TEXT NOT NULL,
    effect_entity_id TEXT,
    effect_text TEXT NOT NULL,

    -- Causal Type
    causal_type TEXT NOT NULL,  -- because, caused_by, resulted_in, led_to,
                                 -- enabled, prevented, triggered, preceded

    -- Temporal Validation
    cause_timestamp TIMESTAMPTZ,
    effect_timestamp TIMESTAMPTZ,
    temporal_valid BOOLEAN,  -- TRUE = cause precedes effect
    temporal_gap_hours FLOAT,

    -- Strength Scoring (objective, measurable)
    explicitness_score FLOAT DEFAULT 0.5,  -- how explicitly stated
    authority_score FLOAT DEFAULT 0.5,      -- source authority
    corroboration_count INTEGER DEFAULT 0,

    -- Computed strength (0.4*explicitness + 0.3*authority + 0.3*corroboration)
    strength_score FLOAT GENERATED ALWAYS AS (
        0.4 * COALESCE(explicitness_score, 0.5) +
        0.3 * COALESCE(authority_score, 0.5) +
        0.3 * LEAST(1.0, COALESCE(corroboration_count, 0)::float / 3.0)
    ) STORED,

    -- Provenance
    source_segment_id TEXT,
    extraction_pattern TEXT,
    extraction_method TEXT DEFAULT 'pattern',  -- pattern, llm, intervention

    -- Attribution
    created_by_agent_id TEXT,

    status TEXT DEFAULT 'active',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_causal_cause
    ON causal_relations(cause_entity_id) WHERE status = 'active';
CREATE INDEX IF NOT EXISTS idx_causal_effect
    ON causal_relations(effect_entity_id) WHERE status = 'active';
CREATE INDEX IF NOT EXISTS idx_causal_tenant
    ON causal_relations(owner_tenant_id);
CREATE INDEX IF NOT EXISTS idx_causal_strength
    ON causal_relations(strength_score DESC) WHERE status = 'active';
CREATE INDEX IF NOT EXISTS idx_causal_type
    ON causal_relations(causal_type) WHERE status = 'active';


-- ===========================================================================
-- 5. MEMORY CLUSTERS - Consolidation Groups
-- ===========================================================================

CREATE TABLE IF NOT EXISTS memory_clusters (
    id BIGSERIAL PRIMARY KEY,
    cluster_id TEXT NOT NULL UNIQUE DEFAULT gen_random_uuid()::text,
    owner_tenant_id TEXT NOT NULL,

    -- Cluster Identity
    cluster_name TEXT,
    cluster_embedding vector(768),
    centroid_segment_id TEXT,  -- most representative member

    -- Members
    member_ids TEXT[] NOT NULL,
    member_count INTEGER DEFAULT 0,

    -- Abstraction (merged representation)
    abstraction_text TEXT,
    abstraction_embedding vector(768),

    -- Stability & Learning
    stability_score FLOAT DEFAULT 0.5,
    repetition_count INTEGER DEFAULT 1,  -- how many times pattern seen
    last_reinforced_at TIMESTAMPTZ,

    -- Decay
    last_accessed_at TIMESTAMPTZ DEFAULT NOW(),
    access_count INTEGER DEFAULT 0,
    decay_rate FLOAT DEFAULT 0.01,  -- per day
    current_decay FLOAT DEFAULT 1.0,  -- 1.0 = full strength

    -- Inferences Generated
    inference_ids TEXT[],

    status TEXT DEFAULT 'active',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_memory_clusters_tenant
    ON memory_clusters(owner_tenant_id);
CREATE INDEX IF NOT EXISTS idx_memory_clusters_stability
    ON memory_clusters(stability_score DESC) WHERE status = 'active';

-- Vector index for cluster similarity (create after data exists)
-- CREATE INDEX idx_memory_clusters_vec ON memory_clusters
--     USING ivfflat (cluster_embedding vector_cosine_ops);


-- ===========================================================================
-- 6. SESSIONS - Conversation State Tracking
-- ===========================================================================

CREATE TABLE IF NOT EXISTS sessions (
    id BIGSERIAL PRIMARY KEY,
    session_id TEXT NOT NULL UNIQUE,
    tenant_id TEXT NOT NULL,
    agent_id TEXT,
    user_id TEXT,

    -- State Machine
    conversation_state TEXT DEFAULT 'idle',  -- idle, debugging, planning,
                                              -- recalling, exploring, executing
    state_confidence FLOAT DEFAULT 0.5,
    state_history JSONB DEFAULT '[]'::jsonb,  -- [{state, timestamp, confidence}]

    -- Active Context
    active_entities TEXT[],
    active_topics TEXT[],
    hot_context_ids TEXT[],  -- pre-loaded segment IDs

    -- Metrics
    message_count INTEGER DEFAULT 0,
    retrieval_count INTEGER DEFAULT 0,
    learning_signals_positive INTEGER DEFAULT 0,
    learning_signals_negative INTEGER DEFAULT 0,

    -- Lifecycle
    started_at TIMESTAMPTZ DEFAULT NOW(),
    last_activity_at TIMESTAMPTZ DEFAULT NOW(),
    ended_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_sessions_tenant
    ON sessions(tenant_id, last_activity_at DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_agent
    ON sessions(agent_id) WHERE agent_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_sessions_active
    ON sessions(last_activity_at DESC) WHERE ended_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_sessions_state
    ON sessions(conversation_state) WHERE ended_at IS NULL;


-- ===========================================================================
-- 7. SUBCONSCIOUS EVENTS - Daemon Audit Trail
-- ===========================================================================

CREATE TABLE IF NOT EXISTS subconscious_events (
    id BIGSERIAL PRIMARY KEY,
    event_id TEXT NOT NULL UNIQUE DEFAULT gen_random_uuid()::text,
    session_id TEXT,

    event_type TEXT NOT NULL,  -- state_change, prediction, preload,
                                -- injection, cache_update

    -- State Classification
    detected_state TEXT,
    state_confidence FLOAT,
    previous_state TEXT,

    -- Predictive Retrieval
    predicted_entities TEXT[],
    predicted_topics TEXT[],
    preloaded_segment_ids TEXT[],
    prediction_confidence FLOAT,

    -- Context Injection
    injected_context TEXT,
    injection_token_count INTEGER,
    injection_accepted BOOLEAN,

    -- Performance
    latency_ms INTEGER,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_subconscious_session
    ON subconscious_events(session_id, created_at DESC) WHERE session_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_subconscious_type
    ON subconscious_events(event_type, created_at DESC);


-- ===========================================================================
-- 8. COLUMN ADDITIONS TO EXISTING TABLES
-- NOTE: These require table ownership. Run as postgres user if gami doesn't own tables.
-- ===========================================================================

-- Attribution columns for segments (run as owner)
DO $$
BEGIN
    ALTER TABLE segments ADD COLUMN IF NOT EXISTS created_by_agent_id TEXT;
    ALTER TABLE segments ADD COLUMN IF NOT EXISTS created_by_user_id TEXT;
    ALTER TABLE segments ADD COLUMN IF NOT EXISTS derived_from TEXT[];
    ALTER TABLE segments ADD COLUMN IF NOT EXISTS derivation_type TEXT;
    ALTER TABLE segments ADD COLUMN IF NOT EXISTS stability_score FLOAT DEFAULT 0.5;
    ALTER TABLE segments ADD COLUMN IF NOT EXISTS decay_score FLOAT DEFAULT 1.0;
    ALTER TABLE segments ADD COLUMN IF NOT EXISTS cluster_id TEXT;
    ALTER TABLE segments ADD COLUMN IF NOT EXISTS last_reinforced_at TIMESTAMPTZ;
EXCEPTION WHEN insufficient_privilege THEN
    RAISE NOTICE 'segments columns require postgres user - skipping';
END $$;

-- Attribution columns for entities
DO $$
BEGIN
    ALTER TABLE entities ADD COLUMN IF NOT EXISTS created_by_agent_id TEXT;
    ALTER TABLE entities ADD COLUMN IF NOT EXISTS created_by_user_id TEXT;
EXCEPTION WHEN insufficient_privilege THEN
    RAISE NOTICE 'entities columns require postgres user - skipping';
END $$;

-- Attribution columns for claims
DO $$
BEGIN
    ALTER TABLE claims ADD COLUMN IF NOT EXISTS created_by_agent_id TEXT;
    ALTER TABLE claims ADD COLUMN IF NOT EXISTS created_by_user_id TEXT;
    ALTER TABLE claims ADD COLUMN IF NOT EXISTS derived_from TEXT[];
    ALTER TABLE claims ADD COLUMN IF NOT EXISTS derivation_type TEXT;
EXCEPTION WHEN insufficient_privilege THEN
    RAISE NOTICE 'claims columns require postgres user - skipping';
END $$;

-- Attribution columns for relations
DO $$
BEGIN
    ALTER TABLE relations ADD COLUMN IF NOT EXISTS created_by_agent_id TEXT;
EXCEPTION WHEN insufficient_privilege THEN
    RAISE NOTICE 'relations columns require postgres user - skipping';
END $$;

-- Attribution columns for assistant_memories
DO $$
BEGIN
    ALTER TABLE assistant_memories ADD COLUMN IF NOT EXISTS created_by_agent_id TEXT;
    ALTER TABLE assistant_memories ADD COLUMN IF NOT EXISTS created_by_user_id TEXT;
    ALTER TABLE assistant_memories ADD COLUMN IF NOT EXISTS derived_from TEXT[];
    ALTER TABLE assistant_memories ADD COLUMN IF NOT EXISTS cluster_id TEXT;
EXCEPTION WHEN insufficient_privilege THEN
    RAISE NOTICE 'assistant_memories columns require postgres user - skipping';
END $$;


-- ===========================================================================
-- 9. SEED DEFAULT PROMPT TEMPLATES
-- ===========================================================================

INSERT INTO prompt_templates (template_id, tenant_id, template_name, template_type,
                              system_prompt, user_prompt_template, temperature, max_tokens)
VALUES
    ('entity_default', '*', 'Default Entity Extraction', 'entity_extraction',
     'You are a precise entity extraction system. Extract named entities with their types and descriptions. Return ONLY valid JSON arrays. No explanations or commentary.',
     E'Extract entities from the following text:\n\n{{ text }}\n\nReturn a JSON array of objects with: name, type (person/place/organization/technology/service/infrastructure/concept/credential), description (brief), aliases (array).',
     0.1, 2048),

    ('claim_default', '*', 'Default Claim Extraction', 'claim_extraction',
     'You are a precise claim extraction system. Extract factual assertions as subject-predicate-object triples. Return ONLY valid JSON arrays.',
     E'Extract factual claims from:\n\n{{ text }}\n\nReturn JSON array of {subject, predicate, object, confidence (0-1), modality (asserted/tentative/negated/hypothetical)}.',
     0.1, 2048),

    ('relation_default', '*', 'Default Relation Extraction', 'relation_extraction',
     'You are a relation extraction system. Identify typed relationships between entities.',
     E'Given entities: {{ entities | join(", ") }}\n\nExtract relationships from:\n{{ text }}\n\nReturn JSON array of {from_entity, to_entity, relation_type, confidence, description}.',
     0.1, 2048),

    ('causal_default', '*', 'Default Causal Extraction', 'causal_extraction',
     'You are a causal relationship extraction system. Identify cause-effect relationships.',
     E'Extract causal relationships from:\n\n{{ text }}\n\nReturn JSON array of {cause_text, effect_text, causal_type (because/caused_by/resulted_in/led_to/triggered/enabled/prevented), confidence, temporal_order (cause_first/effect_first/unknown)}.',
     0.1, 2048),

    ('summarization_default', '*', 'Default Summarization', 'summarization',
     'You are a summarization system. Create concise, factual summaries preserving key technical details like IPs, ports, credentials, and version numbers.',
     E'Summarize the following in 2-3 paragraphs:\n\n{{ text }}\n\nPreserve specific technical details (IPs, ports, credentials, versions). Be factual and concise.',
     0.2, 1024),

    ('state_classify', '*', 'State Classification', 'state_classification',
     'Classify the conversation state. Return JSON with state and confidence.',
     E'Recent messages:\n{% for msg in messages %}{{ msg }}\n{% endfor %}\n\nClassify into one of: idle, debugging, planning, recalling, exploring, executing.\nReturn: {"state": "...", "confidence": 0.0-1.0}',
     0.3, 100),

    ('abstraction_default', '*', 'Memory Abstraction', 'abstraction',
     'Create a unified abstraction from related memories. Preserve key facts while generalizing common patterns.',
     E'Create a single unified summary from these related memories:\n\n{% for text in texts %}---\n{{ text }}\n{% endfor %}\n\nSynthesize into one cohesive paragraph capturing the common theme and key facts.',
     0.3, 500)

ON CONFLICT DO NOTHING;


-- ===========================================================================
-- 10. SEED DEFAULT AGENT FOR SYSTEM OPERATIONS
-- ===========================================================================

INSERT INTO agent_configs (agent_id, owner_tenant_id, agent_name, agent_type,
                           default_model, default_temperature, default_max_tokens)
VALUES
    ('system-background', 'shared', 'System Background Worker', 'background',
     'qwen35-27b-unredacted', 0.1, 2048),
    ('dream-cycle', 'shared', 'Dream Cycle Processor', 'background',
     'qwen35-27b-unredacted', 0.1, 2048)
ON CONFLICT (agent_id) DO NOTHING;


-- Migration complete (tables and seeds)

-- ===========================================================================
-- POST-MIGRATION NOTES
-- ===========================================================================
--
-- After running this migration:
--
-- 1. Vector indexes should be created after data backfill:
--    CREATE INDEX idx_memory_clusters_vec ON memory_clusters
--        USING ivfflat (cluster_embedding vector_cosine_ops);
--
-- 2. Existing segments/entities/claims will have NULL attribution fields.
--    Run a backfill to set created_by_agent_id = 'legacy' for existing records.
--
-- 3. Enable the subconscious daemon by setting:
--    MANIFOLD_SUBCONSCIOUS_ENABLED=true
--
-- 4. The learning system starts collecting signals immediately but only
--    begins adjusting scores after min_observations threshold is met.
