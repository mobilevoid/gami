--
-- PostgreSQL database dump
--


-- Dumped from database version 16.13 (Debian 16.13-1.pgdg12+1)
-- Dumped by pg_dump version 16.13 (Debian 16.13-1.pgdg12+1)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: age; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS age WITH SCHEMA ag_catalog;


--
-- Name: EXTENSION age; Type: COMMENT; Schema: -; Owner: -
--

COMMENT ON EXTENSION age IS 'AGE database extension';


--
-- Name: pg_trgm; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS pg_trgm WITH SCHEMA public;


--
-- Name: EXTENSION pg_trgm; Type: COMMENT; Schema: -; Owner: -
--

COMMENT ON EXTENSION pg_trgm IS 'text similarity measurement and index searching based on trigrams';


--
-- Name: vector; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;


--
-- Name: EXTENSION vector; Type: COMMENT; Schema: -; Owner: -
--

COMMENT ON EXTENSION vector IS 'vector data type and ivfflat and hnsw access methods';


--
-- Name: segments_lexical_tsv_trigger(); Type: FUNCTION; Schema: public; Owner: -
--

CREATE FUNCTION public.segments_lexical_tsv_trigger() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    NEW.lexical_tsv := to_tsvector('english', NEW.text);
    RETURN NEW;
END;
$$;


SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: agent_configs; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.agent_configs (
    id integer NOT NULL,
    agent_id text NOT NULL,
    owner_tenant_id text NOT NULL,
    agent_name text NOT NULL,
    agent_type text DEFAULT 'assistant'::text,
    personality_json jsonb DEFAULT '{}'::jsonb,
    default_model text,
    endpoint_url text,
    credentials_encrypted bytea,
    credential_key_id text,
    default_temperature double precision DEFAULT 0.7,
    default_max_tokens integer DEFAULT 2048,
    context_window_size integer DEFAULT 8192,
    system_prompt_override text,
    extraction_prompt_overrides jsonb DEFAULT '{}'::jsonb,
    scoring_overrides jsonb DEFAULT '{}'::jsonb,
    rate_limit_rpm integer DEFAULT 60,
    token_budget_daily integer DEFAULT 1000000,
    tokens_used_today integer DEFAULT 0,
    budget_reset_at timestamp with time zone DEFAULT (now() + '1 day'::interval),
    accuracy_score double precision DEFAULT 0.5,
    verified_claims integer DEFAULT 0,
    disputed_claims integer DEFAULT 0,
    total_claims integer DEFAULT 0,
    status text DEFAULT 'active'::text,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


--
-- Name: agent_configs_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.agent_configs_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: agent_configs_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.agent_configs_id_seq OWNED BY public.agent_configs.id;


--
-- Name: agent_trust_history; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.agent_trust_history (
    id bigint NOT NULL,
    agent_id text NOT NULL,
    accuracy_score double precision,
    verified_claims integer,
    disputed_claims integer,
    recorded_at timestamp with time zone DEFAULT now()
);


--
-- Name: agent_trust_history_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.agent_trust_history_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: agent_trust_history_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.agent_trust_history_id_seq OWNED BY public.agent_trust_history.id;


--
-- Name: assistant_memories; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.assistant_memories (
    memory_id text NOT NULL,
    owner_tenant_id text NOT NULL,
    memory_type text NOT NULL,
    subject_id text NOT NULL,
    normalized_text text NOT NULL,
    canonical_form_json jsonb DEFAULT '{}'::jsonb NOT NULL,
    embedding public.vector(768),
    importance_score double precision DEFAULT 0.5 NOT NULL,
    stability_score double precision DEFAULT 0.3 NOT NULL,
    recall_score double precision DEFAULT 0 NOT NULL,
    confirmation_count integer DEFAULT 0 NOT NULL,
    last_confirmed_at timestamp with time zone,
    expiration_policy jsonb DEFAULT '{}'::jsonb NOT NULL,
    expires_at timestamp with time zone,
    status text DEFAULT 'provisional'::text NOT NULL,
    superseded_by_id text,
    linked_entity_ids jsonb DEFAULT '[]'::jsonb NOT NULL,
    source_count integer DEFAULT 0 NOT NULL,
    sensitivity text DEFAULT 'normal'::text,
    version integer DEFAULT 1 NOT NULL,
    storage_tier text DEFAULT 'hot'::text NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    created_by_agent_id text,
    created_by_user_id text,
    derived_from text[],
    cluster_id text
);


--
-- Name: canonical_claims; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.canonical_claims (
    id bigint NOT NULL,
    claim_id text NOT NULL,
    subject text NOT NULL,
    predicate text NOT NULL,
    object text,
    modality text DEFAULT 'factual'::text,
    temporal_scope text,
    canonical_text text NOT NULL,
    confidence double precision DEFAULT 0.5,
    created_at timestamp with time zone DEFAULT now()
);


--
-- Name: canonical_claims_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.canonical_claims_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: canonical_claims_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.canonical_claims_id_seq OWNED BY public.canonical_claims.id;


--
-- Name: canonical_procedures; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.canonical_procedures (
    id bigint NOT NULL,
    procedure_id text NOT NULL,
    action text NOT NULL,
    target text,
    steps_json jsonb DEFAULT '[]'::jsonb,
    canonical_text text NOT NULL,
    confidence double precision DEFAULT 0.5,
    created_at timestamp with time zone DEFAULT now()
);


--
-- Name: canonical_procedures_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.canonical_procedures_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: canonical_procedures_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.canonical_procedures_id_seq OWNED BY public.canonical_procedures.id;


--
-- Name: causal_relations; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.causal_relations (
    id bigint NOT NULL,
    causal_id text DEFAULT (gen_random_uuid())::text NOT NULL,
    owner_tenant_id text NOT NULL,
    cause_entity_id text,
    cause_text text NOT NULL,
    effect_entity_id text,
    effect_text text NOT NULL,
    causal_type text NOT NULL,
    cause_timestamp timestamp with time zone,
    effect_timestamp timestamp with time zone,
    temporal_valid boolean,
    temporal_gap_hours double precision,
    explicitness_score double precision DEFAULT 0.5,
    authority_score double precision DEFAULT 0.5,
    corroboration_count integer DEFAULT 0,
    strength_score double precision GENERATED ALWAYS AS (((((0.4)::double precision * COALESCE(explicitness_score, (0.5)::double precision)) + ((0.3)::double precision * COALESCE(authority_score, (0.5)::double precision))) + ((0.3)::double precision * LEAST((1.0)::double precision, ((COALESCE(corroboration_count, 0))::double precision / (3.0)::double precision))))) STORED,
    source_segment_id text,
    extraction_pattern text,
    extraction_method text DEFAULT 'pattern'::text,
    created_by_agent_id text,
    status text DEFAULT 'active'::text,
    created_at timestamp with time zone DEFAULT now()
);


--
-- Name: causal_relations_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.causal_relations_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: causal_relations_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.causal_relations_id_seq OWNED BY public.causal_relations.id;


--
-- Name: claims; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.claims (
    claim_id text NOT NULL,
    owner_tenant_id text NOT NULL,
    subject_entity_id text,
    predicate text NOT NULL,
    object_entity_id text,
    object_literal_json jsonb,
    qualifiers_json jsonb DEFAULT '{}'::jsonb NOT NULL,
    modality text DEFAULT 'asserted'::text NOT NULL,
    temporal_scope_json jsonb DEFAULT '{}'::jsonb NOT NULL,
    confidence double precision NOT NULL,
    status text DEFAULT 'active'::text NOT NULL,
    contradiction_group_id text,
    superseded_by_id text,
    summary_text text,
    embedding public.vector(768),
    salience_score double precision DEFAULT 0 NOT NULL,
    novelty_score double precision DEFAULT 0 NOT NULL,
    support_count integer DEFAULT 0 NOT NULL,
    storage_tier text DEFAULT 'hot'::text NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    created_by_agent_id text,
    created_by_user_id text,
    derived_from text[],
    derivation_type text
);


--
-- Name: clusters; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.clusters (
    cluster_id text NOT NULL,
    owner_tenant_id text NOT NULL,
    cluster_type text NOT NULL,
    canonical_node_id text,
    member_ids jsonb DEFAULT '[]'::jsonb NOT NULL,
    summary_id text,
    confidence double precision DEFAULT 0 NOT NULL,
    status text DEFAULT 'active'::text NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: cold_tombstones; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.cold_tombstones (
    id integer NOT NULL,
    object_type text NOT NULL,
    object_id text NOT NULL,
    cold_path text NOT NULL,
    original_tier text NOT NULL,
    archive_reason text NOT NULL,
    archived_by text NOT NULL,
    archived_at timestamp with time zone DEFAULT now() NOT NULL,
    restore_count integer DEFAULT 0 NOT NULL,
    last_restored_at timestamp with time zone,
    metadata_json jsonb DEFAULT '{}'::jsonb NOT NULL
);


--
-- Name: cold_tombstones_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.cold_tombstones_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: cold_tombstones_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.cold_tombstones_id_seq OWNED BY public.cold_tombstones.id;


--
-- Name: compression_deltas; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.compression_deltas (
    id bigint NOT NULL,
    delta_id text DEFAULT (gen_random_uuid())::text,
    segment_id text NOT NULL,
    cluster_id text NOT NULL,
    unique_facts jsonb NOT NULL,
    unique_facts_text text,
    embedding public.vector(768),
    fact_count integer DEFAULT 0,
    total_importance double precision DEFAULT 0.0,
    compression_ratio double precision,
    extracted_by text DEFAULT 'dream_compress'::text,
    extraction_confidence double precision DEFAULT 0.7,
    created_at timestamp with time zone DEFAULT now()
);


--
-- Name: compression_deltas_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.compression_deltas_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: compression_deltas_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.compression_deltas_id_seq OWNED BY public.compression_deltas.id;


--
-- Name: distillation_datasets; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.distillation_datasets (
    dataset_id text NOT NULL,
    owner_tenant_id text NOT NULL,
    name text NOT NULL,
    task_type text NOT NULL,
    example_count integer DEFAULT 0 NOT NULL,
    file_path text,
    quality_score double precision,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    metadata_json jsonb DEFAULT '{}'::jsonb NOT NULL
);


--
-- Name: distillation_examples; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.distillation_examples (
    example_id text NOT NULL,
    dataset_id text NOT NULL,
    input_text text NOT NULL,
    output_text text NOT NULL,
    source_ids jsonb DEFAULT '[]'::jsonb NOT NULL,
    quality_score double precision DEFAULT 0 NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: entities; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.entities (
    entity_id text NOT NULL,
    owner_tenant_id text NOT NULL,
    entity_type text NOT NULL,
    canonical_name text NOT NULL,
    aliases_json jsonb DEFAULT '[]'::jsonb NOT NULL,
    description text,
    embedding public.vector(768),
    importance_score double precision DEFAULT 0 NOT NULL,
    ambiguity_score double precision DEFAULT 0 NOT NULL,
    graph_centrality double precision DEFAULT 0 NOT NULL,
    retrieval_count bigint DEFAULT 0 NOT NULL,
    status text DEFAULT 'provisional'::text NOT NULL,
    merged_into_id text,
    first_seen_at timestamp with time zone,
    last_seen_at timestamp with time zone,
    source_count integer DEFAULT 0 NOT NULL,
    mention_count bigint DEFAULT 0 NOT NULL,
    storage_tier text DEFAULT 'hot'::text NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    created_by_agent_id text,
    created_by_user_id text
);


--
-- Name: events; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.events (
    event_id text NOT NULL,
    owner_tenant_id text NOT NULL,
    event_type text NOT NULL,
    summary text NOT NULL,
    start_time timestamp with time zone,
    end_time timestamp with time zone,
    location_entity_id text,
    actor_entity_ids jsonb DEFAULT '[]'::jsonb NOT NULL,
    object_entity_ids jsonb DEFAULT '[]'::jsonb NOT NULL,
    confidence double precision NOT NULL,
    status text DEFAULT 'active'::text NOT NULL,
    storage_tier text DEFAULT 'hot'::text NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    created_by_agent_id text
);


--
-- Name: graph_fingerprints; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.graph_fingerprints (
    id bigint NOT NULL,
    target_id text NOT NULL,
    target_type text NOT NULL,
    fingerprint double precision[] NOT NULL,
    in_degree integer DEFAULT 0,
    out_degree integer DEFAULT 0,
    edge_types_json jsonb DEFAULT '{}'::jsonb,
    computed_at timestamp with time zone DEFAULT now()
);


--
-- Name: graph_fingerprints_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.graph_fingerprints_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: graph_fingerprints_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.graph_fingerprints_id_seq OWNED BY public.graph_fingerprints.id;


--
-- Name: jobs; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.jobs (
    job_id text NOT NULL,
    owner_tenant_id text NOT NULL,
    job_type text NOT NULL,
    target_type text,
    target_id text,
    status text DEFAULT 'pending'::text NOT NULL,
    priority integer DEFAULT 0 NOT NULL,
    attempt_count integer DEFAULT 0 NOT NULL,
    max_attempts integer DEFAULT 3 NOT NULL,
    scheduled_at timestamp with time zone DEFAULT now() NOT NULL,
    started_at timestamp with time zone,
    completed_at timestamp with time zone,
    worker_id text,
    worker_version text,
    input_hash text,
    output_hash text,
    result_json jsonb DEFAULT '{}'::jsonb NOT NULL,
    error_json jsonb DEFAULT '{}'::jsonb NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: manifold_embeddings; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.manifold_embeddings (
    id bigint NOT NULL,
    target_id text NOT NULL,
    target_type text NOT NULL,
    manifold_type text NOT NULL,
    embedding public.vector(768),
    canonical_form text,
    embedding_model text DEFAULT 'nomic-embed-text'::text,
    embedding_version integer DEFAULT 1,
    confidence_score double precision DEFAULT 0.5,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


--
-- Name: manifold_embeddings_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.manifold_embeddings_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: manifold_embeddings_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.manifold_embeddings_id_seq OWNED BY public.manifold_embeddings.id;


--
-- Name: memory_clusters; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.memory_clusters (
    id bigint NOT NULL,
    cluster_id text DEFAULT (gen_random_uuid())::text NOT NULL,
    owner_tenant_id text NOT NULL,
    cluster_name text,
    cluster_embedding public.vector(768),
    centroid_segment_id text,
    member_ids text[] NOT NULL,
    member_count integer DEFAULT 0,
    abstraction_text text,
    abstraction_embedding public.vector(768),
    stability_score double precision DEFAULT 0.5,
    repetition_count integer DEFAULT 1,
    last_reinforced_at timestamp with time zone,
    last_accessed_at timestamp with time zone DEFAULT now(),
    access_count integer DEFAULT 0,
    decay_rate double precision DEFAULT 0.01,
    current_decay double precision DEFAULT 1.0,
    inference_ids text[],
    status text DEFAULT 'active'::text,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


--
-- Name: memory_clusters_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.memory_clusters_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: memory_clusters_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.memory_clusters_id_seq OWNED BY public.memory_clusters.id;


--
-- Name: memory_operations; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.memory_operations (
    id bigint NOT NULL,
    operation_id text DEFAULT (gen_random_uuid())::text,
    operation_type text NOT NULL,
    new_memory_text text NOT NULL,
    new_memory_embedding public.vector(768),
    target_memory_id text,
    target_memory_text text,
    similarity_score double precision,
    llm_decision text,
    llm_confidence double precision,
    decision_reason text,
    executed boolean DEFAULT false,
    executed_at timestamp with time zone,
    result_memory_id text,
    tenant_id text NOT NULL,
    agent_id text,
    created_at timestamp with time zone DEFAULT now()
);


--
-- Name: memory_operation_stats; Type: VIEW; Schema: public; Owner: -
--

CREATE VIEW public.memory_operation_stats AS
 SELECT tenant_id,
    operation_type,
    count(*) AS count,
    avg(similarity_score) AS avg_similarity,
    avg(llm_confidence) AS avg_confidence,
    count(*) FILTER (WHERE executed) AS executed_count,
    max(created_at) AS last_operation
   FROM public.memory_operations
  GROUP BY tenant_id, operation_type;


--
-- Name: memory_operations_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.memory_operations_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: memory_operations_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.memory_operations_id_seq OWNED BY public.memory_operations.id;


--
-- Name: procedures; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.procedures (
    id bigint NOT NULL,
    procedure_id text DEFAULT (gen_random_uuid())::text,
    owner_tenant_id text NOT NULL,
    name text NOT NULL,
    description text,
    category text,
    trigger_patterns jsonb DEFAULT '[]'::jsonb,
    preconditions jsonb DEFAULT '[]'::jsonb,
    steps jsonb NOT NULL,
    postconditions jsonb DEFAULT '[]'::jsonb,
    parameters jsonb DEFAULT '[]'::jsonb,
    embedding public.vector(768),
    canonical_text text,
    execution_count integer DEFAULT 0,
    success_count integer DEFAULT 0,
    failure_count integer DEFAULT 0,
    success_rate double precision GENERATED ALWAYS AS (
CASE
    WHEN (execution_count > 0) THEN ((success_count)::double precision / (execution_count)::double precision)
    ELSE (0.5)::double precision
END) STORED,
    source_session_ids text[],
    source_segment_ids text[],
    confidence double precision DEFAULT 0.5,
    status text DEFAULT 'active'::text,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


--
-- Name: procedures_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.procedures_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: procedures_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.procedures_id_seq OWNED BY public.procedures.id;


--
-- Name: prompt_templates; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.prompt_templates (
    id integer NOT NULL,
    template_id text NOT NULL,
    tenant_id text DEFAULT '*'::text,
    agent_id text,
    template_name text NOT NULL,
    template_type text NOT NULL,
    system_prompt text,
    user_prompt_template text,
    output_format text DEFAULT 'json'::text,
    output_schema jsonb,
    temperature double precision,
    max_tokens integer,
    model_override text,
    version integer DEFAULT 1,
    is_active boolean DEFAULT true,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


--
-- Name: prompt_templates_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.prompt_templates_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: prompt_templates_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.prompt_templates_id_seq OWNED BY public.prompt_templates.id;


--
-- Name: proposed_changes; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.proposed_changes (
    proposal_id text NOT NULL,
    proposer_tenant_id text NOT NULL,
    change_type text NOT NULL,
    target_type text NOT NULL,
    target_id text NOT NULL,
    proposed_state_json jsonb NOT NULL,
    reason text NOT NULL,
    confidence double precision NOT NULL,
    evidence_ids jsonb DEFAULT '[]'::jsonb NOT NULL,
    status text DEFAULT 'pending'::text NOT NULL,
    reviewed_by text,
    reviewed_at timestamp with time zone,
    review_notes text,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    expires_at timestamp with time zone
);


--
-- Name: provenance; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.provenance (
    provenance_id text NOT NULL,
    target_type text NOT NULL,
    target_id text NOT NULL,
    source_id text NOT NULL,
    segment_id text,
    quote_text text,
    page_start integer,
    page_end integer,
    line_start integer,
    line_end integer,
    char_start bigint,
    char_end bigint,
    timestamp_start timestamp with time zone,
    timestamp_end timestamp with time zone,
    speaker_name text,
    extraction_method text NOT NULL,
    extractor_version text NOT NULL,
    confidence double precision NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: relations; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.relations (
    relation_id text NOT NULL,
    owner_tenant_id text NOT NULL,
    from_node_type text NOT NULL,
    from_node_id text NOT NULL,
    to_node_type text NOT NULL,
    to_node_id text NOT NULL,
    relation_type text NOT NULL,
    confidence double precision NOT NULL,
    weight double precision DEFAULT 1.0 NOT NULL,
    support_count integer DEFAULT 0 NOT NULL,
    contradiction_count integer DEFAULT 0 NOT NULL,
    first_created_at timestamp with time zone DEFAULT now() NOT NULL,
    last_confirmed_at timestamp with time zone,
    created_by text NOT NULL,
    extractor_version text,
    status text DEFAULT 'active'::text NOT NULL,
    storage_tier text DEFAULT 'hot'::text NOT NULL,
    metadata_json jsonb DEFAULT '{}'::jsonb NOT NULL,
    created_by_agent_id text
);


--
-- Name: retrieval_logs; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.retrieval_logs (
    id bigint NOT NULL,
    log_id text DEFAULT (gen_random_uuid())::text NOT NULL,
    session_id text,
    query_text text NOT NULL,
    query_embedding public.vector(768),
    query_mode text,
    segments_returned text[],
    scores_returned double precision[],
    latency_ms integer,
    outcome_type text,
    outcome_signal double precision,
    correction_text text,
    outcome_recorded_at timestamp with time zone,
    tenant_id text NOT NULL,
    agent_id text,
    user_id text,
    created_at timestamp with time zone DEFAULT now(),
    processed_in_dream boolean DEFAULT false
);


--
-- Name: retrieval_logs_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.retrieval_logs_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: retrieval_logs_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.retrieval_logs_id_seq OWNED BY public.retrieval_logs.id;


--
-- Name: segment_compression_status; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.segment_compression_status (
    segment_id text NOT NULL,
    compression_status text DEFAULT 'raw'::text,
    delta_id text,
    cluster_id text,
    processed_at timestamp with time zone DEFAULT now()
);


--
-- Name: segment_event_times; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.segment_event_times (
    segment_id text NOT NULL,
    event_time timestamp with time zone,
    event_time_confidence double precision DEFAULT 0.0,
    event_time_source text DEFAULT 'unknown'::text,
    temporal_refs jsonb DEFAULT '[]'::jsonb,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now()
);


--
-- Name: segments; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.segments (
    segment_id text NOT NULL,
    source_id text NOT NULL,
    owner_tenant_id text NOT NULL,
    parent_segment_id text,
    segment_type text NOT NULL,
    ordinal bigint,
    depth integer DEFAULT 0 NOT NULL,
    title_or_heading text,
    text text NOT NULL,
    token_count integer,
    char_start bigint,
    char_end bigint,
    byte_start bigint,
    byte_end bigint,
    page_start integer,
    page_end integer,
    line_start integer,
    line_end integer,
    bbox_json jsonb,
    speaker_role text,
    speaker_name text,
    message_timestamp timestamp with time zone,
    language text,
    embedding public.vector(768),
    quality_flags_json jsonb DEFAULT '{}'::jsonb NOT NULL,
    storage_tier text DEFAULT 'hot'::text NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    lexical_tsv tsvector,
    retrieval_count bigint DEFAULT 0,
    last_retrieved_at timestamp with time zone,
    created_by_agent_id text,
    created_by_user_id text,
    derived_from text[],
    derivation_type text,
    stability_score double precision DEFAULT 0.5,
    decay_score double precision DEFAULT 1.0,
    cluster_id text,
    last_reinforced_at timestamp with time zone,
    compression_status text DEFAULT 'raw'::text,
    compression_delta_id text,
    event_time timestamp with time zone,
    event_time_confidence double precision DEFAULT 0.0,
    event_time_source text DEFAULT 'unknown'::text
);


--
-- Name: sessions; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.sessions (
    id bigint NOT NULL,
    session_id text NOT NULL,
    tenant_id text NOT NULL,
    agent_id text,
    user_id text,
    conversation_state text DEFAULT 'idle'::text,
    state_confidence double precision DEFAULT 0.5,
    state_history jsonb DEFAULT '[]'::jsonb,
    active_entities text[],
    active_topics text[],
    hot_context_ids text[],
    message_count integer DEFAULT 0,
    retrieval_count integer DEFAULT 0,
    learning_signals_positive integer DEFAULT 0,
    learning_signals_negative integer DEFAULT 0,
    started_at timestamp with time zone DEFAULT now(),
    last_activity_at timestamp with time zone DEFAULT now(),
    ended_at timestamp with time zone
);


--
-- Name: sessions_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.sessions_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: sessions_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.sessions_id_seq OWNED BY public.sessions.id;


--
-- Name: sources; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.sources (
    source_id text NOT NULL,
    owner_tenant_id text NOT NULL,
    source_type text NOT NULL,
    title text,
    author_or_origin text,
    publisher_or_system text,
    language text DEFAULT 'en'::text,
    created_at timestamp with time zone,
    ingested_at timestamp with time zone DEFAULT now() NOT NULL,
    source_uri text,
    raw_file_path text,
    checksum text NOT NULL,
    file_size_bytes bigint,
    mime_type text,
    license_or_rights text,
    parse_status text DEFAULT 'pending'::text NOT NULL,
    ocr_status text DEFAULT 'not_required'::text NOT NULL,
    parser_version text,
    metadata_json jsonb DEFAULT '{}'::jsonb NOT NULL,
    storage_tier text DEFAULT 'hot'::text NOT NULL,
    cold_storage_path text,
    archived_at timestamp with time zone,
    archive_reason text
);


--
-- Name: subconscious_events; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.subconscious_events (
    id bigint NOT NULL,
    event_id text DEFAULT (gen_random_uuid())::text NOT NULL,
    session_id text,
    event_type text NOT NULL,
    detected_state text,
    state_confidence double precision,
    previous_state text,
    predicted_entities text[],
    predicted_topics text[],
    preloaded_segment_ids text[],
    prediction_confidence double precision,
    injected_context text,
    injection_token_count integer,
    injection_accepted boolean,
    latency_ms integer,
    created_at timestamp with time zone DEFAULT now()
);


--
-- Name: subconscious_events_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.subconscious_events_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: subconscious_events_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.subconscious_events_id_seq OWNED BY public.subconscious_events.id;


--
-- Name: summaries; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.summaries (
    summary_id text NOT NULL,
    owner_tenant_id text NOT NULL,
    scope_type text NOT NULL,
    scope_id text NOT NULL,
    abstraction_level text NOT NULL,
    summary_text text NOT NULL,
    embedding public.vector(768),
    based_on_ids jsonb DEFAULT '[]'::jsonb NOT NULL,
    quality_score double precision DEFAULT 0 NOT NULL,
    freshness_score double precision DEFAULT 1.0 NOT NULL,
    compression_ratio double precision DEFAULT 0 NOT NULL,
    summarizer_version text,
    status text DEFAULT 'active'::text NOT NULL,
    storage_tier text DEFAULT 'hot'::text NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: temporal_features; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.temporal_features (
    id bigint NOT NULL,
    target_id text NOT NULL,
    target_type text NOT NULL,
    features double precision[] NOT NULL,
    has_timestamp boolean DEFAULT false,
    timestamp_value timestamp with time zone,
    temporal_keywords text[],
    created_at timestamp with time zone DEFAULT now()
);


--
-- Name: temporal_features_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.temporal_features_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: temporal_features_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.temporal_features_id_seq OWNED BY public.temporal_features.id;


--
-- Name: tenant_audit_log; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.tenant_audit_log (
    id bigint NOT NULL,
    tenant_id text NOT NULL,
    action text NOT NULL,
    target_type text NOT NULL,
    target_id text,
    details_json jsonb DEFAULT '{}'::jsonb NOT NULL,
    "timestamp" timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: tenant_audit_log_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.tenant_audit_log_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: tenant_audit_log_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.tenant_audit_log_id_seq OWNED BY public.tenant_audit_log.id;


--
-- Name: tenant_permissions; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.tenant_permissions (
    id integer NOT NULL,
    tenant_id text NOT NULL,
    target_tenant_id text NOT NULL,
    permission text NOT NULL,
    scope text DEFAULT 'all'::text NOT NULL,
    granted_at timestamp with time zone DEFAULT now() NOT NULL,
    granted_by text NOT NULL,
    expires_at timestamp with time zone
);


--
-- Name: tenant_permissions_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.tenant_permissions_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: tenant_permissions_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.tenant_permissions_id_seq OWNED BY public.tenant_permissions.id;


--
-- Name: tenants; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.tenants (
    tenant_id text NOT NULL,
    name text NOT NULL,
    description text,
    tenant_type text DEFAULT 'agent'::text NOT NULL,
    api_key_hash text,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    status text DEFAULT 'active'::text NOT NULL,
    config_json jsonb DEFAULT '{}'::jsonb NOT NULL,
    daily_write_budget integer DEFAULT 10000,
    daily_delete_budget integer DEFAULT 100
);


--
-- Name: agent_configs id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.agent_configs ALTER COLUMN id SET DEFAULT nextval('public.agent_configs_id_seq'::regclass);


--
-- Name: agent_trust_history id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.agent_trust_history ALTER COLUMN id SET DEFAULT nextval('public.agent_trust_history_id_seq'::regclass);


--
-- Name: canonical_claims id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.canonical_claims ALTER COLUMN id SET DEFAULT nextval('public.canonical_claims_id_seq'::regclass);


--
-- Name: canonical_procedures id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.canonical_procedures ALTER COLUMN id SET DEFAULT nextval('public.canonical_procedures_id_seq'::regclass);


--
-- Name: causal_relations id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.causal_relations ALTER COLUMN id SET DEFAULT nextval('public.causal_relations_id_seq'::regclass);


--
-- Name: cold_tombstones id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cold_tombstones ALTER COLUMN id SET DEFAULT nextval('public.cold_tombstones_id_seq'::regclass);


--
-- Name: compression_deltas id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.compression_deltas ALTER COLUMN id SET DEFAULT nextval('public.compression_deltas_id_seq'::regclass);


--
-- Name: graph_fingerprints id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.graph_fingerprints ALTER COLUMN id SET DEFAULT nextval('public.graph_fingerprints_id_seq'::regclass);


--
-- Name: manifold_embeddings id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.manifold_embeddings ALTER COLUMN id SET DEFAULT nextval('public.manifold_embeddings_id_seq'::regclass);


--
-- Name: memory_clusters id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.memory_clusters ALTER COLUMN id SET DEFAULT nextval('public.memory_clusters_id_seq'::regclass);


--
-- Name: memory_operations id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.memory_operations ALTER COLUMN id SET DEFAULT nextval('public.memory_operations_id_seq'::regclass);


--
-- Name: procedures id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.procedures ALTER COLUMN id SET DEFAULT nextval('public.procedures_id_seq'::regclass);


--
-- Name: prompt_templates id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prompt_templates ALTER COLUMN id SET DEFAULT nextval('public.prompt_templates_id_seq'::regclass);


--
-- Name: retrieval_logs id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.retrieval_logs ALTER COLUMN id SET DEFAULT nextval('public.retrieval_logs_id_seq'::regclass);


--
-- Name: sessions id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sessions ALTER COLUMN id SET DEFAULT nextval('public.sessions_id_seq'::regclass);


--
-- Name: subconscious_events id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.subconscious_events ALTER COLUMN id SET DEFAULT nextval('public.subconscious_events_id_seq'::regclass);


--
-- Name: temporal_features id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.temporal_features ALTER COLUMN id SET DEFAULT nextval('public.temporal_features_id_seq'::regclass);


--
-- Name: tenant_audit_log id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.tenant_audit_log ALTER COLUMN id SET DEFAULT nextval('public.tenant_audit_log_id_seq'::regclass);


--
-- Name: tenant_permissions id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.tenant_permissions ALTER COLUMN id SET DEFAULT nextval('public.tenant_permissions_id_seq'::regclass);


--
-- Name: agent_configs agent_configs_agent_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.agent_configs
    ADD CONSTRAINT agent_configs_agent_id_key UNIQUE (agent_id);


--
-- Name: agent_configs agent_configs_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.agent_configs
    ADD CONSTRAINT agent_configs_pkey PRIMARY KEY (id);


--
-- Name: agent_trust_history agent_trust_history_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.agent_trust_history
    ADD CONSTRAINT agent_trust_history_pkey PRIMARY KEY (id);


--
-- Name: assistant_memories assistant_memories_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.assistant_memories
    ADD CONSTRAINT assistant_memories_pkey PRIMARY KEY (memory_id);


--
-- Name: canonical_claims canonical_claims_claim_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.canonical_claims
    ADD CONSTRAINT canonical_claims_claim_id_key UNIQUE (claim_id);


--
-- Name: canonical_claims canonical_claims_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.canonical_claims
    ADD CONSTRAINT canonical_claims_pkey PRIMARY KEY (id);


--
-- Name: canonical_procedures canonical_procedures_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.canonical_procedures
    ADD CONSTRAINT canonical_procedures_pkey PRIMARY KEY (id);


--
-- Name: canonical_procedures canonical_procedures_procedure_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.canonical_procedures
    ADD CONSTRAINT canonical_procedures_procedure_id_key UNIQUE (procedure_id);


--
-- Name: causal_relations causal_relations_causal_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.causal_relations
    ADD CONSTRAINT causal_relations_causal_id_key UNIQUE (causal_id);


--
-- Name: causal_relations causal_relations_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.causal_relations
    ADD CONSTRAINT causal_relations_pkey PRIMARY KEY (id);


--
-- Name: claims claims_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.claims
    ADD CONSTRAINT claims_pkey PRIMARY KEY (claim_id);


--
-- Name: clusters clusters_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.clusters
    ADD CONSTRAINT clusters_pkey PRIMARY KEY (cluster_id);


--
-- Name: cold_tombstones cold_tombstones_object_type_object_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cold_tombstones
    ADD CONSTRAINT cold_tombstones_object_type_object_id_key UNIQUE (object_type, object_id);


--
-- Name: cold_tombstones cold_tombstones_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.cold_tombstones
    ADD CONSTRAINT cold_tombstones_pkey PRIMARY KEY (id);


--
-- Name: compression_deltas compression_deltas_delta_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.compression_deltas
    ADD CONSTRAINT compression_deltas_delta_id_key UNIQUE (delta_id);


--
-- Name: compression_deltas compression_deltas_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.compression_deltas
    ADD CONSTRAINT compression_deltas_pkey PRIMARY KEY (id);


--
-- Name: distillation_datasets distillation_datasets_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.distillation_datasets
    ADD CONSTRAINT distillation_datasets_pkey PRIMARY KEY (dataset_id);


--
-- Name: distillation_examples distillation_examples_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.distillation_examples
    ADD CONSTRAINT distillation_examples_pkey PRIMARY KEY (example_id);


--
-- Name: entities entities_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.entities
    ADD CONSTRAINT entities_pkey PRIMARY KEY (entity_id);


--
-- Name: events events_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.events
    ADD CONSTRAINT events_pkey PRIMARY KEY (event_id);


--
-- Name: graph_fingerprints graph_fingerprints_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.graph_fingerprints
    ADD CONSTRAINT graph_fingerprints_pkey PRIMARY KEY (id);


--
-- Name: jobs jobs_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.jobs
    ADD CONSTRAINT jobs_pkey PRIMARY KEY (job_id);


--
-- Name: manifold_embeddings manifold_embeddings_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.manifold_embeddings
    ADD CONSTRAINT manifold_embeddings_pkey PRIMARY KEY (id);


--
-- Name: memory_clusters memory_clusters_cluster_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.memory_clusters
    ADD CONSTRAINT memory_clusters_cluster_id_key UNIQUE (cluster_id);


--
-- Name: memory_clusters memory_clusters_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.memory_clusters
    ADD CONSTRAINT memory_clusters_pkey PRIMARY KEY (id);


--
-- Name: memory_operations memory_operations_operation_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.memory_operations
    ADD CONSTRAINT memory_operations_operation_id_key UNIQUE (operation_id);


--
-- Name: memory_operations memory_operations_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.memory_operations
    ADD CONSTRAINT memory_operations_pkey PRIMARY KEY (id);


--
-- Name: procedures procedures_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.procedures
    ADD CONSTRAINT procedures_pkey PRIMARY KEY (id);


--
-- Name: procedures procedures_procedure_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.procedures
    ADD CONSTRAINT procedures_procedure_id_key UNIQUE (procedure_id);


--
-- Name: prompt_templates prompt_templates_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prompt_templates
    ADD CONSTRAINT prompt_templates_pkey PRIMARY KEY (id);


--
-- Name: proposed_changes proposed_changes_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.proposed_changes
    ADD CONSTRAINT proposed_changes_pkey PRIMARY KEY (proposal_id);


--
-- Name: provenance provenance_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.provenance
    ADD CONSTRAINT provenance_pkey PRIMARY KEY (provenance_id);


--
-- Name: relations relations_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.relations
    ADD CONSTRAINT relations_pkey PRIMARY KEY (relation_id);


--
-- Name: retrieval_logs retrieval_logs_log_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.retrieval_logs
    ADD CONSTRAINT retrieval_logs_log_id_key UNIQUE (log_id);


--
-- Name: retrieval_logs retrieval_logs_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.retrieval_logs
    ADD CONSTRAINT retrieval_logs_pkey PRIMARY KEY (id);


--
-- Name: segment_compression_status segment_compression_status_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.segment_compression_status
    ADD CONSTRAINT segment_compression_status_pkey PRIMARY KEY (segment_id);


--
-- Name: segment_event_times segment_event_times_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.segment_event_times
    ADD CONSTRAINT segment_event_times_pkey PRIMARY KEY (segment_id);


--
-- Name: segments segments_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.segments
    ADD CONSTRAINT segments_pkey PRIMARY KEY (segment_id);


--
-- Name: sessions sessions_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sessions
    ADD CONSTRAINT sessions_pkey PRIMARY KEY (id);


--
-- Name: sessions sessions_session_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sessions
    ADD CONSTRAINT sessions_session_id_key UNIQUE (session_id);


--
-- Name: sources sources_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sources
    ADD CONSTRAINT sources_pkey PRIMARY KEY (source_id);


--
-- Name: subconscious_events subconscious_events_event_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.subconscious_events
    ADD CONSTRAINT subconscious_events_event_id_key UNIQUE (event_id);


--
-- Name: subconscious_events subconscious_events_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.subconscious_events
    ADD CONSTRAINT subconscious_events_pkey PRIMARY KEY (id);


--
-- Name: summaries summaries_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.summaries
    ADD CONSTRAINT summaries_pkey PRIMARY KEY (summary_id);


--
-- Name: temporal_features temporal_features_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.temporal_features
    ADD CONSTRAINT temporal_features_pkey PRIMARY KEY (id);


--
-- Name: tenant_audit_log tenant_audit_log_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.tenant_audit_log
    ADD CONSTRAINT tenant_audit_log_pkey PRIMARY KEY (id);


--
-- Name: tenant_permissions tenant_permissions_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.tenant_permissions
    ADD CONSTRAINT tenant_permissions_pkey PRIMARY KEY (id);


--
-- Name: tenant_permissions tenant_permissions_tenant_id_target_tenant_id_permission_sc_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.tenant_permissions
    ADD CONSTRAINT tenant_permissions_tenant_id_target_tenant_id_permission_sc_key UNIQUE (tenant_id, target_tenant_id, permission, scope);


--
-- Name: tenants tenants_name_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.tenants
    ADD CONSTRAINT tenants_name_key UNIQUE (name);


--
-- Name: tenants tenants_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.tenants
    ADD CONSTRAINT tenants_pkey PRIMARY KEY (tenant_id);


--
-- Name: graph_fingerprints uq_graph_fingerprints_target; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.graph_fingerprints
    ADD CONSTRAINT uq_graph_fingerprints_target UNIQUE (target_id, target_type);


--
-- Name: manifold_embeddings uq_manifold_embeddings_target_manifold; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.manifold_embeddings
    ADD CONSTRAINT uq_manifold_embeddings_target_manifold UNIQUE (target_id, target_type, manifold_type);


--
-- Name: temporal_features uq_temporal_features_target; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.temporal_features
    ADD CONSTRAINT uq_temporal_features_target UNIQUE (target_id, target_type);


--
-- Name: idx_agent_configs_status; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_agent_configs_status ON public.agent_configs USING btree (status) WHERE (status = 'active'::text);


--
-- Name: idx_agent_configs_tenant; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_agent_configs_tenant ON public.agent_configs USING btree (owner_tenant_id);


--
-- Name: idx_agent_trust_history_agent; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_agent_trust_history_agent ON public.agent_trust_history USING btree (agent_id, recorded_at DESC);


--
-- Name: idx_canonical_claims_subject; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_canonical_claims_subject ON public.canonical_claims USING btree (subject);


--
-- Name: idx_canonical_procedures_action; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_canonical_procedures_action ON public.canonical_procedures USING btree (action);


--
-- Name: idx_causal_cause; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_causal_cause ON public.causal_relations USING btree (cause_entity_id) WHERE (status = 'active'::text);


--
-- Name: idx_causal_effect; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_causal_effect ON public.causal_relations USING btree (effect_entity_id) WHERE (status = 'active'::text);


--
-- Name: idx_causal_strength; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_causal_strength ON public.causal_relations USING btree (strength_score DESC) WHERE (status = 'active'::text);


--
-- Name: idx_causal_tenant; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_causal_tenant ON public.causal_relations USING btree (owner_tenant_id);


--
-- Name: idx_causal_type; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_causal_type ON public.causal_relations USING btree (causal_type) WHERE (status = 'active'::text);


--
-- Name: idx_claims_embedding; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_claims_embedding ON public.claims USING ivfflat (embedding public.vector_cosine_ops) WITH (lists='100');


--
-- Name: idx_claims_owner; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_claims_owner ON public.claims USING btree (owner_tenant_id);


--
-- Name: idx_claims_subject; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_claims_subject ON public.claims USING btree (subject_entity_id);


--
-- Name: idx_deltas_cluster; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_deltas_cluster ON public.compression_deltas USING btree (cluster_id);


--
-- Name: idx_deltas_importance; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_deltas_importance ON public.compression_deltas USING btree (total_importance DESC);


--
-- Name: idx_deltas_segment; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_deltas_segment ON public.compression_deltas USING btree (segment_id);


--
-- Name: idx_entities_embedding; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_entities_embedding ON public.entities USING ivfflat (embedding public.vector_cosine_ops) WITH (lists='100');


--
-- Name: idx_entities_owner; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_entities_owner ON public.entities USING btree (owner_tenant_id);


--
-- Name: idx_entities_status; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_entities_status ON public.entities USING btree (status);


--
-- Name: idx_event_times_confidence; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_event_times_confidence ON public.segment_event_times USING btree (event_time_confidence DESC);


--
-- Name: idx_event_times_time; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_event_times_time ON public.segment_event_times USING btree (event_time DESC) WHERE (event_time IS NOT NULL);


--
-- Name: idx_jobs_status; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_jobs_status ON public.jobs USING btree (status, priority DESC);


--
-- Name: idx_me_claim_vec; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_me_claim_vec ON public.manifold_embeddings USING ivfflat (embedding public.vector_cosine_ops) WITH (lists='100') WHERE (manifold_type = 'CLAIM'::text);


--
-- Name: idx_me_evidence_vec; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_me_evidence_vec ON public.manifold_embeddings USING ivfflat (embedding public.vector_cosine_ops) WITH (lists='100') WHERE (manifold_type = 'EVIDENCE'::text);


--
-- Name: idx_me_manifold_type; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_me_manifold_type ON public.manifold_embeddings USING btree (manifold_type);


--
-- Name: idx_me_procedure_vec; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_me_procedure_vec ON public.manifold_embeddings USING ivfflat (embedding public.vector_cosine_ops) WITH (lists='100') WHERE (manifold_type = 'PROCEDURE'::text);


--
-- Name: idx_me_relation_vec; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_me_relation_vec ON public.manifold_embeddings USING ivfflat (embedding public.vector_cosine_ops) WITH (lists='100') WHERE (manifold_type = 'RELATION'::text);


--
-- Name: idx_me_target; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_me_target ON public.manifold_embeddings USING btree (target_id, target_type);


--
-- Name: idx_me_time_vec; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_me_time_vec ON public.manifold_embeddings USING ivfflat (embedding public.vector_cosine_ops) WITH (lists='100') WHERE (manifold_type = 'TIME'::text);


--
-- Name: idx_me_topic_vec; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_me_topic_vec ON public.manifold_embeddings USING ivfflat (embedding public.vector_cosine_ops) WITH (lists='100') WHERE (manifold_type = 'TOPIC'::text);


--
-- Name: idx_memories_embedding; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_memories_embedding ON public.assistant_memories USING ivfflat (embedding public.vector_cosine_ops) WITH (lists='100');


--
-- Name: idx_memories_owner; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_memories_owner ON public.assistant_memories USING btree (owner_tenant_id);


--
-- Name: idx_memories_status; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_memories_status ON public.assistant_memories USING btree (status);


--
-- Name: idx_memory_clusters_abstraction_embedding; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_memory_clusters_abstraction_embedding ON public.memory_clusters USING ivfflat (abstraction_embedding public.vector_cosine_ops) WITH (lists='100');


--
-- Name: idx_memory_clusters_embedding; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_memory_clusters_embedding ON public.memory_clusters USING ivfflat (cluster_embedding public.vector_cosine_ops) WITH (lists='100');


--
-- Name: idx_memory_clusters_stability; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_memory_clusters_stability ON public.memory_clusters USING btree (stability_score DESC) WHERE (status = 'active'::text);


--
-- Name: idx_memory_clusters_tenant; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_memory_clusters_tenant ON public.memory_clusters USING btree (owner_tenant_id);


--
-- Name: idx_memory_ops_tenant; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_memory_ops_tenant ON public.memory_operations USING btree (tenant_id, created_at DESC);


--
-- Name: idx_memory_ops_type; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_memory_ops_type ON public.memory_operations USING btree (operation_type);


--
-- Name: idx_memory_ops_unexecuted; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_memory_ops_unexecuted ON public.memory_operations USING btree (executed) WHERE (NOT executed);


--
-- Name: idx_procedures_category; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_procedures_category ON public.procedures USING btree (category) WHERE (status = 'active'::text);


--
-- Name: idx_procedures_success; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_procedures_success ON public.procedures USING btree (success_rate DESC) WHERE (status = 'active'::text);


--
-- Name: idx_procedures_tenant; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_procedures_tenant ON public.procedures USING btree (owner_tenant_id) WHERE (status = 'active'::text);


--
-- Name: idx_prompt_templates_lookup; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_prompt_templates_lookup ON public.prompt_templates USING btree (template_type, tenant_id, agent_id) WHERE is_active;


--
-- Name: idx_prompt_templates_type; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_prompt_templates_type ON public.prompt_templates USING btree (template_type) WHERE is_active;


--
-- Name: idx_proposals_status; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_proposals_status ON public.proposed_changes USING btree (status);


--
-- Name: idx_provenance_target; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_provenance_target ON public.provenance USING btree (target_type, target_id);


--
-- Name: idx_relations_from; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_relations_from ON public.relations USING btree (from_node_type, from_node_id);


--
-- Name: idx_relations_owner; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_relations_owner ON public.relations USING btree (owner_tenant_id);


--
-- Name: idx_relations_to; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_relations_to ON public.relations USING btree (to_node_type, to_node_id);


--
-- Name: idx_retrieval_logs_agent; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_retrieval_logs_agent ON public.retrieval_logs USING btree (agent_id) WHERE (agent_id IS NOT NULL);


--
-- Name: idx_retrieval_logs_query_embedding; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_retrieval_logs_query_embedding ON public.retrieval_logs USING ivfflat (query_embedding public.vector_cosine_ops) WITH (lists='100');


--
-- Name: idx_retrieval_logs_session; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_retrieval_logs_session ON public.retrieval_logs USING btree (session_id) WHERE (session_id IS NOT NULL);


--
-- Name: idx_retrieval_logs_tenant; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_retrieval_logs_tenant ON public.retrieval_logs USING btree (tenant_id, created_at DESC);


--
-- Name: idx_retrieval_logs_unprocessed; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_retrieval_logs_unprocessed ON public.retrieval_logs USING btree (created_at DESC) WHERE ((NOT processed_in_dream) AND (outcome_signal IS NOT NULL));


--
-- Name: idx_seg_comp_cluster; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_seg_comp_cluster ON public.segment_compression_status USING btree (cluster_id);


--
-- Name: idx_seg_comp_status; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_seg_comp_status ON public.segment_compression_status USING btree (compression_status);


--
-- Name: idx_segments_embedding; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_segments_embedding ON public.segments USING ivfflat (embedding public.vector_cosine_ops) WITH (lists='100');


--
-- Name: idx_segments_event_time; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_segments_event_time ON public.segments USING btree (event_time DESC) WHERE (event_time IS NOT NULL);


--
-- Name: idx_segments_hot; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_segments_hot ON public.segments USING btree (retrieval_count DESC, last_retrieved_at DESC);


--
-- Name: idx_segments_lexical; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_segments_lexical ON public.segments USING gin (lexical_tsv);


--
-- Name: idx_segments_owner; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_segments_owner ON public.segments USING btree (owner_tenant_id);


--
-- Name: idx_segments_source; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_segments_source ON public.segments USING btree (source_id);


--
-- Name: idx_segments_type; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_segments_type ON public.segments USING btree (segment_type);


--
-- Name: idx_sessions_active; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_sessions_active ON public.sessions USING btree (last_activity_at DESC) WHERE (ended_at IS NULL);


--
-- Name: idx_sessions_agent; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_sessions_agent ON public.sessions USING btree (agent_id) WHERE (agent_id IS NOT NULL);


--
-- Name: idx_sessions_state; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_sessions_state ON public.sessions USING btree (conversation_state) WHERE (ended_at IS NULL);


--
-- Name: idx_sessions_tenant; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_sessions_tenant ON public.sessions USING btree (tenant_id, last_activity_at DESC);


--
-- Name: idx_sources_checksum; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_sources_checksum ON public.sources USING btree (checksum);


--
-- Name: idx_sources_owner; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_sources_owner ON public.sources USING btree (owner_tenant_id);


--
-- Name: idx_subconscious_session; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_subconscious_session ON public.subconscious_events USING btree (session_id, created_at DESC) WHERE (session_id IS NOT NULL);


--
-- Name: idx_subconscious_type; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_subconscious_type ON public.subconscious_events USING btree (event_type, created_at DESC);


--
-- Name: idx_summaries_embedding; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_summaries_embedding ON public.summaries USING ivfflat (embedding public.vector_cosine_ops) WITH (lists='100');


--
-- Name: idx_tenant_audit_tenant; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_tenant_audit_tenant ON public.tenant_audit_log USING btree (tenant_id, "timestamp");


--
-- Name: uq_procedures_tenant_name; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX uq_procedures_tenant_name ON public.procedures USING btree (owner_tenant_id, lower(name));


--
-- Name: uq_prompt_template; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX uq_prompt_template ON public.prompt_templates USING btree (template_id, tenant_id, COALESCE(agent_id, ''::text), version);


--
-- Name: segments trg_segments_lexical_tsv; Type: TRIGGER; Schema: public; Owner: -
--

CREATE TRIGGER trg_segments_lexical_tsv BEFORE INSERT OR UPDATE OF text ON public.segments FOR EACH ROW EXECUTE FUNCTION public.segments_lexical_tsv_trigger();


--
-- Name: assistant_memories assistant_memories_owner_tenant_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.assistant_memories
    ADD CONSTRAINT assistant_memories_owner_tenant_id_fkey FOREIGN KEY (owner_tenant_id) REFERENCES public.tenants(tenant_id);


--
-- Name: assistant_memories assistant_memories_superseded_by_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.assistant_memories
    ADD CONSTRAINT assistant_memories_superseded_by_id_fkey FOREIGN KEY (superseded_by_id) REFERENCES public.assistant_memories(memory_id);


--
-- Name: claims claims_object_entity_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.claims
    ADD CONSTRAINT claims_object_entity_id_fkey FOREIGN KEY (object_entity_id) REFERENCES public.entities(entity_id);


--
-- Name: claims claims_owner_tenant_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.claims
    ADD CONSTRAINT claims_owner_tenant_id_fkey FOREIGN KEY (owner_tenant_id) REFERENCES public.tenants(tenant_id);


--
-- Name: claims claims_subject_entity_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.claims
    ADD CONSTRAINT claims_subject_entity_id_fkey FOREIGN KEY (subject_entity_id) REFERENCES public.entities(entity_id);


--
-- Name: claims claims_superseded_by_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.claims
    ADD CONSTRAINT claims_superseded_by_id_fkey FOREIGN KEY (superseded_by_id) REFERENCES public.claims(claim_id);


--
-- Name: clusters clusters_owner_tenant_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.clusters
    ADD CONSTRAINT clusters_owner_tenant_id_fkey FOREIGN KEY (owner_tenant_id) REFERENCES public.tenants(tenant_id);


--
-- Name: clusters clusters_summary_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.clusters
    ADD CONSTRAINT clusters_summary_id_fkey FOREIGN KEY (summary_id) REFERENCES public.summaries(summary_id);


--
-- Name: distillation_datasets distillation_datasets_owner_tenant_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.distillation_datasets
    ADD CONSTRAINT distillation_datasets_owner_tenant_id_fkey FOREIGN KEY (owner_tenant_id) REFERENCES public.tenants(tenant_id);


--
-- Name: distillation_examples distillation_examples_dataset_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.distillation_examples
    ADD CONSTRAINT distillation_examples_dataset_id_fkey FOREIGN KEY (dataset_id) REFERENCES public.distillation_datasets(dataset_id);


--
-- Name: entities entities_merged_into_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.entities
    ADD CONSTRAINT entities_merged_into_id_fkey FOREIGN KEY (merged_into_id) REFERENCES public.entities(entity_id);


--
-- Name: entities entities_owner_tenant_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.entities
    ADD CONSTRAINT entities_owner_tenant_id_fkey FOREIGN KEY (owner_tenant_id) REFERENCES public.tenants(tenant_id);


--
-- Name: events events_location_entity_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.events
    ADD CONSTRAINT events_location_entity_id_fkey FOREIGN KEY (location_entity_id) REFERENCES public.entities(entity_id);


--
-- Name: events events_owner_tenant_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.events
    ADD CONSTRAINT events_owner_tenant_id_fkey FOREIGN KEY (owner_tenant_id) REFERENCES public.tenants(tenant_id);


--
-- Name: jobs jobs_owner_tenant_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.jobs
    ADD CONSTRAINT jobs_owner_tenant_id_fkey FOREIGN KEY (owner_tenant_id) REFERENCES public.tenants(tenant_id);


--
-- Name: proposed_changes proposed_changes_proposer_tenant_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.proposed_changes
    ADD CONSTRAINT proposed_changes_proposer_tenant_id_fkey FOREIGN KEY (proposer_tenant_id) REFERENCES public.tenants(tenant_id);


--
-- Name: provenance provenance_segment_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.provenance
    ADD CONSTRAINT provenance_segment_id_fkey FOREIGN KEY (segment_id) REFERENCES public.segments(segment_id);


--
-- Name: provenance provenance_source_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.provenance
    ADD CONSTRAINT provenance_source_id_fkey FOREIGN KEY (source_id) REFERENCES public.sources(source_id);


--
-- Name: relations relations_owner_tenant_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.relations
    ADD CONSTRAINT relations_owner_tenant_id_fkey FOREIGN KEY (owner_tenant_id) REFERENCES public.tenants(tenant_id);


--
-- Name: segments segments_owner_tenant_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.segments
    ADD CONSTRAINT segments_owner_tenant_id_fkey FOREIGN KEY (owner_tenant_id) REFERENCES public.tenants(tenant_id);


--
-- Name: segments segments_parent_segment_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.segments
    ADD CONSTRAINT segments_parent_segment_id_fkey FOREIGN KEY (parent_segment_id) REFERENCES public.segments(segment_id);


--
-- Name: segments segments_source_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.segments
    ADD CONSTRAINT segments_source_id_fkey FOREIGN KEY (source_id) REFERENCES public.sources(source_id);


--
-- Name: sources sources_owner_tenant_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.sources
    ADD CONSTRAINT sources_owner_tenant_id_fkey FOREIGN KEY (owner_tenant_id) REFERENCES public.tenants(tenant_id);


--
-- Name: summaries summaries_owner_tenant_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.summaries
    ADD CONSTRAINT summaries_owner_tenant_id_fkey FOREIGN KEY (owner_tenant_id) REFERENCES public.tenants(tenant_id);


--
-- Name: tenant_permissions tenant_permissions_target_tenant_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.tenant_permissions
    ADD CONSTRAINT tenant_permissions_target_tenant_id_fkey FOREIGN KEY (target_tenant_id) REFERENCES public.tenants(tenant_id);


--
-- Name: tenant_permissions tenant_permissions_tenant_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.tenant_permissions
    ADD CONSTRAINT tenant_permissions_tenant_id_fkey FOREIGN KEY (tenant_id) REFERENCES public.tenants(tenant_id);


--
-- PostgreSQL database dump complete
--

\unrestrict z0ukWNsLIPqDuGx6tIm5Bf6IDrA4QWsEAw1ljNYfFA8DuZ84DkelFoFNqPak2Vm

