"""
GAMI Innovation Extensions - Database Schema Migration

Creates all tables for Phase 13 Innovation Extensions:
- retrieval_logs: Learning signal collection
- agent_configs: Per-agent credentials & settings
- prompt_templates: Configurable prompts
- causal_relations: Cause-effect links
- memory_clusters: Consolidation groups
- sessions: Conversation state tracking
- subconscious_events: Daemon audit trail
- agent_trust_history: Trust score history

Also adds attribution columns to existing tables.

Revision ID: 013_innovation
Revises: 012_deep_dream (or latest existing)
Create Date: 2026-04-11
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, BYTEA, ARRAY
from pgvector.sqlalchemy import Vector

# Revision identifiers
revision = '013_innovation'
down_revision = None  # Set to actual previous revision
branch_labels = None
depends_on = None


def upgrade():
    """Apply innovation extensions schema."""

    # ==========================================================================
    # 1. RETRIEVAL LOGS - Learning Signal Collection
    # ==========================================================================
    op.create_table(
        'retrieval_logs',
        sa.Column('id', sa.BigInteger, primary_key=True),
        sa.Column('log_id', sa.Text, nullable=False, unique=True,
                  server_default=sa.text("gen_random_uuid()::text")),
        sa.Column('session_id', sa.Text, index=True),
        sa.Column('query_text', sa.Text, nullable=False),
        sa.Column('query_embedding', Vector(768)),
        sa.Column('query_mode', sa.Text),

        # Results
        sa.Column('segments_returned', ARRAY(sa.Text)),
        sa.Column('scores_returned', ARRAY(sa.Float)),

        # Outcome signals
        sa.Column('outcome_type', sa.Text),
        sa.Column('outcome_signal', sa.Float),
        sa.Column('correction_text', sa.Text),

        # Attribution
        sa.Column('tenant_id', sa.Text, nullable=False, index=True),
        sa.Column('agent_id', sa.Text),
        sa.Column('user_id', sa.Text),

        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()')),
        sa.Column('processed_in_dream', sa.Boolean, server_default='FALSE'),
    )

    op.create_index('idx_retrieval_logs_unprocessed', 'retrieval_logs',
                    ['processed_in_dream'],
                    postgresql_where=sa.text('NOT processed_in_dream'))

    op.create_index('idx_retrieval_logs_tenant_created', 'retrieval_logs',
                    ['tenant_id', sa.text('created_at DESC')])

    # ==========================================================================
    # 2. AGENT CONFIGS - Per-Agent Credentials & Settings
    # ==========================================================================
    op.create_table(
        'agent_configs',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('agent_id', sa.Text, nullable=False, unique=True),
        sa.Column('owner_tenant_id', sa.Text, nullable=False, index=True),

        # Identity
        sa.Column('agent_name', sa.Text, nullable=False),
        sa.Column('agent_type', sa.Text, server_default='assistant'),
        sa.Column('personality_json', JSONB, server_default='{}'),

        # LLM Configuration
        sa.Column('default_model', sa.Text),
        sa.Column('endpoint_url', sa.Text),
        sa.Column('credentials_encrypted', BYTEA),
        sa.Column('credential_key_id', sa.Text),

        # Default Parameters
        sa.Column('default_temperature', sa.Float, server_default='0.7'),
        sa.Column('default_max_tokens', sa.Integer, server_default='2048'),
        sa.Column('context_window_size', sa.Integer, server_default='8192'),

        # Prompt Overrides
        sa.Column('system_prompt_override', sa.Text),
        sa.Column('extraction_prompt_overrides', JSONB, server_default='{}'),

        # Scoring Overrides
        sa.Column('scoring_overrides', JSONB, server_default='{}'),

        # Rate Limits
        sa.Column('rate_limit_rpm', sa.Integer, server_default='60'),
        sa.Column('token_budget_daily', sa.Integer, server_default='1000000'),
        sa.Column('tokens_used_today', sa.Integer, server_default='0'),
        sa.Column('budget_reset_at', sa.DateTime(timezone=True)),

        # Trust Metrics
        sa.Column('accuracy_score', sa.Float, server_default='0.5'),
        sa.Column('verified_claims', sa.Integer, server_default='0'),
        sa.Column('disputed_claims', sa.Integer, server_default='0'),
        sa.Column('total_claims', sa.Integer, server_default='0'),

        sa.Column('status', sa.Text, server_default="'active'"),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()')),
    )

    op.create_index('idx_agent_configs_status', 'agent_configs',
                    ['status'],
                    postgresql_where=sa.text("status = 'active'"))

    # ==========================================================================
    # 3. PROMPT TEMPLATES - Configurable Prompts
    # ==========================================================================
    op.create_table(
        'prompt_templates',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('template_id', sa.Text, nullable=False),
        sa.Column('tenant_id', sa.Text, server_default="'*'"),
        sa.Column('agent_id', sa.Text),

        sa.Column('template_name', sa.Text, nullable=False),
        sa.Column('template_type', sa.Text, nullable=False),

        # Template Content
        sa.Column('system_prompt', sa.Text),
        sa.Column('user_prompt_template', sa.Text),

        # LLM Parameters
        sa.Column('temperature', sa.Float),
        sa.Column('max_tokens', sa.Integer),
        sa.Column('model_override', sa.Text),

        # Output Format
        sa.Column('output_format', sa.Text, server_default="'json'"),
        sa.Column('output_schema', JSONB),

        # Versioning
        sa.Column('version', sa.Integer, server_default='1'),
        sa.Column('is_active', sa.Boolean, server_default='TRUE'),

        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()')),

        sa.UniqueConstraint('template_id', 'tenant_id', 'agent_id', 'version',
                           name='uq_prompt_template'),
    )

    op.create_index('idx_prompt_templates_lookup', 'prompt_templates',
                    ['template_type', 'tenant_id', 'agent_id'],
                    postgresql_where=sa.text('is_active'))

    # ==========================================================================
    # 4. CAUSAL RELATIONS - Cause-Effect Links
    # ==========================================================================
    op.create_table(
        'causal_relations',
        sa.Column('id', sa.BigInteger, primary_key=True),
        sa.Column('causal_id', sa.Text, nullable=False, unique=True,
                  server_default=sa.text("gen_random_uuid()::text")),
        sa.Column('owner_tenant_id', sa.Text, nullable=False, index=True),

        # Causal Structure
        sa.Column('cause_entity_id', sa.Text, index=True),
        sa.Column('cause_text', sa.Text, nullable=False),
        sa.Column('effect_entity_id', sa.Text, index=True),
        sa.Column('effect_text', sa.Text, nullable=False),

        # Causal Type
        sa.Column('causal_type', sa.Text, nullable=False),

        # Temporal Validation
        sa.Column('cause_timestamp', sa.DateTime(timezone=True)),
        sa.Column('effect_timestamp', sa.DateTime(timezone=True)),
        sa.Column('temporal_valid', sa.Boolean),
        sa.Column('temporal_gap_hours', sa.Float),

        # Strength Scoring
        sa.Column('explicitness_score', sa.Float, server_default='0.5'),
        sa.Column('authority_score', sa.Float, server_default='0.5'),
        sa.Column('corroboration_count', sa.Integer, server_default='0'),
        sa.Column('strength_score', sa.Float),  # Computed in application

        # Provenance
        sa.Column('source_segment_id', sa.Text),
        sa.Column('extraction_pattern', sa.Text),
        sa.Column('extraction_method', sa.Text, server_default="'pattern'"),

        # Attribution
        sa.Column('created_by_agent_id', sa.Text),

        sa.Column('status', sa.Text, server_default="'active'"),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()')),
    )

    op.create_index('idx_causal_cause', 'causal_relations',
                    ['cause_entity_id'],
                    postgresql_where=sa.text("status = 'active'"))

    op.create_index('idx_causal_effect', 'causal_relations',
                    ['effect_entity_id'],
                    postgresql_where=sa.text("status = 'active'"))

    op.create_index('idx_causal_strength', 'causal_relations',
                    [sa.text('strength_score DESC')],
                    postgresql_where=sa.text("status = 'active'"))

    # ==========================================================================
    # 5. MEMORY CLUSTERS - Consolidation Groups
    # ==========================================================================
    op.create_table(
        'memory_clusters',
        sa.Column('id', sa.BigInteger, primary_key=True),
        sa.Column('cluster_id', sa.Text, nullable=False, unique=True,
                  server_default=sa.text("gen_random_uuid()::text")),
        sa.Column('owner_tenant_id', sa.Text, nullable=False, index=True),

        # Cluster Identity
        sa.Column('cluster_name', sa.Text),
        sa.Column('cluster_embedding', Vector(768)),
        sa.Column('centroid_segment_id', sa.Text),

        # Members
        sa.Column('member_ids', ARRAY(sa.Text), nullable=False),
        sa.Column('member_count', sa.Integer, server_default='0'),

        # Abstraction
        sa.Column('abstraction_text', sa.Text),
        sa.Column('abstraction_embedding', Vector(768)),

        # Stability & Learning
        sa.Column('stability_score', sa.Float, server_default='0.5'),
        sa.Column('repetition_count', sa.Integer, server_default='1'),
        sa.Column('last_reinforced_at', sa.DateTime(timezone=True)),

        # Decay
        sa.Column('last_accessed_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()')),
        sa.Column('access_count', sa.Integer, server_default='0'),
        sa.Column('decay_rate', sa.Float, server_default='0.01'),
        sa.Column('current_decay', sa.Float, server_default='1.0'),

        # Inferences
        sa.Column('inference_ids', ARRAY(sa.Text)),

        sa.Column('status', sa.Text, server_default="'active'"),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()')),
    )

    op.create_index('idx_memory_clusters_stability', 'memory_clusters',
                    [sa.text('stability_score DESC')],
                    postgresql_where=sa.text("status = 'active'"))

    # ==========================================================================
    # 6. SESSIONS - Conversation State Tracking
    # ==========================================================================
    op.create_table(
        'sessions',
        sa.Column('id', sa.BigInteger, primary_key=True),
        sa.Column('session_id', sa.Text, nullable=False, unique=True),
        sa.Column('tenant_id', sa.Text, nullable=False, index=True),
        sa.Column('agent_id', sa.Text),
        sa.Column('user_id', sa.Text),

        # State Machine
        sa.Column('conversation_state', sa.Text, server_default="'idle'"),
        sa.Column('state_confidence', sa.Float, server_default='0.5'),
        sa.Column('state_history', JSONB, server_default="'[]'"),

        # Active Context
        sa.Column('active_entities', ARRAY(sa.Text)),
        sa.Column('active_topics', ARRAY(sa.Text)),
        sa.Column('hot_context_ids', ARRAY(sa.Text)),

        # Metrics
        sa.Column('message_count', sa.Integer, server_default='0'),
        sa.Column('retrieval_count', sa.Integer, server_default='0'),
        sa.Column('learning_signals_positive', sa.Integer, server_default='0'),
        sa.Column('learning_signals_negative', sa.Integer, server_default='0'),

        # Lifecycle
        sa.Column('started_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()')),
        sa.Column('last_activity_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()')),
        sa.Column('ended_at', sa.DateTime(timezone=True)),
    )

    op.create_index('idx_sessions_tenant_activity', 'sessions',
                    ['tenant_id', sa.text('last_activity_at DESC')])

    op.create_index('idx_sessions_active', 'sessions',
                    [sa.text('last_activity_at DESC')],
                    postgresql_where=sa.text('ended_at IS NULL'))

    # ==========================================================================
    # 7. SUBCONSCIOUS EVENTS - Daemon Audit Trail
    # ==========================================================================
    op.create_table(
        'subconscious_events',
        sa.Column('id', sa.BigInteger, primary_key=True),
        sa.Column('event_id', sa.Text, nullable=False, unique=True,
                  server_default=sa.text("gen_random_uuid()::text")),
        sa.Column('session_id', sa.Text, index=True),

        sa.Column('event_type', sa.Text, nullable=False),

        # State Classification
        sa.Column('detected_state', sa.Text),
        sa.Column('state_confidence', sa.Float),
        sa.Column('previous_state', sa.Text),

        # Predictive Retrieval
        sa.Column('predicted_entities', ARRAY(sa.Text)),
        sa.Column('predicted_topics', ARRAY(sa.Text)),
        sa.Column('preloaded_segment_ids', ARRAY(sa.Text)),
        sa.Column('prediction_confidence', sa.Float),

        # Context Injection
        sa.Column('injected_context', sa.Text),
        sa.Column('injection_token_count', sa.Integer),
        sa.Column('injection_accepted', sa.Boolean),

        # Performance
        sa.Column('latency_ms', sa.Integer),

        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()')),
    )

    op.create_index('idx_subconscious_type_created', 'subconscious_events',
                    ['event_type', sa.text('created_at DESC')])

    # ==========================================================================
    # 8. AGENT TRUST HISTORY - Trust Score Tracking
    # ==========================================================================
    op.create_table(
        'agent_trust_history',
        sa.Column('id', sa.BigInteger, primary_key=True),
        sa.Column('agent_id', sa.Text, nullable=False, index=True),
        sa.Column('accuracy_score', sa.Float),
        sa.Column('verified_claims', sa.Integer),
        sa.Column('disputed_claims', sa.Integer),
        sa.Column('recorded_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()')),
    )

    # ==========================================================================
    # 9. ATTRIBUTION COLUMNS ON EXISTING TABLES
    # ==========================================================================

    # Segments table additions
    for col_name, col_type, default in [
        ('created_by_agent_id', sa.Text(), None),
        ('created_by_user_id', sa.Text(), None),
        ('derived_from', ARRAY(sa.Text), None),
        ('derivation_type', sa.Text(), None),
        ('stability_score', sa.Float(), '0.5'),
        ('decay_score', sa.Float(), '1.0'),
        ('cluster_id', sa.Text(), None),
        ('last_reinforced_at', sa.DateTime(timezone=True), None),
    ]:
        try:
            op.add_column('segments', sa.Column(col_name, col_type, server_default=default))
        except Exception:
            pass  # Column may already exist

    # Entities table additions
    for col_name, col_type in [
        ('created_by_agent_id', sa.Text()),
        ('created_by_user_id', sa.Text()),
    ]:
        try:
            op.add_column('entities', sa.Column(col_name, col_type))
        except Exception:
            pass

    # Claims table additions
    for col_name, col_type in [
        ('created_by_agent_id', sa.Text()),
        ('created_by_user_id', sa.Text()),
        ('derived_from', ARRAY(sa.Text)),
        ('derivation_type', sa.Text()),
    ]:
        try:
            op.add_column('claims', sa.Column(col_name, col_type))
        except Exception:
            pass

    # Relations table additions
    try:
        op.add_column('relations', sa.Column('created_by_agent_id', sa.Text()))
    except Exception:
        pass

    # Assistant memories table additions
    for col_name, col_type in [
        ('created_by_agent_id', sa.Text()),
        ('created_by_user_id', sa.Text()),
        ('derived_from', ARRAY(sa.Text)),
        ('cluster_id', sa.Text()),
    ]:
        try:
            op.add_column('assistant_memories', sa.Column(col_name, col_type))
        except Exception:
            pass

    # ==========================================================================
    # 10. VECTOR INDEXES FOR NEW EMBEDDING COLUMNS
    # ==========================================================================

    try:
        op.execute("""
            CREATE INDEX IF NOT EXISTS idx_memory_clusters_embedding
            ON memory_clusters USING ivfflat (cluster_embedding vector_cosine_ops)
            WITH (lists = 100)
        """)
    except Exception:
        pass

    try:
        op.execute("""
            CREATE INDEX IF NOT EXISTS idx_memory_clusters_abstraction
            ON memory_clusters USING ivfflat (abstraction_embedding vector_cosine_ops)
            WITH (lists = 100)
        """)
    except Exception:
        pass

    try:
        op.execute("""
            CREATE INDEX IF NOT EXISTS idx_retrieval_logs_embedding
            ON retrieval_logs USING ivfflat (query_embedding vector_cosine_ops)
            WITH (lists = 100)
        """)
    except Exception:
        pass


def downgrade():
    """Remove innovation extensions schema."""

    # Drop new tables
    op.drop_table('agent_trust_history')
    op.drop_table('subconscious_events')
    op.drop_table('sessions')
    op.drop_table('memory_clusters')
    op.drop_table('causal_relations')
    op.drop_table('prompt_templates')
    op.drop_table('agent_configs')
    op.drop_table('retrieval_logs')

    # Note: Attribution columns are NOT dropped to preserve data
    # Manual removal required if needed
