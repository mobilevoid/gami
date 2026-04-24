"""
GAMI Multi-Manifold Embeddings Migration

Creates manifold_embeddings table for true multi-dimensional manifold architecture.
Each object can have up to 6 separate embeddings (one per manifold type).

Manifold Types:
- TOPIC: What is this about?
- CLAIM: What facts does it assert?
- PROCEDURE: How-to/steps content
- RELATION: Graph topology encoded
- TIME: Temporal context
- EVIDENCE: Authority/confidence

Revision ID: 014_multi_manifold
Revises: 013_innovation
Create Date: 2026-04-11
"""

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

# Revision identifiers
revision = '014_multi_manifold'
down_revision = '013_innovation'
branch_labels = None
depends_on = None


def upgrade():
    """Create multi-manifold embedding infrastructure."""

    # ==========================================================================
    # 1. MANIFOLD EMBEDDINGS TABLE
    # ==========================================================================
    # Stores multiple embeddings per object (one per manifold type)
    op.create_table(
        'manifold_embeddings',
        sa.Column('id', sa.BigInteger, primary_key=True),
        sa.Column('target_id', sa.Text, nullable=False),
        sa.Column('target_type', sa.Text, nullable=False),  # segment, claim, entity, memory
        sa.Column('manifold_type', sa.Text, nullable=False),  # TOPIC, CLAIM, PROCEDURE, RELATION, TIME, EVIDENCE

        # Primary embedding (768d for topic/claim/procedure)
        sa.Column('embedding', Vector(768)),

        # The text that was embedded (for debugging/verification)
        sa.Column('canonical_form', sa.Text),

        # Embedding metadata
        sa.Column('embedding_model', sa.Text, server_default="'nomic-embed-text'"),
        sa.Column('embedding_version', sa.Integer, server_default='1'),

        # Confidence/quality scores
        sa.Column('confidence_score', sa.Float, server_default='0.5'),

        # Timestamps
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()')),
    )

    # Unique constraint: one embedding per (target, type, manifold)
    op.create_unique_constraint(
        'uq_manifold_embeddings_target_manifold',
        'manifold_embeddings',
        ['target_id', 'target_type', 'manifold_type']
    )

    # Indexes for lookups
    op.create_index('idx_me_target', 'manifold_embeddings', ['target_id', 'target_type'])
    op.create_index('idx_me_manifold_type', 'manifold_embeddings', ['manifold_type'])

    # ==========================================================================
    # 2. CANONICAL FORMS TABLES
    # ==========================================================================
    # Store the canonical text forms used for embedding different manifolds

    op.create_table(
        'canonical_claims',
        sa.Column('id', sa.BigInteger, primary_key=True),
        sa.Column('claim_id', sa.Text, nullable=False, unique=True),

        # SPO structure
        sa.Column('subject', sa.Text, nullable=False),
        sa.Column('predicate', sa.Text, nullable=False),
        sa.Column('object', sa.Text),

        # Modifiers
        sa.Column('modality', sa.Text, server_default="'factual'"),
        sa.Column('temporal_scope', sa.Text),

        # Full canonical text for embedding
        sa.Column('canonical_text', sa.Text, nullable=False),

        sa.Column('confidence', sa.Float, server_default='0.5'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()')),
    )

    op.create_index('idx_canonical_claims_subject', 'canonical_claims', ['subject'])

    op.create_table(
        'canonical_procedures',
        sa.Column('id', sa.BigInteger, primary_key=True),
        sa.Column('procedure_id', sa.Text, nullable=False, unique=True),

        # Procedure structure
        sa.Column('action', sa.Text, nullable=False),
        sa.Column('target', sa.Text),
        sa.Column('steps_json', sa.JSON, server_default='[]'),

        # Full canonical text for embedding
        sa.Column('canonical_text', sa.Text, nullable=False),

        sa.Column('confidence', sa.Float, server_default='0.5'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()')),
    )

    op.create_index('idx_canonical_procedures_action', 'canonical_procedures', ['action'])

    # ==========================================================================
    # 3. TEMPORAL FEATURES TABLE
    # ==========================================================================
    # Stores 12-dimensional temporal features for TIME manifold

    op.create_table(
        'temporal_features',
        sa.Column('id', sa.BigInteger, primary_key=True),
        sa.Column('target_id', sa.Text, nullable=False),
        sa.Column('target_type', sa.Text, nullable=False),

        # 12 temporal features (stored as array for flexibility)
        sa.Column('features', sa.ARRAY(sa.Float), nullable=False),

        # Extracted temporal markers
        sa.Column('has_timestamp', sa.Boolean, server_default='FALSE'),
        sa.Column('timestamp_value', sa.DateTime(timezone=True)),
        sa.Column('temporal_keywords', sa.ARRAY(sa.Text)),

        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()')),
    )

    op.create_unique_constraint(
        'uq_temporal_features_target',
        'temporal_features',
        ['target_id', 'target_type']
    )

    # ==========================================================================
    # 4. GRAPH FINGERPRINTS TABLE
    # ==========================================================================
    # Stores graph fingerprints for RELATION manifold projection

    op.create_table(
        'graph_fingerprints',
        sa.Column('id', sa.BigInteger, primary_key=True),
        sa.Column('target_id', sa.Text, nullable=False),
        sa.Column('target_type', sa.Text, nullable=False),

        # 64-dimensional fingerprint
        sa.Column('fingerprint', sa.ARRAY(sa.Float), nullable=False),

        # Summary statistics
        sa.Column('in_degree', sa.Integer, server_default='0'),
        sa.Column('out_degree', sa.Integer, server_default='0'),
        sa.Column('edge_types_json', sa.JSON, server_default='{}'),

        sa.Column('computed_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()')),
    )

    op.create_unique_constraint(
        'uq_graph_fingerprints_target',
        'graph_fingerprints',
        ['target_id', 'target_type']
    )

    # ==========================================================================
    # 5. ADD COLUMNS TO SEGMENTS FOR MANIFOLD TRACKING
    # ==========================================================================
    op.add_column('segments', sa.Column('manifold_status', sa.Text, server_default="'pending'"))
    op.add_column('segments', sa.Column('manifolds_populated', sa.ARRAY(sa.Text), server_default='{}'))


def downgrade():
    """Remove multi-manifold infrastructure."""

    # Remove columns from segments
    op.drop_column('segments', 'manifolds_populated')
    op.drop_column('segments', 'manifold_status')

    # Drop tables
    op.drop_table('graph_fingerprints')
    op.drop_table('temporal_features')
    op.drop_table('canonical_procedures')
    op.drop_table('canonical_claims')
    op.drop_table('manifold_embeddings')
