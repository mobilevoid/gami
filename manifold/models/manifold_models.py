"""SQLAlchemy models for the multi-manifold memory system.

These models define the database schema for:
- Manifold embeddings (multi-space vectors)
- Canonical forms (structured claims and procedures)
- Temporal features (time-based retrieval)
- Promotion scores (deciding what gets manifold treatment)
- Query logs (for retrieval frequency tracking)
- Shadow comparisons (A/B testing new vs old retrieval)

WARNING: These models are NOT connected to the running the application system.
The tables do not exist until migrations are explicitly run.
"""
from datetime import datetime
from typing import Optional, List
from enum import Enum

from sqlalchemy import (
    Column,
    Integer,
    BigInteger,
    String,
    Text,
    Float,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    UniqueConstraint,
    Computed,
    ARRAY,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base, relationship

# Use a separate Base to avoid any connection to existing the application models
ManifoldBase = declarative_base()


class ManifoldType(str, Enum):
    """Types of semantic manifolds."""
    TOPIC = "topic"
    CLAIM = "claim"
    PROCEDURE = "procedure"
    RELATION = "relation"
    TIME = "time"
    EVIDENCE = "evidence"


class TargetType(str, Enum):
    """Types of objects that can have manifold embeddings."""
    SEGMENT = "segment"
    CLAIM = "claim"
    ENTITY = "entity"
    SUMMARY = "summary"
    MEMORY = "memory"
    PROCEDURE = "procedure"


class PromotionStatus(str, Enum):
    """Promotion status for objects."""
    RAW = "raw"              # No manifold treatment
    PROVISIONAL = "provisional"  # Basic manifold (topic only)
    PROMOTED = "promoted"    # Full manifold treatment
    MANIFOLD = "manifold"    # All applicable manifolds embedded


class ManifoldEmbedding(ManifoldBase):
    """Multi-manifold embeddings for memory objects.

    Each object can have multiple embeddings, one per manifold type.
    The embedding captures similarity in that specific semantic space.
    """
    __tablename__ = "manifold_embeddings"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    target_id = Column(String(64), nullable=False, index=True)
    target_type = Column(String(32), nullable=False)
    manifold_type = Column(String(32), nullable=False, index=True)

    # The actual embedding vector (768d for nomic-embed-text)
    # Stored as TEXT to avoid pgvector dependency in model definition
    # Cast to vector(768) in actual SQL
    embedding = Column(Text, nullable=True)

    embedding_model = Column(String(64), default="nomic-embed-text")
    embedding_version = Column(Integer, default=1)

    # The canonical form that was embedded (for debugging/verification)
    canonical_form = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("target_id", "target_type", "manifold_type",
                        name="uq_manifold_embeddings_target_manifold"),
        Index("idx_manifold_embeddings_target", "target_id", "target_type"),
        Index("idx_manifold_embeddings_type", "manifold_type"),
    )

    def __repr__(self):
        return f"<ManifoldEmbedding {self.target_type}:{self.target_id} [{self.manifold_type}]>"


class CanonicalClaim(ManifoldBase):
    """Structured SPO form of claims for the claim manifold.

    Converts prose claims like "The Vietnamese Communist Party exercised
    significant control over the CPK" into structured form:
    - subject: "Vietnamese Communist Party"
    - predicate: "exercised control over"
    - object: "CPK"
    - modality: "factual"
    - qualifiers: ["significant"]
    """
    __tablename__ = "canonical_claims"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    claim_id = Column(String(64), nullable=False, unique=True, index=True)

    # SPO structure
    subject = Column(Text, nullable=False, index=True)
    predicate = Column(Text, nullable=False, index=True)
    object = Column(Text, nullable=True)

    # Modifiers
    modality = Column(String(32), default="factual")  # factual, possible, negated
    qualifiers = Column(JSONB, default=list)
    temporal_scope = Column(Text, nullable=True)

    # Full canonical text for embedding
    canonical_text = Column(Text, nullable=False)

    confidence = Column(Float, default=0.5)
    extraction_method = Column(String(32), default="llm")

    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<CanonicalClaim {self.subject} | {self.predicate} | {self.object}>"

    def to_canonical_text(self) -> str:
        """Generate the canonical text form for embedding."""
        parts = [f"[{self.subject}]", "|", f"[{self.predicate}]"]
        if self.object:
            parts.extend(["|", f"[{self.object}]"])
        parts.append(f"| modality={self.modality}")
        if self.qualifiers:
            parts.append(f"| qualifiers={self.qualifiers}")
        if self.temporal_scope:
            parts.append(f"| time=[{self.temporal_scope}]")
        return " ".join(parts)


class CanonicalProcedure(ManifoldBase):
    """Structured form of procedures for the procedure manifold.

    Extracts instructional content into structured form:
    - title: "Deploy the application to Production"
    - prerequisites: ["PostgreSQL 16", "Redis 7"]
    - steps: [{"order": 1, "text": "Run migrations"}, ...]
    - expected_outcome: "the application accessible on :9000"
    """
    __tablename__ = "canonical_procedures"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    source_id = Column(String(64), nullable=True, index=True)
    segment_id = Column(String(64), nullable=True, index=True)

    title = Column(Text, nullable=False, index=True)
    prerequisites = Column(JSONB, default=list)
    steps = Column(JSONB, nullable=False)  # Array of {order, text, optional}
    expected_outcome = Column(Text, nullable=True)

    # Full canonical text for embedding
    canonical_text = Column(Text, nullable=False)

    owner_tenant_id = Column(String(64), nullable=False, index=True)
    confidence = Column(Float, default=0.5)

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("source_id", "segment_id", "title",
                        name="uq_canonical_procedures_source_title"),
    )

    def __repr__(self):
        return f"<CanonicalProcedure {self.title}>"

    def to_canonical_text(self) -> str:
        """Generate the canonical text form for embedding."""
        parts = [f"title=[{self.title}]"]
        if self.prerequisites:
            parts.append(f"prerequisites=[{', '.join(self.prerequisites)}]")
        if self.steps:
            step_texts = [f"{s.get('order', i+1)}. {s.get('text', '')}"
                         for i, s in enumerate(self.steps)]
            parts.append(f"steps=[{'; '.join(step_texts)}]")
        if self.expected_outcome:
            parts.append(f"outcome=[{self.expected_outcome}]")
        return " | ".join(parts)


class TemporalFeatures(ManifoldBase):
    """Structured temporal features for the time manifold.

    Instead of embedding dates as text, we extract structured features:
    - year_normalized: 0-1 scaled year
    - month_sin/cos: cyclical month encoding
    - day_sin/cos: cyclical day encoding
    - is_range: whether this is a date range
    - range_days: duration if range
    - sequence_position: position within source document
    - has_explicit_date: whether date was explicit vs inferred
    - is_relative: "last week" vs "March 15"
    - temporal_precision: year/month/day/hour
    - is_ongoing: "since 2020"
    """
    __tablename__ = "temporal_features"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    target_id = Column(String(64), nullable=False, index=True)
    target_type = Column(String(32), nullable=False)

    # 12-dimensional feature vector
    features = Column(ARRAY(Float), nullable=False)

    # Original text that was parsed
    raw_temporal_text = Column(Text, nullable=True)

    # Parsed dates (if available)
    start_date = Column(DateTime, nullable=True, index=True)
    end_date = Column(DateTime, nullable=True, index=True)

    is_range = Column(Boolean, default=False)
    precision = Column(String(16), default="day")  # year, month, day, hour
    is_relative = Column(Boolean, default=False)

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("target_id", "target_type",
                        name="uq_temporal_features_target"),
        Index("idx_temporal_features_dates", "start_date", "end_date"),
    )

    def __repr__(self):
        return f"<TemporalFeatures {self.target_type}:{self.target_id}>"


class PromotionScore(ManifoldBase):
    """Promotion scores for deciding manifold treatment.

    Objects with higher scores get richer manifold embeddings.
    The formula is:
        total = 0.25*importance + 0.20*retrieval_frequency + 0.15*source_diversity
              + 0.15*confidence + 0.10*novelty + 0.10*graph_centrality
              + 0.05*user_relevance

    Thresholds:
    - < 0.45: keep raw only (no manifold embeddings)
    - 0.45-0.70: provisional (topic manifold only)
    - > 0.70: promoted (topic + applicable specialized manifolds)
    - > 0.85: full manifold treatment
    """
    __tablename__ = "promotion_scores"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    target_id = Column(String(64), nullable=False, index=True)
    target_type = Column(String(32), nullable=False)

    # Individual score components
    importance = Column(Float, default=0.5)
    retrieval_frequency = Column(Float, default=0.0)
    source_diversity = Column(Float, default=0.0)
    confidence = Column(Float, default=0.5)
    novelty = Column(Float, default=0.5)
    graph_centrality = Column(Float, default=0.0)
    user_relevance = Column(Float, default=0.0)

    # Computed total score
    # Note: In real DB, this would be a GENERATED ALWAYS AS column
    # Here we compute it in Python
    total_score = Column(Float, default=0.0)

    promotion_status = Column(String(32), default="raw", index=True)

    computed_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("target_id", "target_type",
                        name="uq_promotion_scores_target"),
        Index("idx_promotion_scores_total", "total_score"),
    )

    def compute_total(self) -> float:
        """Compute the weighted total promotion score."""
        return (
            0.25 * (self.importance or 0.5)
            + 0.20 * (self.retrieval_frequency or 0.0)
            + 0.15 * (self.source_diversity or 0.0)
            + 0.15 * (self.confidence or 0.5)
            + 0.10 * (self.novelty or 0.5)
            + 0.10 * (self.graph_centrality or 0.0)
            + 0.05 * (self.user_relevance or 0.0)
        )

    def determine_status(self) -> str:
        """Determine promotion status based on total score."""
        total = self.total_score or self.compute_total()
        if total >= 0.85:
            return "manifold"
        elif total >= 0.70:
            return "promoted"
        elif total >= 0.45:
            return "provisional"
        else:
            return "raw"

    def __repr__(self):
        return f"<PromotionScore {self.target_type}:{self.target_id} = {self.total_score:.3f}>"


class QueryLog(ManifoldBase):
    """Query logs for tracking retrieval frequency.

    Used to compute retrieval_frequency component of promotion scores.
    Also useful for analyzing query patterns and manifold weight effectiveness.
    """
    __tablename__ = "query_logs"

    id = Column(BigInteger, primary_key=True, autoincrement=True)

    query_text = Column(Text, nullable=False)
    query_hash = Column(String(64), nullable=False, index=True)

    query_mode = Column(String(32), nullable=True)
    tenant_ids = Column(ARRAY(String), nullable=True)

    # Results returned
    result_ids = Column(ARRAY(String), nullable=True)
    result_scores = Column(ARRAY(Float), nullable=True)

    latency_ms = Column(Integer, nullable=True)

    # Manifold weights used (for analysis)
    manifold_weights = Column(JSONB, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def __repr__(self):
        return f"<QueryLog {self.query_hash[:8]} @ {self.created_at}>"


class ManifoldConfig(ManifoldBase):
    """Per-tenant manifold configuration.

    Allows tuning manifold weights, thresholds, and behavior per tenant.
    """
    __tablename__ = "manifold_config"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tenant_id = Column(String(64), nullable=False, index=True)
    config_key = Column(String(64), nullable=False)
    config_value = Column(JSONB, nullable=False)
    description = Column(Text, nullable=True)

    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("tenant_id", "config_key",
                        name="uq_manifold_config_tenant_key"),
    )

    def __repr__(self):
        return f"<ManifoldConfig {self.tenant_id}:{self.config_key}>"


class ShadowComparison(ManifoldBase):
    """Shadow comparison results for A/B testing retrieval paths.

    When shadow mode is enabled, both old (v1) and new (v2) retrieval
    run in parallel. Results are compared here for analysis.
    """
    __tablename__ = "shadow_comparisons"

    id = Column(BigInteger, primary_key=True, autoincrement=True)

    query_hash = Column(String(64), nullable=False, index=True)
    query_text = Column(Text, nullable=True)

    # Old (v1) results
    old_result_ids = Column(ARRAY(String), nullable=True)
    old_scores = Column(ARRAY(Float), nullable=True)

    # New (v2) results
    new_result_ids = Column(ARRAY(String), nullable=True)
    new_scores = Column(ARRAY(Float), nullable=True)

    # Comparison metrics
    overlap_count = Column(Integer, nullable=True)
    rank_correlation = Column(Float, nullable=True)

    latency_old_ms = Column(Integer, nullable=True)
    latency_new_ms = Column(Integer, nullable=True)

    winner = Column(String(16), nullable=True)  # 'old', 'new', 'tie', 'unknown'
    notes = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def __repr__(self):
        return f"<ShadowComparison {self.query_hash[:8]} -> {self.winner}>"
