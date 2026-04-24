"""Pydantic schemas for the multi-manifold memory system.

These schemas define the API contracts for manifold operations.
They are used for validation, serialization, and documentation.
"""
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field, validator


class ManifoldType(str, Enum):
    """Types of semantic manifolds."""
    TOPIC = "topic"
    CLAIM = "claim"
    PROCEDURE = "procedure"
    RELATION = "relation"
    TIME = "time"
    EVIDENCE = "evidence"


class QueryModeV2(str, Enum):
    """Query modes for manifold-aware retrieval."""
    AUTO = "auto"
    FACT_LOOKUP = "fact_lookup"
    SYNTHESIS = "synthesis"
    COMPARISON = "comparison"
    TIMELINE = "timeline"
    PROCEDURE = "procedure"
    ASSISTANT_MEMORY = "assistant_memory"
    VERIFICATION = "verification"
    REPORT = "report"


class TargetType(str, Enum):
    """Types of objects that can have manifold treatment."""
    SEGMENT = "segment"
    CLAIM = "claim"
    ENTITY = "entity"
    SUMMARY = "summary"
    MEMORY = "memory"
    PROCEDURE = "procedure"


class PromotionStatus(str, Enum):
    """Promotion status for objects."""
    RAW = "raw"
    PROVISIONAL = "provisional"
    PROMOTED = "promoted"
    MANIFOLD = "manifold"


class Modality(str, Enum):
    """Claim modality."""
    FACTUAL = "factual"
    POSSIBLE = "possible"
    NEGATED = "negated"


# ---------------------------------------------------------------------------
# Manifold Weight Schemas
# ---------------------------------------------------------------------------

class ManifoldWeights(BaseModel):
    """Weights for each manifold in anchor score computation."""
    topic: float = Field(default=0.25, ge=0.0, le=1.0)
    claim: float = Field(default=0.20, ge=0.0, le=1.0)
    procedure: float = Field(default=0.10, ge=0.0, le=1.0)
    relation: float = Field(default=0.15, ge=0.0, le=1.0)
    time: float = Field(default=0.10, ge=0.0, le=1.0)
    evidence: float = Field(default=0.20, ge=0.0, le=1.0)

    @validator("*", pre=True)
    def ensure_float(cls, v):
        if v is None:
            return 0.0
        return float(v)

    def normalize(self) -> "ManifoldWeights":
        """Normalize weights to sum to 1.0."""
        total = self.topic + self.claim + self.procedure + self.relation + self.time + self.evidence
        if total == 0:
            return ManifoldWeights()
        return ManifoldWeights(
            topic=self.topic / total,
            claim=self.claim / total,
            procedure=self.procedure / total,
            relation=self.relation / total,
            time=self.time / total,
            evidence=self.evidence / total,
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "topic": self.topic,
            "claim": self.claim,
            "procedure": self.procedure,
            "relation": self.relation,
            "time": self.time,
            "evidence": self.evidence,
        }


# Default weights per query mode
ALPHA_WEIGHTS: Dict[QueryModeV2, ManifoldWeights] = {
    QueryModeV2.FACT_LOOKUP: ManifoldWeights(
        topic=0.15, claim=0.35, procedure=0.05,
        relation=0.20, time=0.05, evidence=0.20
    ),
    QueryModeV2.SYNTHESIS: ManifoldWeights(
        topic=0.35, claim=0.20, procedure=0.10,
        relation=0.20, time=0.05, evidence=0.10
    ),
    QueryModeV2.COMPARISON: ManifoldWeights(
        topic=0.25, claim=0.25, procedure=0.05,
        relation=0.25, time=0.05, evidence=0.15
    ),
    QueryModeV2.TIMELINE: ManifoldWeights(
        topic=0.10, claim=0.15, procedure=0.00,
        relation=0.15, time=0.40, evidence=0.20
    ),
    QueryModeV2.PROCEDURE: ManifoldWeights(
        topic=0.15, claim=0.10, procedure=0.45,
        relation=0.15, time=0.00, evidence=0.15
    ),
    QueryModeV2.ASSISTANT_MEMORY: ManifoldWeights(
        topic=0.20, claim=0.20, procedure=0.15,
        relation=0.10, time=0.15, evidence=0.20
    ),
    QueryModeV2.VERIFICATION: ManifoldWeights(
        topic=0.10, claim=0.20, procedure=0.00,
        relation=0.10, time=0.10, evidence=0.50
    ),
    QueryModeV2.REPORT: ManifoldWeights(
        topic=0.30, claim=0.20, procedure=0.15,
        relation=0.20, time=0.05, evidence=0.10
    ),
}

def get_alpha_weights(mode: QueryModeV2) -> ManifoldWeights:
    """Get manifold weights for a query mode."""
    return ALPHA_WEIGHTS.get(mode, ALPHA_WEIGHTS[QueryModeV2.FACT_LOOKUP])


# ---------------------------------------------------------------------------
# Canonical Form Schemas
# ---------------------------------------------------------------------------

class CanonicalClaimSchema(BaseModel):
    """Schema for canonical claim (SPO form)."""
    claim_id: Optional[str] = None
    subject: str
    predicate: str
    object: Optional[str] = None
    modality: Modality = Modality.FACTUAL
    qualifiers: List[str] = Field(default_factory=list)
    temporal_scope: Optional[str] = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    def to_canonical_text(self) -> str:
        """Generate the canonical text form for embedding."""
        parts = [f"[{self.subject}]", "|", f"[{self.predicate}]"]
        if self.object:
            parts.extend(["|", f"[{self.object}]"])
        parts.append(f"| modality={self.modality.value}")
        if self.qualifiers:
            parts.append(f"| qualifiers={self.qualifiers}")
        if self.temporal_scope:
            parts.append(f"| time=[{self.temporal_scope}]")
        return " ".join(parts)


class ProcedureStep(BaseModel):
    """A single step in a procedure."""
    order: int
    text: str
    optional: bool = False


class CanonicalProcedureSchema(BaseModel):
    """Schema for canonical procedure form."""
    source_id: Optional[str] = None
    segment_id: Optional[str] = None
    title: str
    prerequisites: List[str] = Field(default_factory=list)
    steps: List[ProcedureStep]
    expected_outcome: Optional[str] = None
    owner_tenant_id: str = "shared"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    def to_canonical_text(self) -> str:
        """Generate the canonical text form for embedding."""
        parts = [f"title=[{self.title}]"]
        if self.prerequisites:
            parts.append(f"prerequisites=[{', '.join(self.prerequisites)}]")
        if self.steps:
            step_texts = [f"{s.order}. {s.text}" for s in self.steps]
            parts.append(f"steps=[{'; '.join(step_texts)}]")
        if self.expected_outcome:
            parts.append(f"outcome=[{self.expected_outcome}]")
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# Promotion Score Schemas
# ---------------------------------------------------------------------------

class PromotionScoreSchema(BaseModel):
    """Schema for promotion score computation."""
    target_id: str
    target_type: TargetType

    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    retrieval_frequency: float = Field(default=0.0, ge=0.0, le=1.0)
    source_diversity: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    novelty: float = Field(default=0.5, ge=0.0, le=1.0)
    graph_centrality: float = Field(default=0.0, ge=0.0, le=1.0)
    user_relevance: float = Field(default=0.0, ge=0.0, le=1.0)

    total_score: Optional[float] = None
    promotion_status: Optional[PromotionStatus] = None

    def compute_total(self) -> float:
        """Compute the weighted total promotion score."""
        return (
            0.25 * self.importance
            + 0.20 * self.retrieval_frequency
            + 0.15 * self.source_diversity
            + 0.15 * self.confidence
            + 0.10 * self.novelty
            + 0.10 * self.graph_centrality
            + 0.05 * self.user_relevance
        )

    def determine_status(self) -> PromotionStatus:
        """Determine promotion status based on total score."""
        total = self.total_score or self.compute_total()
        if total >= 0.85:
            return PromotionStatus.MANIFOLD
        elif total >= 0.70:
            return PromotionStatus.PROMOTED
        elif total >= 0.45:
            return PromotionStatus.PROVISIONAL
        else:
            return PromotionStatus.RAW


# ---------------------------------------------------------------------------
# Query Classification Schemas
# ---------------------------------------------------------------------------

class QueryClassificationV2(BaseModel):
    """Enhanced query classification with manifold weights."""
    mode: QueryModeV2
    granularity: str = "medium"  # fine, medium, coarse
    needs_citation: bool = True
    manifold_weights: ManifoldWeights = Field(default_factory=ManifoldWeights)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    reasoning: Optional[str] = None


# ---------------------------------------------------------------------------
# Evidence and Retrieval Schemas
# ---------------------------------------------------------------------------

class ManifoldScore(BaseModel):
    """Scores from each manifold for a candidate."""
    topic: float = 0.0
    claim: float = 0.0
    procedure: float = 0.0
    relation: float = 0.0
    time: float = 0.0
    evidence: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "topic": self.topic,
            "claim": self.claim,
            "procedure": self.procedure,
            "relation": self.relation,
            "time": self.time,
            "evidence": self.evidence,
        }


class AnchorCandidate(BaseModel):
    """A candidate anchor from manifold retrieval."""
    item_id: str
    item_type: str
    text: str
    manifold_scores: ManifoldScore
    fused_score: float
    lexical_score: float = 0.0
    alias_match: float = 0.0
    cache_hit: float = 0.0
    prior_importance: float = 0.0
    duplicate_penalty: float = 0.0
    noise_penalty: float = 0.0
    final_anchor_score: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RecallRequestV2(BaseModel):
    """Request schema for manifold-aware recall."""
    query: str
    tenant_id: str = "default"
    tenant_ids: Optional[List[str]] = None
    max_tokens: int = Field(default=2000, ge=100, le=16000)
    mode: QueryModeV2 = QueryModeV2.AUTO
    manifold_override: Optional[ManifoldWeights] = None
    explain: bool = False


class RecallResponseV2(BaseModel):
    """Response schema for manifold-aware recall."""
    query: str
    mode: QueryModeV2
    manifold_weights: ManifoldWeights
    evidence: List[AnchorCandidate]
    context_text: str
    total_tokens_used: int
    total_candidates: int
    classification_ms: float = 0.0
    retrieval_ms: float = 0.0
    total_ms: float = 0.0
    explain_data: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Shadow Comparison Schemas
# ---------------------------------------------------------------------------

class ShadowComparisonResult(BaseModel):
    """Result of comparing old vs new retrieval paths."""
    query_hash: str
    query_text: str

    old_result_ids: List[str]
    old_scores: List[float]
    new_result_ids: List[str]
    new_scores: List[float]

    overlap_count: int
    rank_correlation: Optional[float] = None

    latency_old_ms: int
    latency_new_ms: int

    winner: str  # 'old', 'new', 'tie', 'unknown'
    notes: Optional[str] = None
