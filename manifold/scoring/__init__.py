"""Scoring modules for manifold system."""
from .promotion import (
    compute_promotion_score,
    PromotionFactors,
    should_promote,
    should_demote,
    PROMOTION_THRESHOLD,
    DEMOTION_THRESHOLD,
)
from .evidence import (
    compute_evidence_score,
    EvidenceFactors,
    EvidenceVector,
    compute_source_authority,
    compute_specificity,
)
from .relation import (
    GraphFingerprint,
    compute_graph_fingerprint,
    fingerprint_similarity,
    compute_relation_score,
    find_related_entities,
)

__all__ = [
    # Promotion
    "compute_promotion_score",
    "PromotionFactors",
    "should_promote",
    "should_demote",
    "PROMOTION_THRESHOLD",
    "DEMOTION_THRESHOLD",
    # Evidence
    "compute_evidence_score",
    "EvidenceFactors",
    "EvidenceVector",
    "compute_source_authority",
    "compute_specificity",
    # Relation
    "GraphFingerprint",
    "compute_graph_fingerprint",
    "fingerprint_similarity",
    "compute_relation_score",
    "find_related_entities",
]
