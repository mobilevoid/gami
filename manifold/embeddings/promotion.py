"""Promotion scoring — decides which objects get manifold treatment.

Not all objects need embeddings in all manifolds. The promotion scorer
computes a weighted score based on:

- importance: how important the object is (0-1)
- retrieval_frequency: how often it's retrieved (0-1, normalized)
- source_diversity: how many sources support it (0-1)
- confidence: extraction confidence (0-1)
- novelty: how novel the information is (0-1)
- graph_centrality: centrality in knowledge graph (0-1)
- user_relevance: relevance to user's interests (0-1)

The formula:
    total = 0.25*importance + 0.20*retrieval_frequency + 0.15*source_diversity
          + 0.15*confidence + 0.10*novelty + 0.10*graph_centrality
          + 0.05*user_relevance

Thresholds:
- < 0.45: raw only (no manifold embeddings)
- 0.45-0.70: provisional (topic manifold only)
- > 0.70: promoted (topic + specialized manifolds)
- > 0.85: full manifold treatment
"""
import logging
import math
from typing import Optional, List, Dict, Any, Tuple

from ..models.schemas import (
    PromotionScoreSchema,
    PromotionStatus,
    TargetType,
    ManifoldType,
)

logger = logging.getLogger("manifold.embeddings.promotion")


# Promotion thresholds
THRESHOLD_PROVISIONAL = 0.45
THRESHOLD_PROMOTED = 0.70
THRESHOLD_MANIFOLD = 0.85


# Weight factors for promotion score
PROMOTION_WEIGHTS = {
    "importance": 0.25,
    "retrieval_frequency": 0.20,
    "source_diversity": 0.15,
    "confidence": 0.15,
    "novelty": 0.10,
    "graph_centrality": 0.10,
    "user_relevance": 0.05,
}


class PromotionScorer:
    """Computes promotion scores for objects.

    Uses a weighted formula to determine which objects should receive
    rich manifold treatment vs staying as raw segments.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        thresholds: Optional[Dict[str, float]] = None,
    ):
        """Initialize the promotion scorer.

        Args:
            weights: Custom weight factors (default: PROMOTION_WEIGHTS).
            thresholds: Custom promotion thresholds.
        """
        self.weights = weights or PROMOTION_WEIGHTS.copy()
        self.thresholds = thresholds or {
            "provisional": THRESHOLD_PROVISIONAL,
            "promoted": THRESHOLD_PROMOTED,
            "manifold": THRESHOLD_MANIFOLD,
        }

    def compute_score(
        self,
        target_id: str,
        target_type: TargetType,
        metrics: Dict[str, float],
    ) -> PromotionScoreSchema:
        """Compute promotion score for an object.

        Args:
            target_id: ID of the target object.
            target_type: Type of target (segment, claim, etc.).
            metrics: Dict with score components (importance, confidence, etc.).

        Returns:
            PromotionScoreSchema with computed score and status.
        """
        # Extract and clamp metrics
        importance = max(0.0, min(1.0, metrics.get("importance", 0.5)))
        retrieval_frequency = max(0.0, min(1.0, metrics.get("retrieval_frequency", 0.0)))
        source_diversity = max(0.0, min(1.0, metrics.get("source_diversity", 0.0)))
        confidence = max(0.0, min(1.0, metrics.get("confidence", 0.5)))
        novelty = max(0.0, min(1.0, metrics.get("novelty", 0.5)))
        graph_centrality = max(0.0, min(1.0, metrics.get("graph_centrality", 0.0)))
        user_relevance = max(0.0, min(1.0, metrics.get("user_relevance", 0.0)))

        # Compute weighted total
        total = (
            self.weights["importance"] * importance
            + self.weights["retrieval_frequency"] * retrieval_frequency
            + self.weights["source_diversity"] * source_diversity
            + self.weights["confidence"] * confidence
            + self.weights["novelty"] * novelty
            + self.weights["graph_centrality"] * graph_centrality
            + self.weights["user_relevance"] * user_relevance
        )

        # Determine status
        if total >= self.thresholds["manifold"]:
            status = PromotionStatus.MANIFOLD
        elif total >= self.thresholds["promoted"]:
            status = PromotionStatus.PROMOTED
        elif total >= self.thresholds["provisional"]:
            status = PromotionStatus.PROVISIONAL
        else:
            status = PromotionStatus.RAW

        return PromotionScoreSchema(
            target_id=target_id,
            target_type=target_type,
            importance=importance,
            retrieval_frequency=retrieval_frequency,
            source_diversity=source_diversity,
            confidence=confidence,
            novelty=novelty,
            graph_centrality=graph_centrality,
            user_relevance=user_relevance,
            total_score=total,
            promotion_status=status,
        )

    def get_manifolds_for_status(
        self,
        status: PromotionStatus,
        target_type: TargetType,
    ) -> List[ManifoldType]:
        """Determine which manifolds an object should be embedded into.

        Args:
            status: The promotion status.
            target_type: The type of object.

        Returns:
            List of manifold types to embed into.
        """
        if status == PromotionStatus.RAW:
            return []

        manifolds = [ManifoldType.TOPIC]  # All promoted objects get topic

        if status in (PromotionStatus.PROMOTED, PromotionStatus.MANIFOLD):
            # Add specialized manifolds based on type
            if target_type == TargetType.CLAIM:
                manifolds.append(ManifoldType.CLAIM)
            elif target_type == TargetType.PROCEDURE:
                manifolds.append(ManifoldType.PROCEDURE)
            elif target_type == TargetType.ENTITY:
                manifolds.append(ManifoldType.RELATION)
            elif target_type == TargetType.MEMORY:
                # Memories might have claims or procedures embedded
                pass

        if status == PromotionStatus.MANIFOLD:
            # Full treatment might include additional manifolds
            # based on content analysis
            pass

        return manifolds


# ---------------------------------------------------------------------------
# Metric Computation Helpers
# ---------------------------------------------------------------------------

def normalize_retrieval_count(count: int, max_count: int = 1000) -> float:
    """Normalize retrieval count to 0-1 using log scale.

    Args:
        count: Raw retrieval count.
        max_count: Expected maximum for normalization.

    Returns:
        Normalized score 0-1.
    """
    if count <= 0:
        return 0.0
    # Log normalization: log(count+1) / log(max+1)
    return min(1.0, math.log(count + 1) / math.log(max_count + 1))


def compute_source_diversity(
    source_ids: List[str],
    source_types: Optional[List[str]] = None,
) -> float:
    """Compute source diversity score.

    Args:
        source_ids: List of supporting source IDs.
        source_types: Optional list of source types.

    Returns:
        Diversity score 0-1.
    """
    if not source_ids:
        return 0.0

    unique_sources = len(set(source_ids))

    # Base score from number of unique sources
    # 1 source = 0.2, 2 = 0.4, 3 = 0.6, 5+ = 0.8+
    base_score = min(1.0, unique_sources / 5.0)

    # Bonus for type diversity
    if source_types:
        unique_types = len(set(source_types))
        type_bonus = min(0.2, unique_types / 5.0)
        base_score = min(1.0, base_score + type_bonus)

    return base_score


def compute_novelty(
    text: str,
    existing_embeddings: Optional[List[List[float]]] = None,
) -> float:
    """Compute novelty score based on distance from existing knowledge.

    Args:
        text: The text to assess.
        existing_embeddings: Embeddings of existing related content.

    Returns:
        Novelty score 0-1 (1 = completely novel).
    """
    if not existing_embeddings:
        return 0.5  # Unknown novelty

    # STUB: Would compute embedding and compare to existing
    # For now, return moderate novelty
    return 0.5


def compute_graph_centrality(
    entity_id: str,
    edge_count: int = 0,
    neighbor_importance_sum: float = 0.0,
) -> float:
    """Compute simplified graph centrality score.

    Args:
        entity_id: The entity ID.
        edge_count: Number of edges connected to this entity.
        neighbor_importance_sum: Sum of neighbor importance scores.

    Returns:
        Centrality score 0-1.
    """
    if edge_count <= 0:
        return 0.0

    # Simple degree-based centrality with importance weighting
    # More edges = more central
    degree_score = min(1.0, edge_count / 20.0)

    # Weighted by neighbor importance
    if neighbor_importance_sum > 0:
        avg_neighbor_importance = neighbor_importance_sum / edge_count
        weighted_score = degree_score * (0.5 + 0.5 * avg_neighbor_importance)
    else:
        weighted_score = degree_score * 0.5

    return min(1.0, weighted_score)


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------

def compute_promotion_score(
    target_id: str,
    target_type: TargetType,
    importance: float = 0.5,
    retrieval_count: int = 0,
    source_ids: Optional[List[str]] = None,
    confidence: float = 0.5,
    novelty: float = 0.5,
    edge_count: int = 0,
    user_relevance: float = 0.0,
) -> PromotionScoreSchema:
    """Convenience function for computing promotion score.

    Args:
        target_id: ID of the target.
        target_type: Type of target.
        importance: Importance score.
        retrieval_count: Number of times retrieved.
        source_ids: Supporting source IDs.
        confidence: Extraction confidence.
        novelty: Novelty score.
        edge_count: Graph edge count.
        user_relevance: User relevance score.

    Returns:
        PromotionScoreSchema with computed values.
    """
    scorer = PromotionScorer()

    metrics = {
        "importance": importance,
        "retrieval_frequency": normalize_retrieval_count(retrieval_count),
        "source_diversity": compute_source_diversity(source_ids or []),
        "confidence": confidence,
        "novelty": novelty,
        "graph_centrality": compute_graph_centrality("", edge_count),
        "user_relevance": user_relevance,
    }

    return scorer.compute_score(target_id, target_type, metrics)


def should_embed_manifold(
    score: PromotionScoreSchema,
    manifold_type: ManifoldType,
) -> bool:
    """Check if an object should be embedded in a specific manifold.

    Args:
        score: The promotion score.
        manifold_type: The manifold to check.

    Returns:
        True if the object should be embedded in this manifold.
    """
    scorer = PromotionScorer()
    manifolds = scorer.get_manifolds_for_status(
        score.promotion_status,
        score.target_type,
    )
    return manifold_type in manifolds


def batch_compute_promotion_scores(
    objects: List[Dict[str, Any]],
) -> List[PromotionScoreSchema]:
    """Compute promotion scores for multiple objects.

    Args:
        objects: List of dicts with target_id, target_type, and metrics.

    Returns:
        List of PromotionScoreSchema.
    """
    scorer = PromotionScorer()
    results = []

    for obj in objects:
        target_id = obj.get("target_id", "")
        target_type = TargetType(obj.get("target_type", "segment"))

        metrics = {
            "importance": obj.get("importance", 0.5),
            "retrieval_frequency": normalize_retrieval_count(
                obj.get("retrieval_count", 0)
            ),
            "source_diversity": compute_source_diversity(
                obj.get("source_ids", [])
            ),
            "confidence": obj.get("confidence", 0.5),
            "novelty": obj.get("novelty", 0.5),
            "graph_centrality": compute_graph_centrality(
                target_id,
                obj.get("edge_count", 0),
            ),
            "user_relevance": obj.get("user_relevance", 0.0),
        }

        score = scorer.compute_score(target_id, target_type, metrics)
        results.append(score)

    return results
