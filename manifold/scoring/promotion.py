"""Promotion scoring for tiered manifold embedding.

Implements the 7-factor promotion formula that determines which objects
get specialized manifold embeddings beyond basic topic vectors.

Promotion Score Formula:
    P = w_i * importance
      + w_r * retrieval_frequency
      + w_d * source_diversity
      + w_c * confidence
      + w_n * novelty
      + w_g * graph_centrality
      + w_u * user_relevance

Objects scoring above PROMOTION_THRESHOLD get specialized embeddings.
Objects scoring below DEMOTION_THRESHOLD lose specialized embeddings.
The gap between thresholds provides hysteresis to prevent oscillation.
"""
from dataclasses import dataclass
from typing import Optional

# Promotion thresholds with hysteresis gap
PROMOTION_THRESHOLD = 0.65
DEMOTION_THRESHOLD = 0.35

# Factor weights (sum to 1.0)
FACTOR_WEIGHTS = {
    "importance": 0.20,
    "retrieval_frequency": 0.15,
    "source_diversity": 0.10,
    "confidence": 0.20,
    "novelty": 0.10,
    "graph_centrality": 0.10,
    "user_relevance": 0.15,
}


@dataclass
class PromotionFactors:
    """Input factors for promotion scoring.

    All factors should be in [0, 1] range.
    """
    importance: float = 0.0
    retrieval_frequency: float = 0.0
    source_diversity: float = 0.0
    confidence: float = 0.0
    novelty: float = 0.0
    graph_centrality: float = 0.0
    user_relevance: float = 0.0


def _clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamp value to range."""
    return max(min_val, min(max_val, value))


def compute_promotion_score(factors: PromotionFactors) -> float:
    """Compute promotion score from factors.

    Args:
        factors: The 7 promotion factors, each in [0, 1].

    Returns:
        Promotion score in [0, 1].
    """
    # Clamp all factors to valid range
    importance = _clamp(factors.importance)
    retrieval_frequency = _clamp(factors.retrieval_frequency)
    source_diversity = _clamp(factors.source_diversity)
    confidence = _clamp(factors.confidence)
    novelty = _clamp(factors.novelty)
    graph_centrality = _clamp(factors.graph_centrality)
    user_relevance = _clamp(factors.user_relevance)

    # Weighted sum
    score = (
        FACTOR_WEIGHTS["importance"] * importance
        + FACTOR_WEIGHTS["retrieval_frequency"] * retrieval_frequency
        + FACTOR_WEIGHTS["source_diversity"] * source_diversity
        + FACTOR_WEIGHTS["confidence"] * confidence
        + FACTOR_WEIGHTS["novelty"] * novelty
        + FACTOR_WEIGHTS["graph_centrality"] * graph_centrality
        + FACTOR_WEIGHTS["user_relevance"] * user_relevance
    )

    return _clamp(score)


def should_promote(score: float, current_tier: int = 0) -> bool:
    """Determine if object should be promoted to higher tier.

    Args:
        score: Current promotion score.
        current_tier: Current embedding tier (0=topic only, 1=specialized).

    Returns:
        True if object should be promoted.
    """
    if current_tier >= 1:
        # Already at max tier
        return False
    return score >= PROMOTION_THRESHOLD


def should_demote(score: float, current_tier: int = 1) -> bool:
    """Determine if object should be demoted to lower tier.

    Args:
        score: Current promotion score.
        current_tier: Current embedding tier.

    Returns:
        True if object should be demoted.
    """
    if current_tier <= 0:
        # Already at min tier
        return False
    return score < DEMOTION_THRESHOLD


def compute_importance(
    retrieval_count: int,
    citation_count: int,
    recency_days: float,
    source_authority: float = 0.5,
) -> float:
    """Compute importance factor from usage metrics.

    Args:
        retrieval_count: Number of times retrieved in queries.
        citation_count: Number of times cited in responses.
        recency_days: Days since last access.
        source_authority: Authority score of source document.

    Returns:
        Importance score in [0, 1].
    """
    # Logarithmic scaling for counts
    import math

    retrieval_score = min(1.0, math.log1p(retrieval_count) / 5.0)
    citation_score = min(1.0, math.log1p(citation_count) / 4.0)

    # Recency decay (half-life of 30 days)
    recency_score = math.exp(-recency_days / 43.3)  # ln(2)/30 ≈ 0.0231

    # Combine with weights
    importance = (
        0.3 * retrieval_score
        + 0.3 * citation_score
        + 0.2 * recency_score
        + 0.2 * _clamp(source_authority)
    )

    return _clamp(importance)


def compute_graph_centrality(
    in_degree: int,
    out_degree: int,
    betweenness: float = 0.0,
) -> float:
    """Compute graph centrality factor.

    Args:
        in_degree: Number of incoming edges.
        out_degree: Number of outgoing edges.
        betweenness: Betweenness centrality if available.

    Returns:
        Graph centrality score in [0, 1].
    """
    import math

    # Degree centrality (log scaled)
    degree_score = min(1.0, math.log1p(in_degree + out_degree) / 4.0)

    # Betweenness is already normalized if provided
    betweenness_score = _clamp(betweenness)

    # Combine
    centrality = 0.6 * degree_score + 0.4 * betweenness_score

    return _clamp(centrality)


def compute_source_diversity(
    source_ids: list,
    total_sources: int,
) -> float:
    """Compute source diversity factor.

    Higher diversity = information confirmed across multiple sources.

    Args:
        source_ids: List of unique source IDs mentioning this object.
        total_sources: Total number of sources in corpus.

    Returns:
        Source diversity score in [0, 1].
    """
    if total_sources <= 1:
        return 0.5  # Neutral if only one source

    unique_sources = len(set(source_ids))
    # Logarithmic scaling
    import math

    diversity = min(1.0, math.log1p(unique_sources) / math.log1p(min(10, total_sources)))

    return _clamp(diversity)
