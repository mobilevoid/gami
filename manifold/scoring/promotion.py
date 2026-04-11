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

All weights are configurable via ManifoldConfigV2.scoring.
"""
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING
import math

if TYPE_CHECKING:
    from ..config_v2 import ScoringWeights, ManifoldConfigV2


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


def get_promotion_threshold(config: Optional["ManifoldConfigV2"] = None) -> float:
    """Get promotion threshold from config."""
    if config is None:
        from ..config import get_config
        config = get_config()
    return config.promotion_threshold


def get_demotion_threshold(config: Optional["ManifoldConfigV2"] = None) -> float:
    """Get demotion threshold from config."""
    if config is None:
        from ..config import get_config
        config = get_config()
    return config.demotion_threshold


def compute_promotion_score(
    factors: PromotionFactors,
    weights: Optional["ScoringWeights"] = None,
) -> float:
    """Compute promotion score from factors.

    Args:
        factors: The 7 promotion factors, each in [0, 1].
        weights: Optional scoring weights from config.

    Returns:
        Promotion score in [0, 1].
    """
    if weights is None:
        from ..config import get_scoring_weights
        weights = get_scoring_weights()

    # Clamp all factors to valid range
    importance = _clamp(factors.importance)
    retrieval_frequency = _clamp(factors.retrieval_frequency)
    source_diversity = _clamp(factors.source_diversity)
    confidence = _clamp(factors.confidence)
    novelty = _clamp(factors.novelty)
    graph_centrality = _clamp(factors.graph_centrality)
    user_relevance = _clamp(factors.user_relevance)

    # Weighted sum using configurable weights
    score = (
        weights.promotion_importance * importance
        + weights.promotion_retrieval * retrieval_frequency
        + weights.promotion_diversity * source_diversity
        + weights.promotion_confidence * confidence
        + weights.promotion_novelty * novelty
        + weights.promotion_centrality * graph_centrality
        + weights.promotion_relevance * user_relevance
    )

    return _clamp(score)


def should_promote(
    score: float,
    current_tier: int = 0,
    config: Optional["ManifoldConfigV2"] = None,
) -> bool:
    """Determine if object should be promoted to higher tier.

    Args:
        score: Current promotion score.
        current_tier: Current embedding tier (0=topic only, 1=specialized).
        config: Optional config for threshold.

    Returns:
        True if object should be promoted.
    """
    if current_tier >= 1:
        # Already at max tier
        return False
    return score >= get_promotion_threshold(config)


def should_demote(
    score: float,
    current_tier: int = 1,
    config: Optional["ManifoldConfigV2"] = None,
) -> bool:
    """Determine if object should be demoted to lower tier.

    Args:
        score: Current promotion score.
        current_tier: Current embedding tier.
        config: Optional config for threshold.

    Returns:
        True if object should be demoted.
    """
    if current_tier <= 0:
        # Already at min tier
        return False
    return score < get_demotion_threshold(config)


def compute_importance(
    retrieval_count: int,
    citation_count: int,
    recency_days: float,
    source_authority: float = 0.5,
    weights: Optional["ScoringWeights"] = None,
) -> float:
    """Compute importance factor from usage metrics.

    Args:
        retrieval_count: Number of times retrieved in queries.
        citation_count: Number of times cited in responses.
        recency_days: Days since last access.
        source_authority: Authority score of source document.
        weights: Optional scoring weights for halflife.

    Returns:
        Importance score in [0, 1].
    """
    if weights is None:
        from ..config import get_scoring_weights
        weights = get_scoring_weights()

    # Logarithmic scaling for counts
    retrieval_score = min(1.0, math.log1p(retrieval_count) / 5.0)
    citation_score = min(1.0, math.log1p(citation_count) / 4.0)

    # Recency decay using configurable halflife
    halflife = weights.importance_halflife_days
    decay_constant = halflife / math.log(2)
    recency_score = math.exp(-recency_days / decay_constant)

    # Combine with weights
    importance = (
        0.3 * retrieval_score
        + 0.3 * citation_score
        + 0.2 * recency_score
        + 0.2 * _clamp(source_authority)
    )

    return _clamp(importance)


def compute_retrieval_frequency(
    retrieval_count: int,
    time_window_days: float = 30.0,
    max_retrievals: int = 100,
) -> float:
    """Compute retrieval frequency factor.

    Args:
        retrieval_count: Number of retrievals in time window.
        time_window_days: Time window for counting.
        max_retrievals: Maximum expected retrievals for normalization.

    Returns:
        Retrieval frequency score in [0, 1].
    """
    # Normalize to daily rate
    daily_rate = retrieval_count / max(1.0, time_window_days)

    # Log-scaled normalization
    frequency = min(1.0, math.log1p(daily_rate * 10) / math.log1p(max_retrievals / 3))

    return _clamp(frequency)


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
    # Degree centrality (log scaled)
    degree_score = min(1.0, math.log1p(in_degree + out_degree) / 4.0)

    # Betweenness is already normalized if provided
    betweenness_score = _clamp(betweenness)

    # Combine (60% degree, 40% betweenness)
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
    diversity = min(
        1.0,
        math.log1p(unique_sources) / math.log1p(min(10, total_sources))
    )

    return _clamp(diversity)


def compute_novelty(
    first_seen_days_ago: float,
    mention_growth_rate: float = 0.0,
    is_emerging: bool = False,
) -> float:
    """Compute novelty factor.

    Newer entities with growing mentions score higher.

    Args:
        first_seen_days_ago: Days since first observation.
        mention_growth_rate: Rate of mention growth (positive = growing).
        is_emerging: Whether this is flagged as emerging topic.

    Returns:
        Novelty score in [0, 1].
    """
    # Recency component (newer = higher)
    # Peaks at 0 days, decays to 0.2 at 60 days
    recency_component = max(0.2, 1.0 - first_seen_days_ago / 60.0)

    # Growth component
    if mention_growth_rate > 0:
        growth_component = min(0.3, mention_growth_rate / 10.0)
    else:
        growth_component = 0.0

    # Emerging flag boost
    emerging_boost = 0.2 if is_emerging else 0.0

    novelty = recency_component * 0.5 + growth_component + emerging_boost

    return _clamp(novelty)


def compute_confidence(
    extraction_confidence: float,
    corroboration_count: int,
    has_contradictions: bool = False,
) -> float:
    """Compute confidence factor.

    Higher confidence for well-corroborated, uncontested claims.

    Args:
        extraction_confidence: Confidence from extraction (0-1).
        corroboration_count: Number of corroborating sources.
        has_contradictions: Whether contradicting evidence exists.

    Returns:
        Confidence score in [0, 1].
    """
    base_confidence = _clamp(extraction_confidence)

    # Corroboration boost (saturates at 5)
    corroboration_boost = min(0.2, math.log1p(corroboration_count) / 9.0)

    # Contradiction penalty
    contradiction_penalty = 0.3 if has_contradictions else 0.0

    confidence = base_confidence + corroboration_boost - contradiction_penalty

    return _clamp(confidence)


def compute_user_relevance(
    user_retrieval_count: int,
    user_citation_count: int,
    user_feedback_score: float = 0.0,
) -> float:
    """Compute user-specific relevance factor.

    This is a placeholder for future personalization.

    Args:
        user_retrieval_count: User-specific retrieval count.
        user_citation_count: User-specific citation count.
        user_feedback_score: Explicit user feedback if available.

    Returns:
        User relevance score in [0, 1].
    """
    # Usage-based relevance
    usage_score = min(
        1.0,
        (math.log1p(user_retrieval_count) + math.log1p(user_citation_count)) / 6.0
    )

    # Blend with explicit feedback if available
    if user_feedback_score != 0.0:
        # Feedback is in [-1, 1], convert to [0, 1]
        feedback_normalized = (user_feedback_score + 1.0) / 2.0
        relevance = 0.5 * usage_score + 0.5 * feedback_normalized
    else:
        relevance = usage_score

    return _clamp(relevance)


def compute_all_factors(
    retrieval_count: int = 0,
    citation_count: int = 0,
    recency_days: float = 0.0,
    source_authority: float = 0.5,
    source_ids: list = None,
    total_sources: int = 1,
    in_degree: int = 0,
    out_degree: int = 0,
    betweenness: float = 0.0,
    extraction_confidence: float = 0.5,
    corroboration_count: int = 0,
    has_contradictions: bool = False,
    first_seen_days_ago: float = 0.0,
    mention_growth_rate: float = 0.0,
    is_emerging: bool = False,
    user_retrieval_count: int = 0,
    user_citation_count: int = 0,
    user_feedback_score: float = 0.0,
    weights: Optional["ScoringWeights"] = None,
) -> PromotionFactors:
    """Compute all promotion factors from raw metrics.

    This is a convenience function that computes all 7 factors.

    Returns:
        PromotionFactors with all values computed.
    """
    return PromotionFactors(
        importance=compute_importance(
            retrieval_count, citation_count, recency_days, source_authority, weights
        ),
        retrieval_frequency=compute_retrieval_frequency(
            retrieval_count, time_window_days=30.0
        ),
        source_diversity=compute_source_diversity(
            source_ids or [], total_sources
        ),
        confidence=compute_confidence(
            extraction_confidence, corroboration_count, has_contradictions
        ),
        novelty=compute_novelty(
            first_seen_days_ago, mention_growth_rate, is_emerging
        ),
        graph_centrality=compute_graph_centrality(
            in_degree, out_degree, betweenness
        ),
        user_relevance=compute_user_relevance(
            user_retrieval_count, user_citation_count, user_feedback_score
        ),
    )
