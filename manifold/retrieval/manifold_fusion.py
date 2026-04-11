"""Manifold fusion — combines scores from multiple semantic manifolds.

Implements the anchor score fusion formula:

    S_anchor(q, O) = Σ_m α_m(q) · s'_m(q, O)
                   + β_lex · s_lex(q, O)
                   + β_alias · s_alias(q, O)
                   + β_cache · s_cache(q, O)
                   + β_prior · s_prior(O)
                   + β_type · s_type(O, q)
                   - β_dup · p_dup(O)
                   - β_noise · p_noise(O)

Where:
- α_m(q) is the query-conditioned manifold weight
- s'_m(q, O) is the percentile-normalized similarity in manifold m
- β_* are secondary signal weights
"""
import logging
from typing import Optional, List, Dict, Any, Tuple

from ..models.schemas import (
    ManifoldType,
    ManifoldWeights,
    ManifoldScore,
    AnchorCandidate,
    QueryModeV2,
)

logger = logging.getLogger("manifold.retrieval.fusion")


# Secondary signal weights (β coefficients)
BETA_WEIGHTS = {
    "lexical": 0.10,
    "alias_match": 0.08,
    "cache_hit": 0.05,
    "prior_importance": 0.05,
    "type_fit": 0.04,
    "duplicate_penalty": 0.06,
    "noise_penalty": 0.04,
}


class ManifoldFusion:
    """Fuses scores from multiple manifolds into a single anchor score.

    The fusion process:
    1. Collect similarity scores from each manifold
    2. Normalize scores within each manifold (percentile normalization)
    3. Apply query-conditioned weights (α)
    4. Add secondary signals (lexical, alias, cache, etc.)
    5. Apply penalties (duplicates, noise)
    """

    def __init__(
        self,
        beta_weights: Optional[Dict[str, float]] = None,
    ):
        """Initialize the fusion module.

        Args:
            beta_weights: Custom secondary signal weights.
        """
        self.beta_weights = beta_weights or BETA_WEIGHTS.copy()

    def fuse_scores(
        self,
        manifold_scores: ManifoldScore,
        alpha_weights: ManifoldWeights,
        secondary_signals: Optional[Dict[str, float]] = None,
    ) -> float:
        """Compute fused anchor score.

        Args:
            manifold_scores: Similarity scores per manifold.
            alpha_weights: Query-conditioned manifold weights.
            secondary_signals: Optional secondary signal scores.

        Returns:
            Fused anchor score (0-1 range typically).
        """
        secondary_signals = secondary_signals or {}

        # Compute weighted manifold score
        manifold_component = (
            alpha_weights.topic * manifold_scores.topic
            + alpha_weights.claim * manifold_scores.claim
            + alpha_weights.procedure * manifold_scores.procedure
            + alpha_weights.relation * manifold_scores.relation
            + alpha_weights.time * manifold_scores.time
            + alpha_weights.evidence * manifold_scores.evidence
        )

        # Add secondary positive signals
        lexical = secondary_signals.get("lexical", 0.0)
        alias_match = secondary_signals.get("alias_match", 0.0)
        cache_hit = secondary_signals.get("cache_hit", 0.0)
        prior_importance = secondary_signals.get("prior_importance", 0.0)
        type_fit = secondary_signals.get("type_fit", 0.0)

        secondary_positive = (
            self.beta_weights["lexical"] * lexical
            + self.beta_weights["alias_match"] * alias_match
            + self.beta_weights["cache_hit"] * cache_hit
            + self.beta_weights["prior_importance"] * prior_importance
            + self.beta_weights["type_fit"] * type_fit
        )

        # Subtract penalties
        duplicate_penalty = secondary_signals.get("duplicate_penalty", 0.0)
        noise_penalty = secondary_signals.get("noise_penalty", 0.0)

        penalties = (
            self.beta_weights["duplicate_penalty"] * duplicate_penalty
            + self.beta_weights["noise_penalty"] * noise_penalty
        )

        # Final score
        final_score = manifold_component + secondary_positive - penalties

        return max(0.0, final_score)

    def fuse_candidates(
        self,
        candidates: List[Dict[str, Any]],
        alpha_weights: ManifoldWeights,
    ) -> List[AnchorCandidate]:
        """Fuse scores for multiple candidates.

        Args:
            candidates: List of candidate dicts with manifold_scores and signals.
            alpha_weights: Query-conditioned manifold weights.

        Returns:
            List of AnchorCandidate with computed final scores.
        """
        results = []

        for candidate in candidates:
            manifold_scores = ManifoldScore(
                topic=candidate.get("topic_score", 0.0),
                claim=candidate.get("claim_score", 0.0),
                procedure=candidate.get("procedure_score", 0.0),
                relation=candidate.get("relation_score", 0.0),
                time=candidate.get("time_score", 0.0),
                evidence=candidate.get("evidence_score", 0.0),
            )

            secondary_signals = {
                "lexical": candidate.get("lexical_score", 0.0),
                "alias_match": candidate.get("alias_match", 0.0),
                "cache_hit": candidate.get("cache_hit", 0.0),
                "prior_importance": candidate.get("prior_importance", 0.0),
                "type_fit": candidate.get("type_fit", 0.0),
                "duplicate_penalty": candidate.get("duplicate_penalty", 0.0),
                "noise_penalty": candidate.get("noise_penalty", 0.0),
            }

            fused_score = self.fuse_scores(
                manifold_scores, alpha_weights, secondary_signals
            )

            results.append(AnchorCandidate(
                item_id=candidate.get("item_id", ""),
                item_type=candidate.get("item_type", "segment"),
                text=candidate.get("text", ""),
                manifold_scores=manifold_scores,
                fused_score=fused_score,
                lexical_score=secondary_signals["lexical"],
                alias_match=secondary_signals["alias_match"],
                cache_hit=secondary_signals["cache_hit"],
                prior_importance=secondary_signals["prior_importance"],
                duplicate_penalty=secondary_signals["duplicate_penalty"],
                noise_penalty=secondary_signals["noise_penalty"],
                final_anchor_score=fused_score,
                metadata=candidate.get("metadata", {}),
            ))

        # Sort by final score descending
        results.sort(key=lambda x: x.final_anchor_score, reverse=True)

        return results


def percentile_normalize(
    scores: List[float],
) -> List[float]:
    """Normalize scores to percentile ranks.

    Args:
        scores: Raw similarity scores.

    Returns:
        Percentile-normalized scores (0-1).
    """
    if not scores:
        return []

    n = len(scores)
    if n == 1:
        return [0.5]

    # Sort scores with original indices
    indexed = sorted(enumerate(scores), key=lambda x: x[1])

    # Assign percentile ranks
    normalized = [0.0] * n
    for rank, (orig_idx, _) in enumerate(indexed):
        normalized[orig_idx] = rank / (n - 1)

    return normalized


def compute_anchor_score(
    query_embeddings: Dict[ManifoldType, List[float]],
    candidate_embeddings: Dict[ManifoldType, List[float]],
    alpha_weights: ManifoldWeights,
    secondary_signals: Optional[Dict[str, float]] = None,
) -> Tuple[float, ManifoldScore]:
    """Compute anchor score for a single candidate.

    Args:
        query_embeddings: Query embeddings per manifold.
        candidate_embeddings: Candidate embeddings per manifold.
        alpha_weights: Manifold weights.
        secondary_signals: Secondary signal scores.

    Returns:
        Tuple of (final_score, manifold_scores).
    """
    from ..embeddings.manifold_embedder import ManifoldEmbedder

    embedder = ManifoldEmbedder()

    # Compute similarity in each manifold
    manifold_scores = ManifoldScore()

    for manifold in ManifoldType:
        q_emb = query_embeddings.get(manifold)
        c_emb = candidate_embeddings.get(manifold)

        if q_emb and c_emb:
            sim = embedder.compute_similarity(q_emb, c_emb)
            setattr(manifold_scores, manifold.value, sim)

    # Fuse scores
    fusion = ManifoldFusion()
    final_score = fusion.fuse_scores(
        manifold_scores, alpha_weights, secondary_signals
    )

    return final_score, manifold_scores


def compute_type_fit(
    item_type: str,
    query_mode: QueryModeV2,
) -> float:
    """Compute how well an item type fits the query mode.

    Args:
        item_type: Type of the candidate item.
        query_mode: The query mode.

    Returns:
        Type fit score (0-1).
    """
    # Type fit matrix
    type_fit_matrix = {
        QueryModeV2.FACT_LOOKUP: {
            "memory": 1.0,
            "claim": 0.9,
            "entity": 0.8,
            "segment": 0.5,
            "summary": 0.6,
        },
        QueryModeV2.SYNTHESIS: {
            "summary": 1.0,
            "segment": 0.7,
            "claim": 0.6,
            "entity": 0.5,
            "memory": 0.5,
        },
        QueryModeV2.TIMELINE: {
            "event": 1.0,
            "segment": 0.7,
            "claim": 0.6,
            "memory": 0.5,
        },
        QueryModeV2.PROCEDURE: {
            "procedure": 1.0,
            "segment": 0.6,
            "summary": 0.5,
        },
        QueryModeV2.VERIFICATION: {
            "claim": 1.0,
            "memory": 0.8,
            "segment": 0.6,
        },
        QueryModeV2.ASSISTANT_MEMORY: {
            "memory": 1.0,
            "claim": 0.5,
            "segment": 0.4,
        },
        QueryModeV2.REPORT: {
            "summary": 1.0,
            "segment": 0.8,
            "claim": 0.7,
            "entity": 0.7,
        },
    }

    mode_fits = type_fit_matrix.get(query_mode, {})
    return mode_fits.get(item_type, 0.5)


def compute_noise_penalty(
    text: str,
    item_type: str,
) -> float:
    """Compute noise penalty based on content quality signals.

    Args:
        text: The candidate text.
        item_type: Type of item.

    Returns:
        Noise penalty (0-1, higher = more noise).
    """
    penalty = 0.0

    # Very short text
    if len(text) < 20:
        penalty += 0.5
    elif len(text) < 50:
        penalty += 0.2

    # Tool calls/results are often noise in retrieval
    if item_type in ("tool_call", "tool_result"):
        penalty += 0.3

    # Common noise patterns
    noise_patterns = [
        "ok", "okay", "done", "failed", "success", "error",
        "yes", "no", "true", "false", "null", "none",
    ]
    text_lower = text.lower().strip()
    if text_lower in noise_patterns:
        penalty += 0.4

    return min(1.0, penalty)
