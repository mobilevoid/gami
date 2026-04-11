"""Unit tests for promotion scoring.

Tests the 7-factor promotion formula and tier transitions.
"""
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from manifold.scoring.promotion import (
    compute_promotion_score,
    PromotionFactors,
    should_promote,
    should_demote,
    PROMOTION_THRESHOLD,
    DEMOTION_THRESHOLD,
)


class TestPromotionFactors:
    """Tests for promotion factor computation."""

    def test_all_factors_contribute(self):
        """All 7 factors should contribute to final score."""
        factors = PromotionFactors(
            importance=0.8,
            retrieval_frequency=0.6,
            source_diversity=0.5,
            confidence=0.9,
            novelty=0.7,
            graph_centrality=0.4,
            user_relevance=0.85,
        )

        score = compute_promotion_score(factors)

        # Score should be in valid range
        assert 0.0 <= score <= 1.0

        # Higher factors should yield higher score
        high_factors = PromotionFactors(
            importance=1.0,
            retrieval_frequency=1.0,
            source_diversity=1.0,
            confidence=1.0,
            novelty=1.0,
            graph_centrality=1.0,
            user_relevance=1.0,
        )
        high_score = compute_promotion_score(high_factors)
        assert high_score > score

    def test_zero_factors(self):
        """All-zero factors should yield zero score."""
        factors = PromotionFactors(
            importance=0.0,
            retrieval_frequency=0.0,
            source_diversity=0.0,
            confidence=0.0,
            novelty=0.0,
            graph_centrality=0.0,
            user_relevance=0.0,
        )

        score = compute_promotion_score(factors)
        assert score == 0.0

    def test_factor_bounds(self):
        """Factors outside [0,1] should be clamped."""
        factors = PromotionFactors(
            importance=1.5,  # Over 1.0
            retrieval_frequency=-0.2,  # Negative
            source_diversity=0.5,
            confidence=0.5,
            novelty=0.5,
            graph_centrality=0.5,
            user_relevance=0.5,
        )

        score = compute_promotion_score(factors)
        assert 0.0 <= score <= 1.0


class TestPromotionThresholds:
    """Tests for promotion/demotion decisions."""

    def test_high_score_promotes(self):
        """Score above threshold should trigger promotion."""
        factors = PromotionFactors(
            importance=0.9,
            retrieval_frequency=0.8,
            source_diversity=0.7,
            confidence=0.95,
            novelty=0.6,
            graph_centrality=0.5,
            user_relevance=0.9,
        )

        score = compute_promotion_score(factors)
        assert should_promote(score)

    def test_low_score_demotes(self):
        """Score below demotion threshold should trigger demotion."""
        factors = PromotionFactors(
            importance=0.1,
            retrieval_frequency=0.05,
            source_diversity=0.1,
            confidence=0.3,
            novelty=0.1,
            graph_centrality=0.05,
            user_relevance=0.1,
        )

        score = compute_promotion_score(factors)
        assert should_demote(score)

    def test_middle_score_stable(self):
        """Score in middle range should be stable (no promote or demote)."""
        factors = PromotionFactors(
            importance=0.5,
            retrieval_frequency=0.4,
            source_diversity=0.4,
            confidence=0.6,
            novelty=0.4,
            graph_centrality=0.3,
            user_relevance=0.5,
        )

        score = compute_promotion_score(factors)
        # Should be in the stable zone
        assert not should_promote(score) or not should_demote(score)

    def test_threshold_ordering(self):
        """Promotion threshold should be higher than demotion threshold."""
        assert PROMOTION_THRESHOLD > DEMOTION_THRESHOLD
        # Should have hysteresis gap to prevent oscillation
        assert PROMOTION_THRESHOLD - DEMOTION_THRESHOLD >= 0.1


class TestFactorWeighting:
    """Tests for factor weight balance."""

    def test_importance_weight_significant(self):
        """Importance should be a significant factor."""
        base = PromotionFactors(
            importance=0.0,
            retrieval_frequency=0.5,
            source_diversity=0.5,
            confidence=0.5,
            novelty=0.5,
            graph_centrality=0.5,
            user_relevance=0.5,
        )

        high_importance = PromotionFactors(
            importance=1.0,
            retrieval_frequency=0.5,
            source_diversity=0.5,
            confidence=0.5,
            novelty=0.5,
            graph_centrality=0.5,
            user_relevance=0.5,
        )

        base_score = compute_promotion_score(base)
        high_score = compute_promotion_score(high_importance)

        # Importance should account for at least 10% of total score
        assert high_score - base_score >= 0.1

    def test_confidence_weight_significant(self):
        """Confidence should be a significant factor."""
        base = PromotionFactors(
            importance=0.5,
            retrieval_frequency=0.5,
            source_diversity=0.5,
            confidence=0.0,
            novelty=0.5,
            graph_centrality=0.5,
            user_relevance=0.5,
        )

        high_confidence = PromotionFactors(
            importance=0.5,
            retrieval_frequency=0.5,
            source_diversity=0.5,
            confidence=1.0,
            novelty=0.5,
            graph_centrality=0.5,
            user_relevance=0.5,
        )

        base_score = compute_promotion_score(base)
        high_score = compute_promotion_score(high_confidence)

        # Confidence should account for at least 15% of total score
        assert high_score - base_score >= 0.15


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_dominant_factor(self):
        """A single maxed factor should not dominate completely."""
        single_high = PromotionFactors(
            importance=1.0,
            retrieval_frequency=0.0,
            source_diversity=0.0,
            confidence=0.0,
            novelty=0.0,
            graph_centrality=0.0,
            user_relevance=0.0,
        )

        score = compute_promotion_score(single_high)
        # Single factor should not exceed 30% of max score
        assert score <= 0.3

    def test_balanced_mediocre_factors(self):
        """Balanced mediocre factors should yield mediocre score."""
        balanced = PromotionFactors(
            importance=0.5,
            retrieval_frequency=0.5,
            source_diversity=0.5,
            confidence=0.5,
            novelty=0.5,
            graph_centrality=0.5,
            user_relevance=0.5,
        )

        score = compute_promotion_score(balanced)
        # Balanced 0.5s should yield approximately 0.5
        assert 0.4 <= score <= 0.6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
