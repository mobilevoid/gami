"""Unit tests for manifold score fusion.

Tests the anchor score computation, percentile normalization,
and candidate ranking.
"""
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from manifold.retrieval.manifold_fusion import (
    ManifoldFusion,
    percentile_normalize,
    compute_type_fit,
    compute_noise_penalty,
)
from manifold.models.schemas import (
    ManifoldWeights,
    ManifoldScore,
    QueryModeV2,
)


class TestManifoldFusion:
    """Tests for score fusion."""

    def test_basic_fusion(self):
        """Test basic score fusion with equal weights."""
        fusion = ManifoldFusion()

        scores = ManifoldScore(
            topic=0.8,
            claim=0.6,
            procedure=0.4,
            relation=0.2,
            time=0.1,
            evidence=0.5,
        )

        weights = ManifoldWeights(
            topic=0.2, claim=0.2, procedure=0.2,
            relation=0.2, time=0.1, evidence=0.1,
        )

        result = fusion.fuse_scores(scores, weights)

        # Should be weighted average plus secondary signals
        expected_manifold = (
            0.2 * 0.8 +  # topic
            0.2 * 0.6 +  # claim
            0.2 * 0.4 +  # procedure
            0.2 * 0.2 +  # relation
            0.1 * 0.1 +  # time
            0.1 * 0.5    # evidence
        )
        assert abs(result - expected_manifold) < 0.01

    def test_secondary_signals_add(self):
        """Test that secondary signals add to score."""
        fusion = ManifoldFusion()

        scores = ManifoldScore(topic=0.5)
        weights = ManifoldWeights(topic=1.0)

        base_result = fusion.fuse_scores(scores, weights)

        result_with_lexical = fusion.fuse_scores(
            scores, weights,
            secondary_signals={"lexical": 1.0}
        )

        assert result_with_lexical > base_result

    def test_penalties_subtract(self):
        """Test that penalties reduce score."""
        fusion = ManifoldFusion()

        scores = ManifoldScore(topic=0.5)
        weights = ManifoldWeights(topic=1.0)

        base_result = fusion.fuse_scores(scores, weights)

        result_with_penalty = fusion.fuse_scores(
            scores, weights,
            secondary_signals={"noise_penalty": 1.0}
        )

        assert result_with_penalty < base_result

    def test_score_never_negative(self):
        """Score should never go below zero."""
        fusion = ManifoldFusion()

        scores = ManifoldScore()  # All zeros
        weights = ManifoldWeights(topic=1.0)

        result = fusion.fuse_scores(
            scores, weights,
            secondary_signals={
                "duplicate_penalty": 1.0,
                "noise_penalty": 1.0,
            }
        )

        assert result >= 0.0

    def test_candidate_fusion_sorts_descending(self):
        """Candidates should be sorted by score descending."""
        fusion = ManifoldFusion()

        candidates = [
            {"item_id": "low", "item_type": "segment", "text": "x", "topic_score": 0.2},
            {"item_id": "high", "item_type": "segment", "text": "x", "topic_score": 0.9},
            {"item_id": "mid", "item_type": "segment", "text": "x", "topic_score": 0.5},
        ]

        weights = ManifoldWeights(topic=1.0)
        result = fusion.fuse_candidates(candidates, weights)

        assert result[0].item_id == "high"
        assert result[1].item_id == "mid"
        assert result[2].item_id == "low"


class TestPercentileNormalize:
    """Tests for percentile normalization."""

    def test_empty_list(self):
        """Empty list should return empty."""
        assert percentile_normalize([]) == []

    def test_single_value(self):
        """Single value should return 0.5."""
        assert percentile_normalize([5.0]) == [0.5]

    def test_two_values(self):
        """Two values should return 0 and 1."""
        result = percentile_normalize([1.0, 2.0])
        assert result[0] == 0.0  # Lower value
        assert result[1] == 1.0  # Higher value

    def test_preserves_order(self):
        """Higher scores should get higher percentiles."""
        scores = [0.1, 0.5, 0.9, 0.3]
        result = percentile_normalize(scores)

        # 0.9 is highest, should have rank 1.0
        assert result[2] == 1.0
        # 0.1 is lowest, should have rank 0.0
        assert result[0] == 0.0

    def test_handles_ties(self):
        """Should handle tied values."""
        scores = [0.5, 0.5, 0.5]
        result = percentile_normalize(scores)
        # All values assigned some rank
        assert all(0.0 <= r <= 1.0 for r in result)


class TestTypeFit:
    """Tests for type fit computation."""

    def test_memory_fits_fact_lookup(self):
        """Memory type should fit fact_lookup well."""
        fit = compute_type_fit("memory", QueryModeV2.FACT_LOOKUP)
        assert fit >= 0.9

    def test_claim_fits_verification(self):
        """Claim type should fit verification well."""
        fit = compute_type_fit("claim", QueryModeV2.VERIFICATION)
        assert fit >= 0.9

    def test_procedure_fits_procedure(self):
        """Procedure type should fit procedure mode perfectly."""
        fit = compute_type_fit("procedure", QueryModeV2.PROCEDURE)
        assert fit == 1.0

    def test_summary_fits_synthesis(self):
        """Summary type should fit synthesis well."""
        fit = compute_type_fit("summary", QueryModeV2.SYNTHESIS)
        assert fit >= 0.9

    def test_unknown_type_gets_default(self):
        """Unknown types should get default fit."""
        fit = compute_type_fit("unknown_type", QueryModeV2.FACT_LOOKUP)
        assert fit == 0.5


class TestNoisePenalty:
    """Tests for noise penalty computation."""

    def test_very_short_text_penalized(self):
        """Very short text should be penalized."""
        penalty = compute_noise_penalty("ok", "segment")
        assert penalty >= 0.5

    def test_tool_calls_penalized(self):
        """Tool calls should be penalized."""
        penalty = compute_noise_penalty("Some tool output", "tool_result")
        assert penalty >= 0.2

    def test_common_noise_patterns(self):
        """Common noise patterns should be penalized."""
        for pattern in ["OK", "Done", "success", "true", "null"]:
            penalty = compute_noise_penalty(pattern, "segment")
            assert penalty >= 0.3

    def test_normal_text_not_penalized(self):
        """Normal length text should not be heavily penalized."""
        text = "This is a normal segment with reasonable content length."
        penalty = compute_noise_penalty(text, "segment")
        assert penalty < 0.2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
