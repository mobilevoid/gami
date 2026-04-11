"""Unit tests for query classifier v2.

Tests query mode detection, manifold weight computation, and
pattern matching accuracy.
"""
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from manifold.retrieval.query_classifier_v2 import (
    QueryClassifierV2,
    classify_query_v2,
    get_manifold_weights,
    explain_classification,
)
from manifold.models.schemas import QueryModeV2, ManifoldWeights


class TestQueryClassification:
    """Tests for query mode classification."""

    def test_fact_lookup_password(self):
        """Password queries should be fact_lookup."""
        result = classify_query_v2("What's the root password for GitLab?")
        assert result.mode == QueryModeV2.FACT_LOOKUP
        assert result.confidence >= 0.7

    def test_fact_lookup_ip(self):
        """IP address queries should be fact_lookup."""
        result = classify_query_v2("What is the IP address of the database server?")
        assert result.mode == QueryModeV2.FACT_LOOKUP

    def test_timeline_when(self):
        """'When' queries should be timeline."""
        result = classify_query_v2("When did the edge failover happen?")
        assert result.mode == QueryModeV2.TIMELINE
        assert result.manifold_weights.time >= 0.3

    def test_timeline_date(self):
        """Date-specific queries should be timeline."""
        result = classify_query_v2("What happened on March 28?")
        assert result.mode == QueryModeV2.TIMELINE

    def test_procedure_howto(self):
        """'How to' queries should be procedure."""
        result = classify_query_v2("How do I deploy the application to production?")
        assert result.mode == QueryModeV2.PROCEDURE
        assert result.manifold_weights.procedure >= 0.4

    def test_procedure_steps(self):
        """'Steps to' queries should be procedure."""
        result = classify_query_v2("What are the steps to configure SSL?")
        assert result.mode == QueryModeV2.PROCEDURE

    def test_verification_true(self):
        """'Is it true' queries should be verification."""
        result = classify_query_v2("Is it true that PostgreSQL runs on port 5433?")
        assert result.mode == QueryModeV2.VERIFICATION
        assert result.manifold_weights.evidence >= 0.4

    def test_verification_confirm(self):
        """'Confirm' queries should be verification."""
        result = classify_query_v2("Can you confirm the backup runs at 1 AM?")
        assert result.mode == QueryModeV2.VERIFICATION

    def test_comparison(self):
        """Comparison queries should be comparison mode."""
        result = classify_query_v2("What's the difference between AGE and Neo4j?")
        assert result.mode == QueryModeV2.COMPARISON
        assert result.manifold_weights.relation >= 0.2

    def test_synthesis_summarize(self):
        """'Summarize' queries should be synthesis."""
        result = classify_query_v2("Summarize the architecture of the system")
        assert result.mode == QueryModeV2.SYNTHESIS

    def test_report_comprehensive(self):
        """'Comprehensive' queries should be report."""
        result = classify_query_v2("Give me a comprehensive list of all services")
        assert result.mode == QueryModeV2.REPORT

    def test_memory_recall(self):
        """Memory reference queries should be assistant_memory."""
        result = classify_query_v2("Do you remember what we discussed yesterday?")
        assert result.mode == QueryModeV2.ASSISTANT_MEMORY

    def test_memory_previous(self):
        """'Previous conversation' queries should be assistant_memory."""
        result = classify_query_v2("In our previous conversation, you mentioned...")
        assert result.mode == QueryModeV2.ASSISTANT_MEMORY


class TestManifoldWeights:
    """Tests for manifold weight computation."""

    def test_weights_sum_to_one(self):
        """Weights should sum to approximately 1.0."""
        result = classify_query_v2("What is PostgreSQL?")
        weights = result.manifold_weights
        total = (weights.topic + weights.claim + weights.procedure +
                weights.relation + weights.time + weights.evidence)
        assert 0.99 <= total <= 1.01

    def test_timeline_emphasizes_time(self):
        """Timeline queries should weight time manifold highly."""
        result = classify_query_v2("When did the migration happen?")
        assert result.manifold_weights.time >= 0.35

    def test_procedure_emphasizes_procedure(self):
        """Procedure queries should weight procedure manifold highly."""
        result = classify_query_v2("How to install Redis?")
        assert result.manifold_weights.procedure >= 0.4

    def test_verification_emphasizes_evidence(self):
        """Verification queries should weight evidence manifold highly."""
        result = classify_query_v2("Verify that the claim is true")
        assert result.manifold_weights.evidence >= 0.45

    def test_weight_override(self):
        """Manual weight override should work."""
        custom_weights = ManifoldWeights(
            topic=0.5, claim=0.5, procedure=0.0,
            relation=0.0, time=0.0, evidence=0.0
        )
        result = classify_query_v2(
            "What is X?",
            weight_override=custom_weights
        )
        # After normalization
        assert result.manifold_weights.topic == 0.5
        assert result.manifold_weights.claim == 0.5

    def test_mode_override(self):
        """Mode override should bypass classification."""
        result = classify_query_v2(
            "What is the password?",  # Would normally be fact_lookup
            mode_override=QueryModeV2.TIMELINE
        )
        assert result.mode == QueryModeV2.TIMELINE
        assert result.confidence == 1.0


class TestExplainClassification:
    """Tests for classification explanation."""

    def test_explain_returns_dict(self):
        """Explain should return a dict with required keys."""
        result = explain_classification("How to deploy the application?")
        assert "query" in result
        assert "mode" in result
        assert "confidence" in result
        assert "reasoning" in result
        assert "manifold_weights" in result
        assert "weight_explanation" in result

    def test_explain_includes_reasoning(self):
        """Explain should include reasoning for classification."""
        result = explain_classification("When did this happen?")
        assert result["reasoning"] is not None
        assert len(result["reasoning"]) > 0


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_query(self):
        """Empty query should return default."""
        result = classify_query_v2("")
        assert result.mode == QueryModeV2.FACT_LOOKUP
        assert result.confidence < 0.6

    def test_very_short_query(self):
        """Very short queries should still classify."""
        result = classify_query_v2("password")
        assert result.mode == QueryModeV2.FACT_LOOKUP

    def test_mixed_signals(self):
        """Queries with mixed signals should pick strongest."""
        # "How to verify" has both procedure and verification signals
        result = classify_query_v2("How to verify the backup worked?")
        # Either mode is acceptable, but should have reasonable confidence
        assert result.mode in (QueryModeV2.PROCEDURE, QueryModeV2.VERIFICATION)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
