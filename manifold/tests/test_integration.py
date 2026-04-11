"""Integration tests for the multi-manifold system.

Tests the full pipeline from query to ranked results.
These tests verify that all components work together correctly.
"""
import pytest
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from manifold.retrieval.query_classifier_v2 import classify_query_v2
from manifold.retrieval.manifold_fusion import ManifoldFusion, percentile_normalize
from manifold.retrieval.orchestrator import RetrievalOrchestrator, RetrievalCandidate
from manifold.scoring.promotion import compute_promotion_score, PromotionFactors
from manifold.scoring.evidence import compute_evidence_score, EvidenceFactors
from manifold.scoring.relation import compute_graph_fingerprint, fingerprint_similarity
from manifold.temporal.feature_extractor import TemporalExtractor
from manifold.canonical.claim_normalizer import ClaimNormalizer
from manifold.config import ManifoldConfig, get_config
from manifold.models.schemas import QueryModeV2, ManifoldWeights, ManifoldScore


class TestQueryToWeightsPipeline:
    """Test query classification to weight assignment pipeline."""

    def test_fact_lookup_query_weights(self):
        """Fact lookup queries should emphasize claim and evidence."""
        result = classify_query_v2("What's the password for GitLab?")

        assert result.mode == QueryModeV2.FACT_LOOKUP
        # Claim should be weighted for fact extraction
        assert result.manifold_weights.claim >= 0.2
        # Evidence matters for verification
        assert result.manifold_weights.evidence >= 0.1

    def test_timeline_query_weights(self):
        """Timeline queries should emphasize time manifold."""
        result = classify_query_v2("When did the deployment happen?")

        assert result.mode == QueryModeV2.TIMELINE
        assert result.manifold_weights.time >= 0.3

    def test_procedure_query_weights(self):
        """Procedure queries should emphasize procedure manifold."""
        result = classify_query_v2("How do I configure SSL certificates?")

        assert result.mode == QueryModeV2.PROCEDURE
        assert result.manifold_weights.procedure >= 0.4

    def test_weights_always_normalized(self):
        """Weights should always sum to 1.0."""
        queries = [
            "What is X?",
            "When did Y happen?",
            "How to do Z?",
            "Is it true that A?",
            "Compare B and C",
        ]

        for query in queries:
            result = classify_query_v2(query)
            weights = result.manifold_weights
            total = (weights.topic + weights.claim + weights.procedure +
                    weights.relation + weights.time + weights.evidence)
            assert 0.99 <= total <= 1.01, f"Weights sum to {total} for '{query}'"


class TestFusionPipeline:
    """Test score fusion pipeline."""

    def test_fusion_with_all_manifolds(self):
        """Fusion should work with scores from all manifolds."""
        fusion = ManifoldFusion()

        scores = ManifoldScore(
            topic=0.8,
            claim=0.6,
            procedure=0.4,
            relation=0.3,
            time=0.2,
            evidence=0.7,
        )

        weights = ManifoldWeights(
            topic=0.2,
            claim=0.2,
            procedure=0.1,
            relation=0.1,
            time=0.2,
            evidence=0.2,
        )

        result = fusion.fuse_scores(scores, weights)

        assert 0.0 <= result <= 1.0
        assert result > 0.3  # Should be reasonably high with good scores

    def test_percentile_normalization_preserves_ranking(self):
        """Percentile normalization should preserve relative ordering."""
        scores = [0.1, 0.5, 0.9, 0.3, 0.7]
        normalized = percentile_normalize(scores)

        # Original ranking: index 2 > 4 > 1 > 3 > 0
        # Should be preserved in normalized
        assert normalized[2] > normalized[4]
        assert normalized[4] > normalized[1]
        assert normalized[1] > normalized[3]
        assert normalized[3] > normalized[0]


class TestPromotionPipeline:
    """Test promotion scoring pipeline."""

    def test_high_quality_object_gets_promoted(self):
        """High-quality objects should be flagged for promotion."""
        factors = PromotionFactors(
            importance=0.9,
            retrieval_frequency=0.8,
            source_diversity=0.7,
            confidence=0.95,
            novelty=0.6,
            graph_centrality=0.5,
            user_relevance=0.85,
        )

        score = compute_promotion_score(factors)
        from manifold.scoring.promotion import should_promote
        assert should_promote(score)

    def test_low_quality_object_not_promoted(self):
        """Low-quality objects should not be promoted."""
        factors = PromotionFactors(
            importance=0.1,
            retrieval_frequency=0.05,
            source_diversity=0.1,
            confidence=0.3,
            novelty=0.2,
            graph_centrality=0.05,
            user_relevance=0.1,
        )

        score = compute_promotion_score(factors)
        from manifold.scoring.promotion import should_promote
        assert not should_promote(score)


class TestEvidencePipeline:
    """Test evidence scoring pipeline."""

    def test_well_corroborated_claim_scores_high(self):
        """Claims with multiple sources should score high."""
        factors = EvidenceFactors(
            source_authority=0.8,
            corroboration_count=5,
            days_since_observed=7,
            specificity=0.9,
            contradiction_count=0,
            total_mentions=5,
        )

        score = compute_evidence_score(factors)
        assert score >= 0.6

    def test_contradicted_claim_scores_low(self):
        """Contradicted claims should score lower."""
        factors = EvidenceFactors(
            source_authority=0.8,
            corroboration_count=2,
            days_since_observed=7,
            specificity=0.9,
            contradiction_count=4,
            total_mentions=5,
        )

        score = compute_evidence_score(factors)
        assert score < 0.5


class TestTemporalPipeline:
    """Test temporal feature extraction pipeline."""

    def test_temporal_features_for_recent_event(self):
        """Recent events should have high recency score."""
        from datetime import datetime, timedelta

        extractor = TemporalExtractor()
        now = datetime.now()

        features = extractor.extract(
            "The deployment happened",
            timestamp=now - timedelta(hours=2),
        )

        assert features.relative_recency > 0.9

    def test_temporal_features_for_specific_date(self):
        """Specific dates should have high specificity."""
        extractor = TemporalExtractor()

        features = extractor.extract(
            "Meeting scheduled for 2024-03-15 at 14:30",
        )

        assert features.temporal_specificity > 0.7
        assert features.has_explicit_date == 1.0


class TestCanonicalPipeline:
    """Test canonical form extraction pipeline."""

    def test_claim_normalization(self):
        """Claims should be normalized to SPO form."""
        normalizer = ClaimNormalizer()

        result = normalizer.normalize(
            "PostgreSQL runs on port 5433"
        )

        assert result is not None
        assert result.subject  # Has a subject
        assert result.predicate  # Has a predicate
        assert result.object  # Has an object


class TestConfigPipeline:
    """Test configuration loading pipeline."""

    def test_config_validation(self):
        """Config should pass validation with defaults."""
        config = ManifoldConfig()
        errors = config.validate()
        assert len(errors) == 0

    def test_config_detects_invalid_weights(self):
        """Config should detect invalid weight sums."""
        config = ManifoldConfig()
        config.alpha_topic_default = 0.9  # Too high
        errors = config.validate()
        assert len(errors) > 0


class TestEndToEndPipeline:
    """End-to-end pipeline tests."""

    @pytest.mark.asyncio
    async def test_full_recall_pipeline(self):
        """Test complete recall from query to result."""
        orchestrator = RetrievalOrchestrator()

        result = await orchestrator.recall(
            query="What is the database password?",
            top_k=10,
            tenant_id="test",
        )

        # Should complete without error
        assert result.query == "What is the database password?"
        assert result.mode in list(QueryModeV2)
        assert result.confidence >= 0.0
        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_recall_with_weight_override(self):
        """Recall should respect weight overrides."""
        orchestrator = RetrievalOrchestrator()

        custom_weights = ManifoldWeights(
            topic=0.5,
            claim=0.5,
            procedure=0.0,
            relation=0.0,
            time=0.0,
            evidence=0.0,
        )

        result = await orchestrator.recall(
            query="Test query",
            weight_override=custom_weights,
        )

        assert result.manifold_weights.topic == 0.5
        assert result.manifold_weights.claim == 0.5


class TestGraphPipeline:
    """Test graph fingerprinting pipeline."""

    def test_fingerprint_similar_entities(self):
        """Similar entities should have similar fingerprints."""
        # Two services that both use databases
        edges = [
            {"source": "svc1", "target": "db1", "type": "uses"},
            {"source": "svc1", "target": "cache1", "type": "uses"},
            {"source": "svc2", "target": "db2", "type": "uses"},
            {"source": "svc2", "target": "cache2", "type": "uses"},
        ]
        nodes = {
            "db1": {"type": "database"},
            "db2": {"type": "database"},
            "cache1": {"type": "cache"},
            "cache2": {"type": "cache"},
        }

        fp1 = compute_graph_fingerprint("svc1", "service", edges, nodes)
        fp2 = compute_graph_fingerprint("svc2", "service", edges, nodes)

        similarity = fingerprint_similarity(fp1, fp2)
        assert similarity > 0.8  # Should be very similar


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
