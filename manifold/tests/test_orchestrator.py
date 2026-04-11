"""Unit tests for retrieval orchestrator.

Tests the core retrieval coordination logic.
"""
import pytest
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from manifold.retrieval.orchestrator import (
    RetrievalOrchestrator,
    RetrievalCandidate,
    RetrievalResult,
)
from manifold.models.schemas import (
    QueryModeV2,
    ManifoldWeights,
    ManifoldScore,
)


class TestRetrievalOrchestrator:
    """Tests for retrieval orchestration."""

    @pytest.fixture
    def orchestrator(self):
        return RetrievalOrchestrator()

    @pytest.mark.asyncio
    async def test_recall_returns_result(self, orchestrator):
        """Recall should return a RetrievalResult."""
        result = await orchestrator.recall("What is PostgreSQL?")

        assert isinstance(result, RetrievalResult)
        assert result.query == "What is PostgreSQL?"
        assert isinstance(result.mode, QueryModeV2)
        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_recall_with_mode_override(self, orchestrator):
        """Mode override should be respected."""
        result = await orchestrator.recall(
            "What is X?",
            mode_override=QueryModeV2.TIMELINE,
        )

        assert result.mode == QueryModeV2.TIMELINE

    @pytest.mark.asyncio
    async def test_recall_respects_top_k(self, orchestrator):
        """Should not return more than top_k results."""
        result = await orchestrator.recall("Test query", top_k=5)
        assert len(result.candidates) <= 5

    @pytest.mark.asyncio
    async def test_recall_includes_manifold_weights(self, orchestrator):
        """Result should include manifold weights used."""
        result = await orchestrator.recall("How to deploy?")

        assert result.manifold_weights is not None
        weights = result.manifold_weights
        # Weights should sum to approximately 1
        total = (weights.topic + weights.claim + weights.procedure +
                weights.relation + weights.time + weights.evidence)
        assert 0.99 <= total <= 1.01


class TestScoreFusion:
    """Tests for score fusion logic."""

    def test_merge_scores_takes_max(self):
        """Merging should take maximum of each manifold."""
        orchestrator = RetrievalOrchestrator()

        existing = RetrievalCandidate(
            item_id="test",
            item_type="segment",
            text="test",
            manifold_scores=ManifoldScore(topic=0.5, claim=0.8),
        )
        new = RetrievalCandidate(
            item_id="test",
            item_type="segment",
            text="test",
            manifold_scores=ManifoldScore(topic=0.9, claim=0.3),
        )

        orchestrator._merge_scores(existing, new)

        assert existing.manifold_scores.topic == 0.9  # Max of 0.5, 0.9
        assert existing.manifold_scores.claim == 0.8  # Max of 0.8, 0.3

    def test_fuse_and_rank_orders_by_score(self):
        """Candidates should be ordered by fused score descending."""
        orchestrator = RetrievalOrchestrator()

        candidates = [
            RetrievalCandidate(
                item_id="low",
                item_type="segment",
                text="low",
                manifold_scores=ManifoldScore(topic=0.2),
            ),
            RetrievalCandidate(
                item_id="high",
                item_type="segment",
                text="high",
                manifold_scores=ManifoldScore(topic=0.9),
            ),
            RetrievalCandidate(
                item_id="mid",
                item_type="segment",
                text="mid",
                manifold_scores=ManifoldScore(topic=0.5),
            ),
        ]

        weights = ManifoldWeights(topic=1.0)
        ranked = orchestrator._fuse_and_rank(candidates, weights)

        assert ranked[0].item_id == "high"
        assert ranked[1].item_id == "mid"
        assert ranked[2].item_id == "low"

    def test_empty_candidates_returns_empty(self):
        """Empty input should return empty output."""
        orchestrator = RetrievalOrchestrator()
        weights = ManifoldWeights(topic=1.0)
        ranked = orchestrator._fuse_and_rank([], weights)
        assert ranked == []


class TestCaching:
    """Tests for caching behavior."""

    def test_cache_key_deterministic(self):
        """Same inputs should produce same cache key."""
        orchestrator = RetrievalOrchestrator()

        key1 = orchestrator._cache_key("test query", "tenant1", 20)
        key2 = orchestrator._cache_key("test query", "tenant1", 20)

        assert key1 == key2

    def test_cache_key_varies_with_inputs(self):
        """Different inputs should produce different cache keys."""
        orchestrator = RetrievalOrchestrator()

        key1 = orchestrator._cache_key("query1", "tenant1", 20)
        key2 = orchestrator._cache_key("query2", "tenant1", 20)
        key3 = orchestrator._cache_key("query1", "tenant2", 20)

        assert key1 != key2
        assert key1 != key3


class TestShadowMode:
    """Tests for shadow mode behavior."""

    def test_shadow_mode_disabled_by_default(self):
        """Shadow mode should be disabled by default."""
        orchestrator = RetrievalOrchestrator()
        assert orchestrator.shadow_mode is False

    def test_shadow_mode_can_be_enabled(self):
        """Shadow mode should be configurable."""
        orchestrator = RetrievalOrchestrator(shadow_mode=True)
        assert orchestrator.shadow_mode is True


class TestRetrievalCandidate:
    """Tests for RetrievalCandidate dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        candidate = RetrievalCandidate(
            item_id="test",
            item_type="segment",
            text="Test text",
            manifold_scores=ManifoldScore(),
        )

        assert candidate.fused_score == 0.0
        assert candidate.citations == []
        assert candidate.metadata == {}

    def test_manifold_scores_stored(self):
        """Manifold scores should be accessible."""
        scores = ManifoldScore(topic=0.8, claim=0.6)
        candidate = RetrievalCandidate(
            item_id="test",
            item_type="segment",
            text="Test",
            manifold_scores=scores,
        )

        assert candidate.manifold_scores.topic == 0.8
        assert candidate.manifold_scores.claim == 0.6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
