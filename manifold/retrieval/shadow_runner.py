"""Shadow runner — A/B comparison of old vs new retrieval paths.

When shadow mode is enabled, both the old (v1) and new (v2) retrieval
paths run in parallel. Results are compared for quality metrics:

- Overlap count: How many results appear in both
- Rank correlation: Spearman correlation of shared result rankings
- Latency comparison: Which path is faster
- Winner determination: Based on configurable heuristics

Shadow comparisons are logged for analysis but don't affect production output.
"""
import hashlib
import logging
import time
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field

from ..models.schemas import (
    ShadowComparisonResult,
    AnchorCandidate,
    QueryModeV2,
    ManifoldWeights,
)

logger = logging.getLogger("manifold.retrieval.shadow")


@dataclass
class OldRetrievalResult:
    """Result from old (v1) retrieval path."""
    result_ids: List[str]
    scores: List[float]
    latency_ms: float
    context_text: str = ""


@dataclass
class NewRetrievalResult:
    """Result from new (v2) manifold retrieval path."""
    result_ids: List[str]
    scores: List[float]
    latency_ms: float
    context_text: str = ""
    manifold_weights: Optional[Dict[str, float]] = None


class ShadowRunner:
    """Runs parallel old/new retrieval for comparison.

    The shadow runner:
    1. Runs both retrieval paths in parallel
    2. Compares results without affecting output
    3. Logs comparison metrics
    4. Optionally stores comparisons for analysis
    """

    def __init__(
        self,
        store_comparisons: bool = True,
    ):
        """Initialize the shadow runner.

        Args:
            store_comparisons: Whether to store comparison results.
        """
        self.store_comparisons = store_comparisons
        self._comparisons: List[ShadowComparisonResult] = []

    def run_comparison(
        self,
        query: str,
        tenant_ids: List[str],
        old_result: OldRetrievalResult,
        new_result: NewRetrievalResult,
    ) -> ShadowComparisonResult:
        """Compare old and new retrieval results.

        Args:
            query: The query text.
            tenant_ids: Tenants that were searched.
            old_result: Results from old path.
            new_result: Results from new path.

        Returns:
            ShadowComparisonResult with metrics.
        """
        query_hash = hashlib.md5(query.encode()).hexdigest()[:16]

        # Compute overlap
        old_set = set(old_result.result_ids)
        new_set = set(new_result.result_ids)
        overlap = old_set & new_set
        overlap_count = len(overlap)

        # Compute rank correlation for overlapping results
        rank_correlation = self._compute_rank_correlation(
            old_result.result_ids,
            old_result.scores,
            new_result.result_ids,
            new_result.scores,
        )

        # Determine winner
        winner = self._determine_winner(
            old_result,
            new_result,
            overlap_count,
            rank_correlation,
        )

        result = ShadowComparisonResult(
            query_hash=query_hash,
            query_text=query,
            old_result_ids=old_result.result_ids,
            old_scores=old_result.scores,
            new_result_ids=new_result.result_ids,
            new_scores=new_result.scores,
            overlap_count=overlap_count,
            rank_correlation=rank_correlation,
            latency_old_ms=int(old_result.latency_ms),
            latency_new_ms=int(new_result.latency_ms),
            winner=winner,
        )

        if self.store_comparisons:
            self._comparisons.append(result)

        return result

    def _compute_rank_correlation(
        self,
        old_ids: List[str],
        old_scores: List[float],
        new_ids: List[str],
        new_scores: List[float],
    ) -> Optional[float]:
        """Compute Spearman rank correlation for overlapping results.

        Args:
            old_ids: Result IDs from old path.
            old_scores: Scores from old path.
            new_ids: Result IDs from new path.
            new_scores: Scores from new path.

        Returns:
            Spearman correlation (-1 to 1) or None if not computable.
        """
        # Build rank maps
        old_ranks = {id_: rank for rank, id_ in enumerate(old_ids)}
        new_ranks = {id_: rank for rank, id_ in enumerate(new_ids)}

        # Find common IDs
        common = set(old_ids) & set(new_ids)
        if len(common) < 3:
            return None  # Not enough overlap

        # Compute Spearman correlation
        old_rank_list = [old_ranks[id_] for id_ in common]
        new_rank_list = [new_ranks[id_] for id_ in common]

        n = len(common)

        # Compute d^2 sum
        d_squared_sum = sum(
            (old_ranks[id_] - new_ranks[id_]) ** 2
            for id_ in common
        )

        # Spearman formula
        rho = 1 - (6 * d_squared_sum) / (n * (n ** 2 - 1))

        return rho

    def _determine_winner(
        self,
        old_result: OldRetrievalResult,
        new_result: NewRetrievalResult,
        overlap_count: int,
        rank_correlation: Optional[float],
    ) -> str:
        """Determine which path "won" based on heuristics.

        Args:
            old_result: Old path results.
            new_result: New path results.
            overlap_count: Number of overlapping results.
            rank_correlation: Rank correlation if computed.

        Returns:
            Winner string: 'old', 'new', 'tie', or 'unknown'.
        """
        # If no results from one path, the other wins
        if not old_result.result_ids and new_result.result_ids:
            return "new"
        if not new_result.result_ids and old_result.result_ids:
            return "old"
        if not old_result.result_ids and not new_result.result_ids:
            return "tie"

        # High overlap with similar ranking = tie
        min_len = min(len(old_result.result_ids), len(new_result.result_ids))
        if min_len > 0:
            overlap_ratio = overlap_count / min_len
            if overlap_ratio > 0.8 and rank_correlation and rank_correlation > 0.8:
                return "tie"

        # Significant latency difference
        if new_result.latency_ms < old_result.latency_ms * 0.7:
            # New is 30%+ faster with similar results
            if overlap_ratio > 0.5:
                return "new"
        if old_result.latency_ms < new_result.latency_ms * 0.7:
            # Old is 30%+ faster with similar results
            if overlap_ratio > 0.5:
                return "old"

        # Can't determine without relevance judgments
        return "unknown"

    def get_comparison_stats(self) -> Dict[str, Any]:
        """Get aggregate statistics from stored comparisons.

        Returns:
            Dict with aggregate metrics.
        """
        if not self._comparisons:
            return {"count": 0}

        total = len(self._comparisons)

        # Winner distribution
        winners = {"old": 0, "new": 0, "tie": 0, "unknown": 0}
        for c in self._comparisons:
            winners[c.winner] = winners.get(c.winner, 0) + 1

        # Average overlap
        avg_overlap = sum(c.overlap_count for c in self._comparisons) / total

        # Average latencies
        avg_latency_old = sum(c.latency_old_ms for c in self._comparisons) / total
        avg_latency_new = sum(c.latency_new_ms for c in self._comparisons) / total

        # Average rank correlation (where computable)
        correlations = [c.rank_correlation for c in self._comparisons if c.rank_correlation]
        avg_correlation = sum(correlations) / len(correlations) if correlations else None

        return {
            "count": total,
            "winners": winners,
            "avg_overlap": avg_overlap,
            "avg_latency_old_ms": avg_latency_old,
            "avg_latency_new_ms": avg_latency_new,
            "avg_rank_correlation": avg_correlation,
            "latency_ratio": avg_latency_new / avg_latency_old if avg_latency_old > 0 else None,
        }

    def clear_comparisons(self):
        """Clear stored comparisons."""
        self._comparisons = []


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------

def run_shadow_comparison(
    query: str,
    tenant_ids: List[str],
    old_result: OldRetrievalResult,
    new_result: NewRetrievalResult,
) -> ShadowComparisonResult:
    """Run a single shadow comparison.

    Args:
        query: Query text.
        tenant_ids: Tenants searched.
        old_result: Old path results.
        new_result: New path results.

    Returns:
        ShadowComparisonResult.
    """
    runner = ShadowRunner(store_comparisons=False)
    return runner.run_comparison(query, tenant_ids, old_result, new_result)


async def shadow_recall(
    query: str,
    tenant_ids: List[str],
    mode: Optional[QueryModeV2] = None,
) -> Tuple[Any, ShadowComparisonResult]:
    """Run recall through both paths and compare.

    NOTE: This is a stub. Actual implementation will call both
    the old recall() and new recall_v2() functions.

    Args:
        query: Query text.
        tenant_ids: Tenants to search.
        mode: Optional mode override.

    Returns:
        Tuple of (old_result, comparison).
    """
    # STUB: Both paths not connected in isolated module
    logger.debug("Shadow recall not connected in isolated module")

    # Return placeholder
    old_result = OldRetrievalResult(
        result_ids=[],
        scores=[],
        latency_ms=0,
    )
    new_result = NewRetrievalResult(
        result_ids=[],
        scores=[],
        latency_ms=0,
    )

    comparison = run_shadow_comparison(query, tenant_ids, old_result, new_result)

    return old_result, comparison
