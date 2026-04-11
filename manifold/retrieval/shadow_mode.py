"""Shadow mode for A/B comparison of retrieval systems.

Allows safe rollout of the new multi-manifold retrieval by:
1. Running both old and new systems on sampled queries
2. Comparing results without affecting user experience
3. Logging discrepancies for analysis
4. Computing quality metrics for validation

Shadow mode is controlled by:
- MANIFOLD_SHADOW_MODE=true/false
- MANIFOLD_SHADOW_SAMPLE_RATE=0.1 (10% of queries)
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
from enum import Enum

logger = logging.getLogger("gami.manifold.shadow")


class ComparisonResult(Enum):
    """Result of comparing old vs new retrieval."""
    MATCH = "match"           # Same results (by ID)
    PARTIAL_MATCH = "partial" # Some overlap
    DIVERGENT = "divergent"   # Completely different
    NEW_BETTER = "new_better" # New found more relevant results
    OLD_BETTER = "old_better" # Old found more relevant results


@dataclass
class ShadowComparison:
    """Comparison between old and new retrieval systems."""
    query: str
    timestamp: datetime
    tenant_id: str

    # Results
    new_ids: List[str]
    old_ids: List[str]
    overlap_ids: List[str]

    # Scores
    new_top_score: float
    old_top_score: float

    # Latencies
    new_latency_ms: float
    old_latency_ms: float

    # Analysis
    result: ComparisonResult
    overlap_ratio: float  # |intersection| / |union|
    rank_correlation: float  # Spearman correlation of shared items

    # Metadata
    new_mode: str = ""
    old_mode: str = ""
    notes: str = ""


class ShadowRunner:
    """Runs shadow comparisons between retrieval systems."""

    def __init__(
        self,
        new_retriever,  # The new multi-manifold retriever
        old_retriever,  # The legacy retriever (for comparison)
        storage=None,   # Where to store comparison results
    ):
        self.new_retriever = new_retriever
        self.old_retriever = old_retriever
        self.storage = storage

    async def compare(
        self,
        query: str,
        top_k: int,
        tenant_id: str,
        new_results: List[Any],  # Already computed new results
    ) -> ShadowComparison:
        """Run comparison against old system.

        Args:
            query: The query being compared.
            top_k: Number of results.
            tenant_id: Tenant context.
            new_results: Results from new system (already computed).

        Returns:
            ShadowComparison with analysis.
        """
        # Extract IDs from new results
        new_ids = [r.item_id for r in new_results]
        new_top_score = new_results[0].fused_score if new_results else 0.0

        # Run old retriever
        old_start = time.time()
        try:
            old_results = await self._run_old_retriever(query, top_k, tenant_id)
            old_latency_ms = (time.time() - old_start) * 1000
        except Exception as e:
            logger.warning(f"Old retriever failed: {e}")
            old_results = []
            old_latency_ms = 0

        old_ids = [r.get("id", r.get("item_id", "")) for r in old_results]
        old_top_score = old_results[0].get("score", 0) if old_results else 0.0

        # Compute overlap
        new_set = set(new_ids)
        old_set = set(old_ids)
        overlap = new_set & old_set
        union = new_set | old_set

        overlap_ratio = len(overlap) / len(union) if union else 1.0

        # Compute rank correlation for overlapping items
        rank_correlation = self._compute_rank_correlation(new_ids, old_ids, overlap)

        # Determine comparison result
        result = self._classify_comparison(
            new_ids, old_ids, overlap_ratio,
            new_top_score, old_top_score,
        )

        comparison = ShadowComparison(
            query=query,
            timestamp=datetime.utcnow(),
            tenant_id=tenant_id,
            new_ids=new_ids,
            old_ids=old_ids,
            overlap_ids=list(overlap),
            new_top_score=new_top_score,
            old_top_score=old_top_score,
            new_latency_ms=0,  # Already measured elsewhere
            old_latency_ms=old_latency_ms,
            result=result,
            overlap_ratio=overlap_ratio,
            rank_correlation=rank_correlation,
            new_mode=new_results[0].metadata.get("mode", "") if new_results else "",
        )

        # Store comparison
        if self.storage:
            await self._store_comparison(comparison)

        return comparison

    async def _run_old_retriever(
        self,
        query: str,
        top_k: int,
        tenant_id: str,
    ) -> List[Dict[str, Any]]:
        """Run the old retrieval system.

        In production, this would call the legacy API endpoint.
        """
        if self.old_retriever is None:
            return []

        try:
            # Assume old retriever has a search method
            results = await self.old_retriever.search(
                query=query,
                limit=top_k,
                tenant_id=tenant_id,
            )
            return results
        except Exception as e:
            logger.error(f"Old retriever error: {e}")
            return []

    def _compute_rank_correlation(
        self,
        new_ids: List[str],
        old_ids: List[str],
        overlap: Set[str],
    ) -> float:
        """Compute Spearman rank correlation for overlapping items."""
        if len(overlap) < 2:
            return 0.0

        # Build rank maps
        new_ranks = {id_: i for i, id_ in enumerate(new_ids)}
        old_ranks = {id_: i for i, id_ in enumerate(old_ids)}

        # Get ranks for overlapping items
        n = len(overlap)
        d_squared_sum = 0
        for item_id in overlap:
            d = new_ranks[item_id] - old_ranks[item_id]
            d_squared_sum += d * d

        # Spearman's rho
        rho = 1 - (6 * d_squared_sum) / (n * (n * n - 1))
        return rho

    def _classify_comparison(
        self,
        new_ids: List[str],
        old_ids: List[str],
        overlap_ratio: float,
        new_top_score: float,
        old_top_score: float,
    ) -> ComparisonResult:
        """Classify the comparison result."""
        if overlap_ratio >= 0.9:
            return ComparisonResult.MATCH

        if overlap_ratio >= 0.5:
            return ComparisonResult.PARTIAL_MATCH

        if overlap_ratio < 0.2:
            # Check which seems better based on top scores
            # In practice, this would use relevance judgments
            if new_top_score > old_top_score * 1.1:
                return ComparisonResult.NEW_BETTER
            elif old_top_score > new_top_score * 1.1:
                return ComparisonResult.OLD_BETTER

        return ComparisonResult.DIVERGENT

    async def _store_comparison(self, comparison: ShadowComparison):
        """Store comparison result for analysis."""
        if self.storage is None:
            return

        try:
            await self.storage.insert_comparison(comparison)
        except Exception as e:
            logger.error(f"Failed to store comparison: {e}")


@dataclass
class ShadowStats:
    """Aggregate statistics from shadow comparisons."""
    total_comparisons: int = 0
    match_count: int = 0
    partial_count: int = 0
    divergent_count: int = 0
    new_better_count: int = 0
    old_better_count: int = 0

    avg_overlap_ratio: float = 0.0
    avg_rank_correlation: float = 0.0

    avg_new_latency_ms: float = 0.0
    avg_old_latency_ms: float = 0.0

    @property
    def match_rate(self) -> float:
        if self.total_comparisons == 0:
            return 0.0
        return self.match_count / self.total_comparisons

    @property
    def improvement_rate(self) -> float:
        """Rate at which new system is better."""
        comparable = self.new_better_count + self.old_better_count
        if comparable == 0:
            return 0.5  # Neutral
        return self.new_better_count / comparable


class ShadowAnalyzer:
    """Analyzes shadow comparison results."""

    def __init__(self, storage=None):
        self.storage = storage

    async def compute_stats(
        self,
        since: Optional[datetime] = None,
        tenant_id: Optional[str] = None,
    ) -> ShadowStats:
        """Compute aggregate statistics from comparisons."""
        if self.storage is None:
            return ShadowStats()

        comparisons = await self.storage.get_comparisons(
            since=since,
            tenant_id=tenant_id,
        )

        if not comparisons:
            return ShadowStats()

        stats = ShadowStats(total_comparisons=len(comparisons))

        overlap_sum = 0.0
        rank_sum = 0.0
        new_lat_sum = 0.0
        old_lat_sum = 0.0

        for c in comparisons:
            # Count by result type
            if c.result == ComparisonResult.MATCH:
                stats.match_count += 1
            elif c.result == ComparisonResult.PARTIAL_MATCH:
                stats.partial_count += 1
            elif c.result == ComparisonResult.DIVERGENT:
                stats.divergent_count += 1
            elif c.result == ComparisonResult.NEW_BETTER:
                stats.new_better_count += 1
            elif c.result == ComparisonResult.OLD_BETTER:
                stats.old_better_count += 1

            overlap_sum += c.overlap_ratio
            rank_sum += c.rank_correlation
            new_lat_sum += c.new_latency_ms
            old_lat_sum += c.old_latency_ms

        n = len(comparisons)
        stats.avg_overlap_ratio = overlap_sum / n
        stats.avg_rank_correlation = rank_sum / n
        stats.avg_new_latency_ms = new_lat_sum / n
        stats.avg_old_latency_ms = old_lat_sum / n

        return stats

    def should_promote_new(self, stats: ShadowStats) -> Tuple[bool, str]:
        """Determine if new system should replace old.

        Returns (should_promote, reason).
        """
        # Require minimum comparisons
        if stats.total_comparisons < 100:
            return False, f"Insufficient data ({stats.total_comparisons} comparisons)"

        # Check match/partial rate (should be high)
        match_partial_rate = (stats.match_count + stats.partial_count) / stats.total_comparisons
        if match_partial_rate < 0.7:
            return False, f"Low consistency ({match_partial_rate:.1%} match/partial)"

        # Check improvement rate when divergent
        if stats.improvement_rate < 0.4:
            return False, f"New system not better ({stats.improvement_rate:.1%} improvement)"

        # Check latency (new should not be much slower)
        if stats.avg_new_latency_ms > stats.avg_old_latency_ms * 2:
            return False, f"New system too slow ({stats.avg_new_latency_ms:.0f}ms vs {stats.avg_old_latency_ms:.0f}ms)"

        return True, "All criteria met"


def Tuple(*args):
    """Helper for type hints."""
    from typing import Tuple as T
    return T[args]
