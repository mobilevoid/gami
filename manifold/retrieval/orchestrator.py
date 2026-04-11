"""Retrieval orchestrator — the core intelligence layer.

Coordinates all manifolds to produce ranked, cited evidence for queries.
Implements the anchor score fusion formula:

    S_anchor = Σ_m α_m · s'_m + β_lex + β_alias + β_cache - penalties

Where:
    α_m = query-conditioned manifold weight
    s'_m = percentile-normalized manifold score
    β_* = secondary signal boosts
    penalties = noise, duplicate, contradiction penalties
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from ..models.schemas import (
    QueryModeV2,
    ManifoldWeights,
    ManifoldScore,
    ALPHA_WEIGHTS,
)
from ..config import get_config
from .query_classifier_v2 import classify_query_v2, QueryClassification
from .manifold_fusion import ManifoldFusion, percentile_normalize

logger = logging.getLogger("gami.manifold.orchestrator")


@dataclass
class RetrievalCandidate:
    """A candidate result from retrieval."""
    item_id: str
    item_type: str  # segment, claim, entity, summary
    text: str
    manifold_scores: ManifoldScore
    fused_score: float = 0.0
    citations: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Result of a retrieval operation."""
    query: str
    mode: QueryModeV2
    confidence: float
    candidates: List[RetrievalCandidate]
    manifold_weights: ManifoldWeights
    latency_ms: float
    from_cache: bool = False
    shadow_comparison: Optional[Dict[str, Any]] = None


class RetrievalOrchestrator:
    """Orchestrates multi-manifold retrieval.

    This is the main entry point for memory.recall operations.
    It coordinates:
    1. Query classification and weight selection
    2. Parallel anchor retrieval across manifolds
    3. Score fusion and reranking
    4. Citation attachment
    5. Optional shadow mode comparison
    """

    def __init__(
        self,
        topic_index=None,      # Vector index for topic manifold
        claim_index=None,      # Vector index for claim manifold
        procedure_index=None,  # Vector index for procedure manifold
        graph_client=None,     # AGE graph client for relation manifold
        cache_client=None,     # Redis for caching
        shadow_mode: bool = False,
    ):
        """Initialize the orchestrator.

        In the isolated module, indexes are None. In production,
        they would be pgvector or other index clients.
        """
        self.topic_index = topic_index
        self.claim_index = claim_index
        self.procedure_index = procedure_index
        self.graph_client = graph_client
        self.cache_client = cache_client
        self.shadow_mode = shadow_mode

        self.config = get_config()
        self.fusion = ManifoldFusion()

    async def recall(
        self,
        query: str,
        top_k: int = 20,
        tenant_id: str = "shared",
        mode_override: Optional[QueryModeV2] = None,
        weight_override: Optional[ManifoldWeights] = None,
        include_citations: bool = True,
    ) -> RetrievalResult:
        """Execute a recall query.

        Args:
            query: The user's query.
            top_k: Number of results to return.
            tenant_id: Tenant for access control.
            mode_override: Force specific query mode.
            weight_override: Force specific manifold weights.
            include_citations: Whether to attach citations.

        Returns:
            RetrievalResult with ranked, cited candidates.
        """
        start_time = time.time()

        # 1. Check cache
        cache_key = self._cache_key(query, tenant_id, top_k)
        if self.cache_client:
            cached = await self._get_cached(cache_key)
            if cached:
                cached.from_cache = True
                return cached

        # 2. Classify query
        classification = classify_query_v2(
            query,
            mode_override=mode_override,
            weight_override=weight_override,
        )

        # 3. Parallel anchor retrieval
        candidates = await self._parallel_anchor_retrieval(
            query,
            classification,
            top_k * self.config.rerank_multiplier,
            tenant_id,
        )

        # 4. Score fusion and ranking
        ranked = self._fuse_and_rank(candidates, classification.manifold_weights)

        # 5. Take top_k
        top_candidates = ranked[:top_k]

        # 6. Attach citations
        if include_citations:
            await self._attach_citations(top_candidates)

        # 7. Shadow mode comparison
        shadow_comparison = None
        if self.shadow_mode and self._should_shadow():
            shadow_comparison = await self._run_shadow_comparison(
                query, top_candidates, tenant_id
            )

        latency_ms = (time.time() - start_time) * 1000

        result = RetrievalResult(
            query=query,
            mode=classification.mode,
            confidence=classification.confidence,
            candidates=top_candidates,
            manifold_weights=classification.manifold_weights,
            latency_ms=latency_ms,
            shadow_comparison=shadow_comparison,
        )

        # Cache result
        if self.cache_client:
            await self._set_cached(cache_key, result)

        return result

    async def _parallel_anchor_retrieval(
        self,
        query: str,
        classification: QueryClassification,
        fetch_k: int,
        tenant_id: str,
    ) -> List[RetrievalCandidate]:
        """Retrieve candidates from all relevant manifolds in parallel."""
        weights = classification.manifold_weights

        # Build retrieval tasks based on weights
        tasks = []

        if weights.topic > 0.05 and self.topic_index:
            tasks.append(self._retrieve_topic(query, fetch_k, tenant_id))

        if weights.claim > 0.05 and self.claim_index:
            tasks.append(self._retrieve_claims(query, fetch_k, tenant_id))

        if weights.procedure > 0.05 and self.procedure_index:
            tasks.append(self._retrieve_procedures(query, fetch_k, tenant_id))

        if weights.relation > 0.05 and self.graph_client:
            tasks.append(self._retrieve_relations(query, fetch_k, tenant_id))

        # Execute in parallel
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            results = []

        # Flatten and deduplicate
        candidates = {}
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Retrieval error: {result}")
                continue
            for candidate in result:
                if candidate.item_id not in candidates:
                    candidates[candidate.item_id] = candidate
                else:
                    # Merge scores from same item across manifolds
                    existing = candidates[candidate.item_id]
                    self._merge_scores(existing, candidate)

        return list(candidates.values())

    async def _retrieve_topic(
        self,
        query: str,
        k: int,
        tenant_id: str,
    ) -> List[RetrievalCandidate]:
        """Retrieve from topic manifold (dense vector)."""
        # Stub: would use pgvector similarity search
        return []

    async def _retrieve_claims(
        self,
        query: str,
        k: int,
        tenant_id: str,
    ) -> List[RetrievalCandidate]:
        """Retrieve from claim manifold (SPO-aware vectors)."""
        # Stub: would search canonical_claims with SPO embeddings
        return []

    async def _retrieve_procedures(
        self,
        query: str,
        k: int,
        tenant_id: str,
    ) -> List[RetrievalCandidate]:
        """Retrieve from procedure manifold."""
        # Stub: would search canonical_procedures
        return []

    async def _retrieve_relations(
        self,
        query: str,
        k: int,
        tenant_id: str,
    ) -> List[RetrievalCandidate]:
        """Retrieve via graph expansion (relation manifold)."""
        # Stub: would do AGE graph queries
        return []

    def _fuse_and_rank(
        self,
        candidates: List[RetrievalCandidate],
        weights: ManifoldWeights,
    ) -> List[RetrievalCandidate]:
        """Apply score fusion and rank candidates."""
        if not candidates:
            return []

        # Percentile normalize each manifold score across candidates
        topic_scores = [c.manifold_scores.topic for c in candidates]
        claim_scores = [c.manifold_scores.claim for c in candidates]
        procedure_scores = [c.manifold_scores.procedure for c in candidates]
        relation_scores = [c.manifold_scores.relation for c in candidates]
        time_scores = [c.manifold_scores.time for c in candidates]
        evidence_scores = [c.manifold_scores.evidence for c in candidates]

        norm_topic = percentile_normalize(topic_scores)
        norm_claim = percentile_normalize(claim_scores)
        norm_procedure = percentile_normalize(procedure_scores)
        norm_relation = percentile_normalize(relation_scores)
        norm_time = percentile_normalize(time_scores)
        norm_evidence = percentile_normalize(evidence_scores)

        # Fuse scores for each candidate
        for i, candidate in enumerate(candidates):
            normalized_scores = ManifoldScore(
                topic=norm_topic[i] if norm_topic else 0,
                claim=norm_claim[i] if norm_claim else 0,
                procedure=norm_procedure[i] if norm_procedure else 0,
                relation=norm_relation[i] if norm_relation else 0,
                time=norm_time[i] if norm_time else 0,
                evidence=norm_evidence[i] if norm_evidence else 0,
            )

            # Get secondary signals from metadata
            secondary = {
                "lexical": candidate.metadata.get("lexical_score", 0),
                "alias": candidate.metadata.get("alias_score", 0),
                "cache": candidate.metadata.get("cache_score", 0),
                "noise_penalty": candidate.metadata.get("noise_penalty", 0),
                "duplicate_penalty": candidate.metadata.get("duplicate_penalty", 0),
            }

            candidate.fused_score = self.fusion.fuse_scores(
                normalized_scores,
                weights,
                secondary_signals=secondary,
            )

        # Sort by fused score descending
        candidates.sort(key=lambda c: c.fused_score, reverse=True)

        return candidates

    def _merge_scores(
        self,
        existing: RetrievalCandidate,
        new: RetrievalCandidate,
    ):
        """Merge manifold scores from duplicate candidates."""
        # Take max of each manifold score
        existing.manifold_scores.topic = max(
            existing.manifold_scores.topic,
            new.manifold_scores.topic,
        )
        existing.manifold_scores.claim = max(
            existing.manifold_scores.claim,
            new.manifold_scores.claim,
        )
        existing.manifold_scores.procedure = max(
            existing.manifold_scores.procedure,
            new.manifold_scores.procedure,
        )
        existing.manifold_scores.relation = max(
            existing.manifold_scores.relation,
            new.manifold_scores.relation,
        )
        existing.manifold_scores.time = max(
            existing.manifold_scores.time,
            new.manifold_scores.time,
        )
        existing.manifold_scores.evidence = max(
            existing.manifold_scores.evidence,
            new.manifold_scores.evidence,
        )

    async def _attach_citations(
        self,
        candidates: List[RetrievalCandidate],
    ):
        """Attach source citations to candidates."""
        for candidate in candidates:
            # Stub: would query provenance table
            candidate.citations = [{
                "source_id": candidate.metadata.get("source_id"),
                "segment_id": candidate.item_id,
                "confidence": candidate.fused_score,
            }]

    def _should_shadow(self) -> bool:
        """Determine if this query should be shadowed."""
        import random
        return random.random() < self.config.shadow_sample_rate

    async def _run_shadow_comparison(
        self,
        query: str,
        new_results: List[RetrievalCandidate],
        tenant_id: str,
    ) -> Dict[str, Any]:
        """Run old retrieval and compare with new."""
        # Stub: would call legacy retrieval endpoint
        return {
            "enabled": True,
            "old_system": "legacy_retrieval",
            "comparison_pending": True,
        }

    def _cache_key(self, query: str, tenant_id: str, top_k: int) -> str:
        """Generate cache key for query."""
        import hashlib
        content = f"{query}:{tenant_id}:{top_k}"
        return f"recall:{hashlib.md5(content.encode()).hexdigest()}"

    async def _get_cached(self, key: str) -> Optional[RetrievalResult]:
        """Get cached result."""
        # Stub: would use Redis
        return None

    async def _set_cached(self, key: str, result: RetrievalResult):
        """Cache result."""
        # Stub: would use Redis with TTL
        pass


# Convenience function
async def recall(
    query: str,
    top_k: int = 20,
    tenant_id: str = "shared",
    **kwargs,
) -> RetrievalResult:
    """Execute a recall query using default orchestrator."""
    orchestrator = RetrievalOrchestrator()
    return await orchestrator.recall(query, top_k, tenant_id, **kwargs)
