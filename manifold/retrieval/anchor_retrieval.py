"""Anchor retrieval — manifold-aware candidate retrieval.

The anchor retrieval process:
1. Classify query to determine mode and manifold weights
2. Embed query in relevant manifolds
3. Search each manifold for candidates
4. Apply lexical and alias matching
5. Check cache for hot candidates
6. Fuse scores from all sources
7. Return ranked anchor candidates for graph expansion

This module provides the interface but doesn't connect to the production
database. Connection happens during activation.
"""
import hashlib
import logging
import time
from typing import Optional, List, Dict, Any

from ..models.schemas import (
    ManifoldType,
    ManifoldWeights,
    ManifoldScore,
    AnchorCandidate,
    QueryModeV2,
    QueryClassificationV2,
    RecallRequestV2,
    RecallResponseV2,
)
from .query_classifier_v2 import classify_query_v2
from .manifold_fusion import ManifoldFusion, compute_type_fit, compute_noise_penalty
from ..embeddings.manifold_embedder import ManifoldEmbedder

logger = logging.getLogger("manifold.retrieval.anchor")


class AnchorRetriever:
    """Retrieves anchor candidates using multi-manifold search.

    The retriever coordinates:
    - Query classification
    - Multi-manifold embedding
    - Parallel search in each manifold
    - Score fusion
    - Candidate ranking
    """

    def __init__(
        self,
        embedder: Optional[ManifoldEmbedder] = None,
        fusion: Optional[ManifoldFusion] = None,
    ):
        """Initialize the anchor retriever.

        Args:
            embedder: Manifold embedder instance.
            fusion: Manifold fusion instance.
        """
        self.embedder = embedder or ManifoldEmbedder()
        self.fusion = fusion or ManifoldFusion()

    def retrieve(
        self,
        query: str,
        tenant_ids: List[str],
        mode: Optional[QueryModeV2] = None,
        manifold_override: Optional[ManifoldWeights] = None,
        limit: int = 50,
    ) -> List[AnchorCandidate]:
        """Retrieve anchor candidates for a query.

        Args:
            query: The query text.
            tenant_ids: Tenants to search.
            mode: Optional mode override.
            manifold_override: Optional weight override.
            limit: Maximum candidates to return.

        Returns:
            List of ranked AnchorCandidate.
        """
        t_start = time.monotonic()

        # 1. Classify query
        classification = classify_query_v2(query, mode, manifold_override)
        alpha_weights = classification.manifold_weights

        logger.debug(
            f"Query classified as {classification.mode.value} "
            f"with confidence {classification.confidence:.2f}"
        )

        # 2. Embed query in relevant manifolds
        query_embeddings = self._embed_query(query, classification)

        # 3. Search each manifold
        candidates = self._search_manifolds(
            query,
            query_embeddings,
            tenant_ids,
            alpha_weights,
            limit * 3,  # Over-retrieve for fusion
        )

        # 4. Add lexical and alias matches
        candidates = self._add_lexical_matches(query, tenant_ids, candidates)

        # 5. Check cache for hot candidates
        candidates = self._add_cache_hits(query, tenant_ids, candidates)

        # 6. Fuse scores
        anchor_candidates = self._fuse_and_rank(
            candidates,
            alpha_weights,
            classification.mode,
        )

        # 7. Limit results
        anchor_candidates = anchor_candidates[:limit]

        elapsed_ms = (time.monotonic() - t_start) * 1000
        logger.debug(
            f"Retrieved {len(anchor_candidates)} anchors in {elapsed_ms:.1f}ms"
        )

        return anchor_candidates

    def _embed_query(
        self,
        query: str,
        classification: QueryClassificationV2,
    ) -> Dict[ManifoldType, List[float]]:
        """Embed query into relevant manifolds.

        Args:
            query: Query text.
            classification: Query classification.

        Returns:
            Dict of manifold type to embedding.
        """
        embeddings = {}

        # Always embed in topic manifold
        embeddings[ManifoldType.TOPIC] = self.embedder.embed_for_manifold(
            query, ManifoldType.TOPIC
        )

        # Embed in claim manifold if it has significant weight
        if classification.manifold_weights.claim > 0.1:
            embeddings[ManifoldType.CLAIM] = self.embedder.embed_for_manifold(
                query, ManifoldType.TOPIC  # Use topic embedding as proxy
            )

        # Embed in procedure manifold if needed
        if classification.manifold_weights.procedure > 0.1:
            embeddings[ManifoldType.PROCEDURE] = self.embedder.embed_for_manifold(
                query, ManifoldType.TOPIC
            )

        return embeddings

    def _search_manifolds(
        self,
        query: str,
        query_embeddings: Dict[ManifoldType, List[float]],
        tenant_ids: List[str],
        alpha_weights: ManifoldWeights,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Search each manifold for candidates.

        NOTE: This is a stub in the isolated module.
        Returns empty list - actual DB search happens during activation.
        """
        # STUB: Database search not connected in isolated module
        logger.debug("Manifold search not connected to database")
        return []

    def _add_lexical_matches(
        self,
        query: str,
        tenant_ids: List[str],
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Add lexical/BM25 matches to candidates.

        NOTE: This is a stub in the isolated module.
        """
        # STUB: Lexical search not connected
        return candidates

    def _add_cache_hits(
        self,
        query: str,
        tenant_ids: List[str],
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Add cache hits to candidates.

        NOTE: This is a stub in the isolated module.
        """
        # STUB: Cache not connected
        return candidates

    def _fuse_and_rank(
        self,
        candidates: List[Dict[str, Any]],
        alpha_weights: ManifoldWeights,
        query_mode: QueryModeV2,
    ) -> List[AnchorCandidate]:
        """Fuse scores and rank candidates.

        Args:
            candidates: Raw candidates with scores.
            alpha_weights: Manifold weights.
            query_mode: Query mode for type fit.

        Returns:
            Ranked list of AnchorCandidate.
        """
        if not candidates:
            return []

        # Enrich candidates with type fit and noise penalty
        for candidate in candidates:
            candidate["type_fit"] = compute_type_fit(
                candidate.get("item_type", "segment"),
                query_mode,
            )
            candidate["noise_penalty"] = compute_noise_penalty(
                candidate.get("text", ""),
                candidate.get("item_type", "segment"),
            )

        # Fuse and rank
        return self.fusion.fuse_candidates(candidates, alpha_weights)


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------

def retrieve_anchors(
    query: str,
    tenant_ids: Optional[List[str]] = None,
    mode: Optional[QueryModeV2] = None,
    limit: int = 50,
) -> List[AnchorCandidate]:
    """Convenience function for anchor retrieval.

    Args:
        query: Query text.
        tenant_ids: Tenants to search.
        mode: Optional mode override.
        limit: Maximum results.

    Returns:
        List of AnchorCandidate.
    """
    retriever = AnchorRetriever()
    return retriever.retrieve(
        query,
        tenant_ids or ["default", "shared"],
        mode,
        limit=limit,
    )


async def recall_v2(
    request: RecallRequestV2,
) -> RecallResponseV2:
    """Manifold-aware recall (async version for MCP integration).

    Args:
        request: RecallRequestV2 with query and parameters.

    Returns:
        RecallResponseV2 with evidence and context.
    """
    t_start = time.monotonic()

    # Classify
    t_class = time.monotonic()
    classification = classify_query_v2(
        request.query,
        request.mode if request.mode != QueryModeV2.AUTO else None,
        request.manifold_override,
    )
    classification_ms = (time.monotonic() - t_class) * 1000

    # Retrieve
    t_retrieve = time.monotonic()
    retriever = AnchorRetriever()
    tenant_ids = request.tenant_ids or [request.tenant_id, "shared"]

    anchors = retriever.retrieve(
        request.query,
        tenant_ids,
        classification.mode,
        classification.manifold_weights,
        limit=50,
    )
    retrieval_ms = (time.monotonic() - t_retrieve) * 1000

    # TODO: Graph expansion
    # TODO: Context assembly with budget
    # TODO: Citation attachment

    # For now, return anchors as evidence
    evidence = anchors

    # Placeholder context
    context_text = "\n\n".join([a.text[:500] for a in evidence[:10]])
    token_count = len(context_text.split()) * 1.3  # Rough estimate

    total_ms = (time.monotonic() - t_start) * 1000

    response = RecallResponseV2(
        query=request.query,
        mode=classification.mode,
        manifold_weights=classification.manifold_weights,
        evidence=evidence,
        context_text=context_text,
        total_tokens_used=int(token_count),
        total_candidates=len(anchors),
        classification_ms=classification_ms,
        retrieval_ms=retrieval_ms,
        total_ms=total_ms,
    )

    if request.explain:
        response.explain_data = {
            "classification": {
                "mode": classification.mode.value,
                "confidence": classification.confidence,
                "reasoning": classification.reasoning,
            },
            "manifold_weights": classification.manifold_weights.to_dict(),
            "tenant_ids": tenant_ids,
        }

    return response
