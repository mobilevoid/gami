"""Cross-encoder reranking for improved retrieval precision.

Uses a lightweight cross-encoder model to rerank bi-encoder results,
typically improving precision by 25-40% on retrieval tasks.

The reranker scores each (query, candidate) pair directly, which is
more accurate than bi-encoder dot product but slower. We use it only
on the top-K candidates from bi-encoder search.
"""
import logging
import os
from typing import List, Tuple
from dataclasses import dataclass

logger = logging.getLogger("gami.search.reranker")

# Lazy-loaded model
_cross_encoder = None
_model_name = os.getenv(
    "GAMI_RERANKER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


@dataclass
class RerankResult:
    """Result from reranking."""
    item_id: str
    text: str
    original_score: float
    rerank_score: float
    final_score: float


def _get_cross_encoder():
    """Lazy-load the cross-encoder model."""
    global _cross_encoder
    if _cross_encoder is None:
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading cross-encoder model: {_model_name}")
            _cross_encoder = CrossEncoder(_model_name)
            logger.info("Cross-encoder loaded successfully")
        except ImportError:
            logger.warning("sentence-transformers not installed, reranking disabled")
            return None
        except Exception as e:
            logger.error(f"Failed to load cross-encoder: {e}")
            return None
    return _cross_encoder


class CrossEncoderReranker:
    """Rerank bi-encoder results with cross-encoder for improved precision.

    Usage:
        reranker = CrossEncoderReranker()
        candidates = [("id1", "text1", 0.8), ("id2", "text2", 0.75), ...]
        reranked = reranker.rerank(query, candidates, top_n=10)
    """

    def __init__(
        self,
        blend_ratio: float = 0.7,
        batch_size: int = 32,
    ):
        """Initialize reranker.

        Args:
            blend_ratio: Weight for rerank score vs original (0.7 = 70% rerank)
            batch_size: Batch size for cross-encoder inference
        """
        self.blend_ratio = blend_ratio
        self.batch_size = batch_size

    def is_available(self) -> bool:
        """Check if reranker model is available."""
        return _get_cross_encoder() is not None

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, str, float]],  # (id, text, original_score)
        top_n: int = 10,
    ) -> List[RerankResult]:
        """Rerank candidates using cross-encoder.

        Args:
            query: The search query
            candidates: List of (item_id, text, original_score) tuples
            top_n: Number of results to return after reranking

        Returns:
            List of RerankResult sorted by final_score descending
        """
        if not candidates:
            return []

        model = _get_cross_encoder()
        if model is None:
            # Fallback: return original order
            logger.debug("Reranker unavailable, returning original order")
            return [
                RerankResult(
                    item_id=cid,
                    text=text,
                    original_score=score,
                    rerank_score=score,
                    final_score=score,
                )
                for cid, text, score in candidates[:top_n]
            ]

        # Prepare pairs for cross-encoder
        pairs = [(query, text) for _, text, _ in candidates]

        # Score in batches
        try:
            rerank_scores = model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Fallback to original scores
            return [
                RerankResult(
                    item_id=cid,
                    text=text,
                    original_score=score,
                    rerank_score=score,
                    final_score=score,
                )
                for cid, text, score in candidates[:top_n]
            ]

        # Normalize rerank scores to 0-1 range
        min_score = min(rerank_scores)
        max_score = max(rerank_scores)
        score_range = max_score - min_score if max_score > min_score else 1.0

        # Blend scores and create results
        results = []
        for i, (cid, text, orig_score) in enumerate(candidates):
            normalized_rerank = (rerank_scores[i] - min_score) / score_range
            final = (
                self.blend_ratio * normalized_rerank +
                (1 - self.blend_ratio) * orig_score
            )
            results.append(RerankResult(
                item_id=cid,
                text=text,
                original_score=orig_score,
                rerank_score=float(rerank_scores[i]),
                final_score=final,
            ))

        # Sort by final score descending
        results.sort(key=lambda x: -x.final_score)

        logger.debug(
            f"Reranked {len(candidates)} candidates, returning top {top_n}"
        )

        return results[:top_n]

    def rerank_evidence(
        self,
        query: str,
        evidence_items: List,  # List of EvidenceItem
        top_n: int = 10,
    ) -> List:
        """Rerank EvidenceItem objects directly.

        Convenience method that works with the retrieval pipeline's
        EvidenceItem dataclass.

        Args:
            query: The search query
            evidence_items: List of EvidenceItem objects with item_id, text, score
            top_n: Number of results to return

        Returns:
            Reordered list of EvidenceItem with updated scores
        """
        if not evidence_items:
            return []

        # Extract data for reranking
        candidates = [
            (e.item_id, e.text, e.score)
            for e in evidence_items
        ]

        reranked = self.rerank(query, candidates, top_n=top_n)

        # Map back to original objects with updated scores
        score_map = {r.item_id: r.final_score for r in reranked}
        reranked_ids = [r.item_id for r in reranked]

        # Filter and reorder original items
        result = []
        for item_id in reranked_ids:
            for e in evidence_items:
                if e.item_id == item_id:
                    e.score = score_map[item_id]
                    result.append(e)
                    break

        return result


# Convenience function for simple usage
def rerank_results(
    query: str,
    candidates: List[Tuple[str, str, float]],
    top_n: int = 10,
    blend_ratio: float = 0.7,
) -> List[RerankResult]:
    """Rerank search results using cross-encoder.

    Args:
        query: Search query
        candidates: List of (id, text, score) tuples
        top_n: Number of results to return
        blend_ratio: Weight for rerank vs original score

    Returns:
        Reranked results
    """
    reranker = CrossEncoderReranker(blend_ratio=blend_ratio)
    return reranker.rerank(query, candidates, top_n=top_n)
