# Manifold embedding services
from .manifold_embedder import ManifoldEmbedder, embed_for_manifold
from .promotion import PromotionScorer, compute_promotion_score

__all__ = [
    "ManifoldEmbedder",
    "embed_for_manifold",
    "PromotionScorer",
    "compute_promotion_score",
]
