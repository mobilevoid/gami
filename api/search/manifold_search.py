"""Two-stage manifold search using true hyperbolic/spherical/Euclidean product space.

This module implements search in the product manifold H^32 × S^16 × E^64:
- Stage 1: Fast pre-filter using pgvector on Euclidean component (64d)
- Stage 2: Precise reranking using full manifold distance

The key insight is that we can leverage pgvector's fast ANN search for the
Euclidean component, then rerank the top-K candidates using true geodesic
distances in the hyperbolic and spherical components.
"""
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


@dataclass
class ManifoldSearchResult:
    """Result from manifold search."""
    target_id: str
    target_type: str
    text: str
    manifold_distance: float
    score: float  # 1 / (1 + distance) for ranking
    hyperbolic_distance: float
    spherical_distance: float
    euclidean_distance: float


@dataclass
class ManifoldWeights:
    """Weights for combining distances in product manifold."""
    hyperbolic: float = 0.3  # Hierarchy importance
    spherical: float = 0.2   # Category importance
    euclidean: float = 0.5   # Semantic similarity importance

    def __post_init__(self):
        # Normalize weights to sum to 1
        total = self.hyperbolic + self.spherical + self.euclidean
        if total > 0:
            self.hyperbolic /= total
            self.spherical /= total
            self.euclidean /= total


def poincare_distance(x: np.ndarray, y: np.ndarray, eps: float = 1e-5) -> float:
    """Compute geodesic distance in the Poincaré ball.

    d(x,y) = arcosh(1 + 2||x-y||² / ((1-||x||²)(1-||y||²)))

    This is the true hyperbolic distance, not cosine similarity.
    """
    diff_sq = np.sum((x - y) ** 2)
    norm_x_sq = np.sum(x ** 2)
    norm_y_sq = np.sum(y ** 2)

    # Clamp norms to stay inside the ball
    norm_x_sq = min(norm_x_sq, 1 - eps)
    norm_y_sq = min(norm_y_sq, 1 - eps)

    denom = (1 - norm_x_sq) * (1 - norm_y_sq)
    if denom < eps:
        denom = eps

    arg = 1 + 2 * diff_sq / denom
    return float(np.arccosh(max(arg, 1.0)))


def spherical_distance(x: np.ndarray, y: np.ndarray, eps: float = 1e-5) -> float:
    """Compute great circle distance on the unit sphere.

    d(x,y) = arccos(x·y / (||x|| ||y||))
    """
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)

    if norm_x < eps or norm_y < eps:
        return np.pi  # Max distance if either is zero

    cos_angle = np.dot(x, y) / (norm_x * norm_y)
    # Clamp for numerical stability
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    return float(np.arccos(cos_angle))


def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Compute standard L2 distance."""
    return float(np.linalg.norm(x - y))


def combined_manifold_distance(
    h1: np.ndarray, s1: np.ndarray, e1: np.ndarray,
    h2: np.ndarray, s2: np.ndarray, e2: np.ndarray,
    weights: ManifoldWeights,
) -> Tuple[float, float, float, float]:
    """Compute weighted distance in product manifold.

    Returns:
        Tuple of (combined_distance, hyperbolic_dist, spherical_dist, euclidean_dist)
    """
    d_h = poincare_distance(h1, h2)
    d_s = spherical_distance(s1, s2)
    d_e = euclidean_distance(e1, e2)

    combined = weights.hyperbolic * d_h + weights.spherical * d_s + weights.euclidean * d_e

    return combined, d_h, d_s, d_e


def format_vector_for_pgvector(coords: np.ndarray) -> str:
    """Format numpy array as pgvector string."""
    return "[" + ",".join(str(float(v)) for v in coords) + "]"


async def product_manifold_search(
    db: AsyncSession,
    query_hyperbolic: np.ndarray,
    query_spherical: np.ndarray,
    query_euclidean: np.ndarray,
    tenant_ids: List[str],
    weights: Optional[ManifoldWeights] = None,
    limit: int = 20,
    prefilter_k: int = 100,
) -> List[ManifoldSearchResult]:
    """
    Two-stage search in product manifold H^32 × S^16 × E^64.

    Stage 1: Use pgvector on Euclidean component for fast approximate nearest neighbor.
    Stage 2: Rerank top-K candidates using full manifold distance (including hyperbolic).

    Args:
        db: Database session
        query_hyperbolic: 32d hyperbolic coordinates (Poincaré ball)
        query_spherical: 16d spherical coordinates (unit sphere)
        query_euclidean: 64d Euclidean coordinates
        tenant_ids: List of tenant IDs to search
        weights: Weights for combining distances
        limit: Number of final results to return
        prefilter_k: Number of candidates to retrieve in stage 1

    Returns:
        List of ManifoldSearchResult sorted by manifold distance
    """
    if weights is None:
        weights = ManifoldWeights()

    # Format Euclidean vector for pgvector
    euc_vec_str = format_vector_for_pgvector(query_euclidean)

    # Stage 1: Fast Euclidean pre-filter using pgvector
    candidates = await db.execute(text("""
        SELECT
            pmc.target_id,
            pmc.target_type,
            pmc.hyperbolic_coords,
            pmc.spherical_coords,
            pmc.euclidean_coords,
            s.text,
            pmc.euclidean_coords <-> CAST(:vec AS vector) as euc_dist
        FROM product_manifold_coords pmc
        JOIN segments s ON pmc.target_id = s.segment_id
            AND pmc.target_type = 'segment'
        WHERE s.owner_tenant_id = ANY(:tids)
        ORDER BY pmc.euclidean_coords <-> CAST(:vec AS vector)
        LIMIT :k
    """), {"vec": euc_vec_str, "tids": tenant_ids, "k": prefilter_k})

    rows = candidates.fetchall()

    if not rows:
        return []

    # Stage 2: Rerank with full manifold distance
    results = []
    for row in rows:
        # Parse coordinates from database
        h_coords = np.array(row.hyperbolic_coords)
        s_coords = np.array(row.spherical_coords)
        e_coords = np.array(row.euclidean_coords)

        # Compute full manifold distance
        combined, d_h, d_s, d_e = combined_manifold_distance(
            query_hyperbolic, query_spherical, query_euclidean,
            h_coords, s_coords, e_coords,
            weights
        )

        results.append(ManifoldSearchResult(
            target_id=row.target_id,
            target_type=row.target_type,
            text=row.text,
            manifold_distance=combined,
            score=1.0 / (1.0 + combined),
            hyperbolic_distance=d_h,
            spherical_distance=d_s,
            euclidean_distance=d_e,
        ))

    # Sort by manifold distance (ascending)
    results.sort(key=lambda x: x.manifold_distance)

    return results[:limit]


async def product_manifold_search_from_text(
    db: AsyncSession,
    query: str,
    tenant_ids: List[str],
    weights: Optional[ManifoldWeights] = None,
    limit: int = 20,
    prefilter_k: int = 100,
) -> List[ManifoldSearchResult]:
    """
    Search using text query - encodes to manifold coordinates first.

    This is the convenience function for most use cases.
    """
    from api.llm.manifold_embeddings import get_manifold_encoder

    encoder = get_manifold_encoder()
    coords = encoder.encode(query)

    return await product_manifold_search(
        db=db,
        query_hyperbolic=coords.hyperbolic,
        query_spherical=coords.spherical,
        query_euclidean=coords.euclidean,
        tenant_ids=tenant_ids,
        weights=weights,
        limit=limit,
        prefilter_k=prefilter_k,
    )


async def manifold_search_with_fallback(
    db: AsyncSession,
    query: str,
    tenant_ids: List[str],
    weights: Optional[ManifoldWeights] = None,
    limit: int = 20,
) -> List[Dict]:
    """
    Search with fallback to traditional vector search if manifold coords not available.

    Returns results in a dict format compatible with existing search interfaces.
    """
    # Check if we have manifold coordinates for this tenant
    check = await db.execute(text("""
        SELECT COUNT(*) FROM product_manifold_coords pmc
        JOIN segments s ON pmc.target_id = s.segment_id AND pmc.target_type = 'segment'
        WHERE s.owner_tenant_id = ANY(:tids)
        LIMIT 1
    """), {"tids": tenant_ids})

    count = check.scalar() or 0

    if count > 0:
        # Use manifold search
        results = await product_manifold_search_from_text(
            db, query, tenant_ids, weights, limit
        )
        return [
            {
                "segment_id": r.target_id,
                "text": r.text,
                "score": r.score,
                "manifold_distance": r.manifold_distance,
                "search_type": "manifold",
            }
            for r in results
        ]
    else:
        # Fallback to traditional vector search
        logger.info("No manifold coordinates found, falling back to vector search")
        from api.search.hybrid import vector_search
        from api.llm.embeddings import embed_text_sync

        # Get embedding for query
        query_embedding = embed_text_sync(query)
        results = await vector_search(db, query_embedding, tenant_ids, limit)
        return [
            {
                "segment_id": r["segment_id"],
                "text": r["text"],
                "score": r["score"],
                "search_type": "vector_fallback",
            }
            for r in results
        ]
