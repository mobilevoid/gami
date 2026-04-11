"""Manifold Store — persistence layer for multi-manifold embeddings.

Handles storage and retrieval of embeddings in the manifold_embeddings table.
Provides vector search capabilities per manifold type.
"""
import logging
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from api.services.db import AsyncSessionLocal

logger = logging.getLogger("gami.services.manifold_store")


class ManifoldType:
    """Manifold type constants."""
    TOPIC = "TOPIC"
    CLAIM = "CLAIM"
    PROCEDURE = "PROCEDURE"
    RELATION = "RELATION"
    TIME = "TIME"
    EVIDENCE = "EVIDENCE"

    ALL = [TOPIC, CLAIM, PROCEDURE, RELATION, TIME, EVIDENCE]


async def upsert_manifold_embedding(
    db: AsyncSession,
    target_id: str,
    target_type: str,
    manifold_type: str,
    embedding: np.ndarray,
    canonical_form: Optional[str] = None,
    confidence_score: float = 0.5,
) -> str:
    """Insert or update a manifold embedding.

    Args:
        db: Database session
        target_id: ID of the target object
        target_type: Type of target (segment, entity, claim, memory)
        manifold_type: Which manifold (TOPIC, CLAIM, PROCEDURE, etc.)
        embedding: The 768d embedding vector
        canonical_form: Text that was embedded
        confidence_score: Confidence in this embedding

    Returns:
        The target_id
    """
    embedding_str = "[" + ",".join(str(x) for x in embedding.flatten()) + "]"

    await db.execute(
        text("""
            INSERT INTO manifold_embeddings
            (target_id, target_type, manifold_type, embedding, canonical_form, confidence_score, updated_at)
            VALUES (:tid, :ttype, :mtype, :emb::vector, :canonical, :conf, NOW())
            ON CONFLICT (target_id, target_type, manifold_type)
            DO UPDATE SET
                embedding = EXCLUDED.embedding,
                canonical_form = EXCLUDED.canonical_form,
                confidence_score = EXCLUDED.confidence_score,
                updated_at = NOW()
        """),
        {
            "tid": target_id,
            "ttype": target_type,
            "mtype": manifold_type,
            "emb": embedding_str,
            "canonical": canonical_form,
            "conf": confidence_score,
        }
    )

    return target_id


async def get_manifold_embedding(
    db: AsyncSession,
    target_id: str,
    target_type: str,
    manifold_type: str,
) -> Optional[np.ndarray]:
    """Get a specific manifold embedding.

    Args:
        db: Database session
        target_id: ID of the target object
        target_type: Type of target
        manifold_type: Which manifold

    Returns:
        768d embedding or None
    """
    result = await db.execute(
        text("""
            SELECT embedding::text
            FROM manifold_embeddings
            WHERE target_id = :tid AND target_type = :ttype AND manifold_type = :mtype
        """),
        {"tid": target_id, "ttype": target_type, "mtype": manifold_type}
    )
    row = result.fetchone()

    if row and row[0]:
        return _parse_vector(row[0])
    return None


async def get_all_manifold_embeddings(
    db: AsyncSession,
    target_id: str,
    target_type: str,
) -> Dict[str, np.ndarray]:
    """Get all manifold embeddings for a target.

    Args:
        db: Database session
        target_id: ID of the target object
        target_type: Type of target

    Returns:
        Dict mapping manifold_type to embedding
    """
    result = await db.execute(
        text("""
            SELECT manifold_type, embedding::text
            FROM manifold_embeddings
            WHERE target_id = :tid AND target_type = :ttype
        """),
        {"tid": target_id, "ttype": target_type}
    )
    rows = result.fetchall()

    embeddings = {}
    for manifold_type, emb_str in rows:
        if emb_str:
            embeddings[manifold_type] = _parse_vector(emb_str)

    return embeddings


async def search_manifold(
    db: AsyncSession,
    query_embedding: np.ndarray,
    manifold_type: str,
    tenant_ids: List[str],
    target_type: str = "segment",
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """Search within a specific manifold.

    Args:
        db: Database session
        query_embedding: Query embedding vector
        manifold_type: Which manifold to search
        tenant_ids: Tenant filter
        target_type: Target type filter
        limit: Maximum results

    Returns:
        List of results with target_id, score, etc.
    """
    emb_str = "[" + ",".join(str(x) for x in query_embedding.flatten()) + "]"

    result = await db.execute(
        text("""
            SELECT
                me.target_id,
                me.target_type,
                1 - (me.embedding <=> :query::vector) as similarity,
                me.canonical_form,
                me.confidence_score
            FROM manifold_embeddings me
            WHERE me.manifold_type = :mtype
            AND me.target_type = :ttype
            AND me.embedding IS NOT NULL
            ORDER BY me.embedding <=> :query::vector
            LIMIT :lim
        """),
        {
            "query": emb_str,
            "mtype": manifold_type,
            "ttype": target_type,
            "lim": limit,
        }
    )
    rows = result.fetchall()

    return [
        {
            "target_id": row[0],
            "target_type": row[1],
            "similarity": float(row[2]),
            "canonical_form": row[3],
            "confidence_score": float(row[4]) if row[4] else 0.5,
        }
        for row in rows
    ]


async def search_all_manifolds(
    db: AsyncSession,
    query_embedding: np.ndarray,
    manifold_weights: Dict[str, float],
    tenant_ids: List[str],
    target_type: str = "segment",
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """Search across all manifolds with weighted fusion.

    Args:
        db: Database session
        query_embedding: Query embedding vector
        manifold_weights: Weights per manifold type
        tenant_ids: Tenant filter
        target_type: Target type filter
        limit: Maximum results

    Returns:
        List of results with fused scores
    """
    # Search each manifold in parallel
    all_results: Dict[str, List[Dict]] = {}

    for manifold_type, weight in manifold_weights.items():
        if weight < 0.05:  # Skip negligible weights
            continue

        results = await search_manifold(
            db=db,
            query_embedding=query_embedding,
            manifold_type=manifold_type,
            tenant_ids=tenant_ids,
            target_type=target_type,
            limit=limit * 2,  # Fetch more for fusion
        )
        all_results[manifold_type] = results

    # Fuse results
    fused = _fuse_manifold_results(all_results, manifold_weights)

    # Sort by fused score and limit
    fused_sorted = sorted(fused.values(), key=lambda x: -x["fused_score"])
    return fused_sorted[:limit]


def _fuse_manifold_results(
    results: Dict[str, List[Dict]],
    weights: Dict[str, float],
) -> Dict[str, Dict]:
    """Fuse results from multiple manifold searches.

    Uses weighted sum with percentile normalization.
    """
    fused: Dict[str, Dict] = {}

    # Collect all results by target_id
    for manifold_type, manifold_results in results.items():
        weight = weights.get(manifold_type, 0.0)
        if weight <= 0:
            continue

        # Percentile normalize scores within this manifold
        scores = [r["similarity"] for r in manifold_results]
        if not scores:
            continue

        percentiles = _percentile_normalize(scores)

        for result, percentile in zip(manifold_results, percentiles):
            target_id = result["target_id"]

            if target_id not in fused:
                fused[target_id] = {
                    "target_id": target_id,
                    "target_type": result["target_type"],
                    "manifold_scores": {},
                    "fused_score": 0.0,
                }

            fused[target_id]["manifold_scores"][manifold_type] = {
                "raw_score": result["similarity"],
                "percentile": percentile,
                "weighted": percentile * weight,
            }
            fused[target_id]["fused_score"] += percentile * weight

    return fused


def _percentile_normalize(scores: List[float]) -> List[float]:
    """Convert scores to percentile ranks (0-1)."""
    if not scores:
        return []
    if len(scores) == 1:
        return [0.5]

    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i])
    percentiles = [0.0] * len(scores)
    n = len(scores)

    for rank, idx in enumerate(sorted_indices):
        percentiles[idx] = rank / (n - 1)

    return percentiles


def _parse_vector(vec_str: str) -> np.ndarray:
    """Parse pgvector string to numpy array."""
    # Remove brackets and split
    cleaned = vec_str.strip("[]()").replace(" ", "")
    values = [float(x) for x in cleaned.split(",") if x]
    return np.array(values, dtype=np.float32)


# ---------------------------------------------------------------------------
# Batch Operations
# ---------------------------------------------------------------------------

async def count_manifold_embeddings(
    db: AsyncSession,
    manifold_type: Optional[str] = None,
) -> int:
    """Count manifold embeddings."""
    if manifold_type:
        result = await db.execute(
            text("SELECT COUNT(*) FROM manifold_embeddings WHERE manifold_type = :mtype"),
            {"mtype": manifold_type}
        )
    else:
        result = await db.execute(
            text("SELECT COUNT(*) FROM manifold_embeddings")
        )
    return result.scalar() or 0


async def get_segments_needing_manifold_embeddings(
    db: AsyncSession,
    manifold_types: List[str],
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """Get segments that don't have all required manifold embeddings.

    Args:
        db: Database session
        manifold_types: Required manifold types
        limit: Maximum segments to return

    Returns:
        List of segment dicts
    """
    # Get segments with embeddings that aren't in manifold_embeddings
    result = await db.execute(
        text("""
            SELECT s.segment_id, s.text, s.owner_tenant_id
            FROM segments s
            WHERE s.embedding IS NOT NULL
            AND s.status = 'active'
            AND s.segment_id NOT IN (
                SELECT DISTINCT target_id
                FROM manifold_embeddings
                WHERE manifold_type = 'TOPIC'
            )
            LIMIT :lim
        """),
        {"lim": limit}
    )
    rows = result.fetchall()

    return [
        {
            "segment_id": row[0],
            "text": row[1],
            "tenant_id": row[2],
        }
        for row in rows
    ]
