"""Hybrid vector + lexical search for GAMI.

Combines pgvector cosine similarity with PostgreSQL tsvector full-text
search, using configurable weights for result fusion.
"""
import asyncio
import logging
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


async def vector_search(
    db: AsyncSession,
    query_embedding: list[float],
    tenant_ids: list[str],
    limit: int = 20,
) -> list[dict]:
    """Search segments by vector similarity using pgvector cosine distance."""
    vec_str = "[" + ",".join(str(v) for v in query_embedding) + "]"

    result = await db.execute(
        text("""
            SELECT segment_id, source_id, owner_tenant_id,
                   segment_type, title_or_heading, text, token_count,
                   speaker_role, speaker_name, message_timestamp,
                   1 - (embedding <=> CAST(:vec AS vector)) AS similarity,
                   COALESCE(retrieval_count, 0) AS retrieval_count,
                   last_retrieved_at
            FROM segments
            WHERE owner_tenant_id = ANY(:tids)
              AND embedding IS NOT NULL
            ORDER BY embedding <=> CAST(:vec AS vector)
            LIMIT :lim
        """),
        {"vec": vec_str, "tids": tenant_ids, "lim": limit},
    )
    rows = result.fetchall()
    return [
        {
            "segment_id": r[0],
            "source_id": r[1],
            "owner_tenant_id": r[2],
            "segment_type": r[3],
            "title_or_heading": r[4],
            "text": r[5],
            "token_count": r[6],
            "speaker_role": r[7],
            "speaker_name": r[8],
            "message_timestamp": r[9].isoformat() if r[9] else None,
            "similarity": float(r[10]),
            "retrieval_count": int(r[11]),
            "last_retrieved_at": r[12].isoformat() if r[12] else None,
            "search_type": "vector",
        }
        for r in rows
    ]


async def lexical_search(
    db: AsyncSession,
    query: str,
    tenant_ids: list[str],
    limit: int = 20,
) -> list[dict]:
    """Full-text search using tsvector."""
    result = await db.execute(
        text("""
            SELECT segment_id, source_id, owner_tenant_id,
                   segment_type, title_or_heading, text, token_count,
                   speaker_role, speaker_name, message_timestamp,
                   ts_rank(lexical_tsv, plainto_tsquery(CAST(:lang AS regconfig), :query)) AS rank,
                   COALESCE(retrieval_count, 0) AS retrieval_count,
                   last_retrieved_at
            FROM segments
            WHERE lexical_tsv @@ plainto_tsquery(CAST(:lang AS regconfig), :query)
              AND owner_tenant_id = ANY(:tids)
            ORDER BY rank DESC
            LIMIT :lim
        """),
        {"lang": "english", "query": query, "tids": tenant_ids, "lim": limit},
    )
    rows = result.fetchall()
    return [
        {
            "segment_id": r[0],
            "source_id": r[1],
            "owner_tenant_id": r[2],
            "segment_type": r[3],
            "title_or_heading": r[4],
            "text": r[5],
            "token_count": r[6],
            "speaker_role": r[7],
            "speaker_name": r[8],
            "message_timestamp": r[9].isoformat() if r[9] else None,
            "rank": float(r[10]),
            "retrieval_count": int(r[11]),
            "last_retrieved_at": r[12].isoformat() if r[12] else None,
            "search_type": "lexical",
        }
        for r in rows
    ]


def _normalize_scores(results: list[dict], score_key: str) -> list[dict]:
    """Normalize scores to 0-1 range using min-max normalization."""
    if not results:
        return results
    scores = [r[score_key] for r in results]
    min_s = min(scores)
    max_s = max(scores)
    rng = max_s - min_s
    if rng == 0:
        for r in results:
            r["normalized_score"] = 1.0
    else:
        for r in results:
            r["normalized_score"] = (r[score_key] - min_s) / rng
    return results


async def hybrid_search(
    db: AsyncSession,
    query: str,
    query_embedding: list[float],
    tenant_ids: list[str],
    limit: int = 20,
    vector_weight: float = 0.7,
    lexical_weight: float = 0.3,
) -> list[dict]:
    """Combined vector + lexical search with weighted scoring.

    Runs both searches in parallel, normalizes scores, deduplicates
    by segment_id, and returns the top results by combined score.
    """
    # Run sequentially (same session can't handle concurrent queries)
    vec_results = await vector_search(db, query_embedding, tenant_ids, limit=limit * 2)
    lex_results = await lexical_search(db, query, tenant_ids, limit=limit * 2)

    # Normalize scores
    vec_results = _normalize_scores(vec_results, "similarity")
    lex_results = _normalize_scores(lex_results, "rank")

    # Merge into a dict keyed by segment_id
    merged: dict[str, dict] = {}

    for r in vec_results:
        sid = r["segment_id"]
        merged[sid] = {
            **r,
            "vector_score": r.get("normalized_score", 0.0),
            "lexical_score": 0.0,
        }

    for r in lex_results:
        sid = r["segment_id"]
        if sid in merged:
            merged[sid]["lexical_score"] = r.get("normalized_score", 0.0)
            merged[sid]["search_type"] = "hybrid"
        else:
            merged[sid] = {
                **r,
                "vector_score": 0.0,
                "lexical_score": r.get("normalized_score", 0.0),
            }

    # Compute combined score
    for sid, entry in merged.items():
        entry["combined_score"] = (
            vector_weight * entry["vector_score"]
            + lexical_weight * entry["lexical_score"]
        )
        # Clean up temp keys
        entry.pop("normalized_score", None)
        entry.pop("similarity", None)
        entry.pop("rank", None)
        entry["search_type"] = "hybrid"

    # Sort by combined score descending
    sorted_results = sorted(
        merged.values(), key=lambda x: x["combined_score"], reverse=True
    )

    return sorted_results[:limit]


# ---------------------------------------------------------------------------
# Multi-Manifold Search
# ---------------------------------------------------------------------------

async def manifold_search(
    db: AsyncSession,
    query_embedding: list[float],
    tenant_ids: list[str],
    manifold_weights: dict[str, float],
    limit: int = 20,
) -> list[dict]:
    """Search across multiple manifolds with weighted fusion.

    Args:
        db: Database session
        query_embedding: The 768d query embedding
        tenant_ids: Tenant IDs to filter by
        manifold_weights: Dict mapping manifold type to weight (e.g., {"TOPIC": 0.4, "CLAIM": 0.2})
        limit: Maximum results to return

    Returns:
        List of results with fused scores from all manifolds
    """
    vec_str = "[" + ",".join(str(v) for v in query_embedding) + "]"

    # Collect results per manifold
    all_results: dict[str, list[dict]] = {}

    for manifold_type, weight in manifold_weights.items():
        if weight < 0.05:  # Skip negligible weights
            continue

        result = await db.execute(
            text("""
                SELECT
                    me.target_id,
                    me.target_type,
                    me.canonical_form,
                    me.confidence_score,
                    1 - (me.embedding <=> CAST(:query AS vector)) AS similarity,
                    s.source_id,
                    s.owner_tenant_id,
                    s.segment_type,
                    s.title_or_heading,
                    s.text,
                    s.token_count,
                    s.speaker_role,
                    s.speaker_name,
                    s.message_timestamp,
                    COALESCE(s.retrieval_count, 0) AS retrieval_count,
                    s.last_retrieved_at
                FROM manifold_embeddings me
                JOIN segments s ON me.target_id = s.segment_id
                WHERE me.manifold_type = :mtype
                AND me.target_type = 'segment'
                AND me.embedding IS NOT NULL
                AND s.owner_tenant_id = ANY(:tids)
                ORDER BY me.embedding <=> CAST(:query AS vector)
                LIMIT :lim
            """),
            {
                "query": vec_str,
                "mtype": manifold_type,
                "tids": tenant_ids,
                "lim": limit * 2,  # Fetch more for fusion
            }
        )
        rows = result.fetchall()

        all_results[manifold_type] = [
            {
                "segment_id": r[0],
                "target_type": r[1],
                "canonical_form": r[2],
                "confidence_score": float(r[3]) if r[3] else 0.5,
                "similarity": float(r[4]),
                "source_id": r[5],
                "owner_tenant_id": r[6],
                "segment_type": r[7],
                "title_or_heading": r[8],
                "text": r[9],
                "token_count": r[10],
                "speaker_role": r[11],
                "speaker_name": r[12],
                "message_timestamp": r[13].isoformat() if r[13] else None,
                "retrieval_count": int(r[14]),
                "last_retrieved_at": r[15].isoformat() if r[15] else None,
                "manifold_type": manifold_type,
            }
            for r in rows
        ]

    # Percentile normalize within each manifold
    for manifold_type, results in all_results.items():
        if not results:
            continue
        scores = [r["similarity"] for r in results]
        percentiles = _percentile_normalize(scores)
        for r, p in zip(results, percentiles):
            r["percentile_score"] = p

    # Fuse results across manifolds
    fused: dict[str, dict] = {}

    for manifold_type, results in all_results.items():
        weight = manifold_weights.get(manifold_type, 0.0)
        if weight <= 0:
            continue

        for r in results:
            segment_id = r["segment_id"]

            if segment_id not in fused:
                fused[segment_id] = {
                    **r,
                    "manifold_scores": {},
                    "combined_score": 0.0,
                }

            fused[segment_id]["manifold_scores"][manifold_type] = {
                "raw_score": r["similarity"],
                "percentile": r.get("percentile_score", 0.5),
                "weighted": r.get("percentile_score", 0.5) * weight,
            }
            fused[segment_id]["combined_score"] += r.get("percentile_score", 0.5) * weight

    # Sort by fused score and return top results
    sorted_results = sorted(fused.values(), key=lambda x: -x["combined_score"])

    # Add search_type marker
    for r in sorted_results:
        r["search_type"] = "manifold"

    return sorted_results[:limit]


def _percentile_normalize(scores: list[float]) -> list[float]:
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


async def hybrid_manifold_search(
    db: AsyncSession,
    query: str,
    query_embedding: list[float],
    tenant_ids: list[str],
    manifold_weights: dict[str, float],
    limit: int = 20,
    vector_weight: float = 0.5,
    manifold_weight: float = 0.5,
) -> list[dict]:
    """Combined traditional hybrid search + multi-manifold search.

    This is the recommended entry point when manifold embeddings are available.
    Falls back gracefully to traditional search if manifolds are empty.

    Args:
        db: Database session
        query: Query text for lexical search
        query_embedding: 768d embedding vector
        tenant_ids: Tenant IDs to filter
        manifold_weights: Weights for each manifold type
        limit: Maximum results
        vector_weight: Weight for traditional hybrid search
        manifold_weight: Weight for manifold search results

    Returns:
        Merged and deduplicated results from both search methods
    """
    # Run both searches
    traditional_results = await hybrid_search(
        db, query, query_embedding, tenant_ids,
        limit=limit * 2,
        vector_weight=0.7,
        lexical_weight=0.3,
    )

    manifold_results = await manifold_search(
        db, query_embedding, tenant_ids, manifold_weights,
        limit=limit * 2,
    )

    # If manifold search returned nothing, fall back to traditional only
    if not manifold_results:
        logger.debug("No manifold results, using traditional search only")
        return traditional_results[:limit]

    # Normalize traditional scores
    if traditional_results:
        trad_scores = [r["combined_score"] for r in traditional_results]
        trad_percentiles = _percentile_normalize(trad_scores)
        for r, p in zip(traditional_results, trad_percentiles):
            r["trad_percentile"] = p

    # Normalize manifold scores
    if manifold_results:
        man_scores = [r["combined_score"] for r in manifold_results]
        man_percentiles = _percentile_normalize(man_scores)
        for r, p in zip(manifold_results, man_percentiles):
            r["manifold_percentile"] = p

    # Merge by segment_id
    merged: dict[str, dict] = {}

    for r in traditional_results:
        sid = r["segment_id"]
        merged[sid] = {
            **r,
            "trad_score": r.get("trad_percentile", 0.5),
            "manifold_score": 0.0,
        }

    for r in manifold_results:
        sid = r["segment_id"]
        if sid in merged:
            merged[sid]["manifold_score"] = r.get("manifold_percentile", 0.5)
            merged[sid]["manifold_scores"] = r.get("manifold_scores", {})
            merged[sid]["search_type"] = "hybrid_manifold"
        else:
            merged[sid] = {
                **r,
                "trad_score": 0.0,
                "manifold_score": r.get("manifold_percentile", 0.5),
            }

    # Compute final combined score
    for sid, entry in merged.items():
        entry["combined_score"] = (
            vector_weight * entry.get("trad_score", 0.0) +
            manifold_weight * entry.get("manifold_score", 0.0)
        )
        entry["search_type"] = "hybrid_manifold"

    # Sort by combined score
    sorted_results = sorted(merged.values(), key=lambda x: -x["combined_score"])

    return sorted_results[:limit]


async def product_manifold_hybrid_search(
    db: AsyncSession,
    query: str,
    query_embedding: list[float],
    tenant_ids: list[str],
    limit: int = 20,
    use_product_manifold: bool = True,
    manifold_weight: float = 0.4,
    shadow_mode: bool = False,
) -> list[dict]:
    """Hybrid search with TRUE product manifold (H^32 × S^16 × E^64).

    This uses the trained hyperbolic/spherical/Euclidean product space
    for semantically and hierarchically aware retrieval.

    Args:
        db: Database session
        query: Query text
        query_embedding: 768d base embedding
        tenant_ids: Tenant IDs to filter
        limit: Maximum results
        use_product_manifold: If True, try product manifold first
        manifold_weight: Weight for manifold results (0-1)
        shadow_mode: If True, run both and log comparison but return traditional

    Returns:
        List of search results with scores
    """
    from api.search.manifold_search import (
        product_manifold_search_from_text,
        ManifoldWeights,
    )

    # Traditional search (always run)
    traditional_results = await hybrid_search(
        db, query, query_embedding, tenant_ids,
        limit=limit * 2,
        vector_weight=0.7,
        lexical_weight=0.3,
    )

    if not use_product_manifold:
        return traditional_results[:limit]

    # Try product manifold search
    try:
        manifold_results = await product_manifold_search_from_text(
            db=db,
            query=query,
            tenant_ids=tenant_ids,
            weights=ManifoldWeights(
                hyperbolic=0.3,  # Hierarchy
                spherical=0.2,   # Categories
                euclidean=0.5,   # Semantic similarity
            ),
            limit=limit * 2,
            prefilter_k=100,
        )
    except Exception as e:
        logger.warning(f"Product manifold search failed: {e}, using traditional only")
        return traditional_results[:limit]

    # If no manifold results, fall back
    if not manifold_results:
        logger.debug("No product manifold results, using traditional search only")
        return traditional_results[:limit]

    # Shadow mode: log comparison, return traditional
    if shadow_mode:
        _log_shadow_comparison(query, traditional_results, manifold_results)
        return traditional_results[:limit]

    # Merge results
    merged = _merge_product_manifold_results(
        traditional_results,
        manifold_results,
        manifold_weight,
    )

    return merged[:limit]


def _merge_product_manifold_results(
    traditional: list[dict],
    manifold: list,
    manifold_weight: float,
) -> list[dict]:
    """Merge traditional and product manifold results."""
    merged: dict[str, dict] = {}

    # Add traditional results
    for i, r in enumerate(traditional):
        sid = r.get("segment_id", r.get("target_id", ""))
        merged[sid] = {
            **r,
            "trad_rank": i,
            "trad_score": r.get("combined_score", 1.0 / (i + 1)),
            "manifold_score": 0.0,
        }

    # Add/merge manifold results
    for i, r in enumerate(manifold):
        sid = r.target_id if hasattr(r, "target_id") else r.get("target_id", "")
        score = r.score if hasattr(r, "score") else r.get("score", 1.0 / (i + 1))

        if sid in merged:
            merged[sid]["manifold_score"] = score
            merged[sid]["manifold_rank"] = i
            merged[sid]["manifold_distance"] = getattr(r, "manifold_distance", None)
        else:
            text_val = r.text if hasattr(r, "text") else r.get("text", "")
            merged[sid] = {
                "segment_id": sid,
                "text": text_val,
                "trad_rank": len(traditional),
                "trad_score": 0.0,
                "manifold_score": score,
                "manifold_rank": i,
                "manifold_distance": getattr(r, "manifold_distance", None),
            }

    # Compute final score
    for entry in merged.values():
        trad = entry.get("trad_score", 0)
        mani = entry.get("manifold_score", 0)
        entry["combined_score"] = (1 - manifold_weight) * trad + manifold_weight * mani
        entry["search_type"] = "product_manifold_hybrid"

    # Sort by combined score
    return sorted(merged.values(), key=lambda x: -x["combined_score"])


def _log_shadow_comparison(query: str, traditional: list, manifold: list):
    """Log comparison between traditional and manifold search for analysis."""
    trad_ids = [r.get("segment_id", "") for r in traditional[:10]]
    mani_ids = [r.target_id if hasattr(r, "target_id") else r.get("target_id", "")
                for r in manifold[:10]]

    overlap = set(trad_ids) & set(mani_ids)
    logger.info(
        f"Shadow comparison for '{query[:50]}...': "
        f"top-10 overlap={len(overlap)}/10, "
        f"trad_first={trad_ids[0] if trad_ids else 'none'}, "
        f"manifold_first={mani_ids[0] if mani_ids else 'none'}"
    )
