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
