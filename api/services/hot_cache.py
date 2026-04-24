"""Hot segment cache using Redis for frequently accessed segments."""
import json
import logging
from typing import Optional

import redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from api.config import settings

logger = logging.getLogger("gami.services.hot_cache")

r = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=settings.REDIS_DB)


def get_hot_segments(tenant_id: str, limit: int = 100) -> list[dict]:
    """Get pre-cached hot segments for a tenant."""
    key = f"gami:hot:{tenant_id}"
    data = r.get(key)
    if data:
        return json.loads(data)[:limit]
    return []


def cache_hot_segments(tenant_id: str, segments: list[dict], ttl: int = 3600):
    """Cache hot segments for a tenant."""
    key = f"gami:hot:{tenant_id}"
    r.setex(key, ttl, json.dumps(segments))


def cache_segment_embedding(segment_id: str, embedding: list[float], text: str, ttl: int = 7200):
    """Cache individual segment embedding + text for fast retrieval."""
    key = f"gami:seg:{segment_id}"
    r.setex(key, ttl, json.dumps({"embedding": embedding, "text": text}))


def get_cached_segment(segment_id: str) -> Optional[dict]:
    """Get a cached segment by ID."""
    key = f"gami:seg:{segment_id}"
    data = r.get(key)
    return json.loads(data) if data else None


async def warm_hot_cache(tenant_id: str, db: AsyncSession):
    """Preload top 200 most-accessed segments into Redis."""
    try:
        result = await db.execute(
            text("""
                SELECT segment_id, text, retrieval_count
                FROM segments
                WHERE owner_tenant_id = :tid
                  AND COALESCE(retrieval_count, 0) > 0
                ORDER BY retrieval_count DESC
                LIMIT 200
            """),
            {"tid": tenant_id},
        )
        rows = result.fetchall()
        segments = [
            {
                "segment_id": row[0],
                "text": row[1],
                "retrieval_count": int(row[2]),
            }
            for row in rows
        ]
        if segments:
            cache_hot_segments(tenant_id, segments, ttl=3600)
            logger.info("Warmed hot cache for tenant %s: %d segments", tenant_id, len(segments))
        return len(segments)
    except Exception as exc:
        logger.warning("Failed to warm hot cache for %s: %s", tenant_id, exc)
        return 0


def search_hot_cache(tenant_id: str, query: str, limit: int = 5) -> list[dict]:
    """Quick keyword match against cached hot segments.

    Returns hot segments whose text contains any of the query keywords.
    This is a fast pre-filter before hitting PostgreSQL.
    """
    segments = get_hot_segments(tenant_id)
    if not segments:
        return []

    # Simple keyword matching
    keywords = [w.lower() for w in query.split() if len(w) >= 3]
    if not keywords:
        return []

    scored = []
    for seg in segments:
        text_lower = seg.get("text", "").lower()
        matches = sum(1 for kw in keywords if kw in text_lower)
        if matches > 0:
            scored.append({**seg, "_hot_match_score": matches / len(keywords)})

    scored.sort(key=lambda x: x["_hot_match_score"], reverse=True)
    return scored[:limit]
