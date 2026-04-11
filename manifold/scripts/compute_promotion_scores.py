#!/usr/bin/env python3
"""Compute promotion scores for all objects in the database.

This script calculates the 7-factor promotion score for:
- Segments
- Claims
- Entities
- Summaries

Objects above the promotion threshold will be flagged for
specialized manifold embeddings.

Usage:
    python compute_promotion_scores.py [--batch-size 500] [--object-type segments]
"""
import argparse
import asyncio
import logging
import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import asyncpg
    from manifold.scoring.promotion import (
        compute_promotion_score,
        PromotionFactors,
        should_promote,
        compute_importance,
        compute_graph_centrality,
        compute_source_diversity,
        PROMOTION_THRESHOLD,
    )
    from manifold.config import get_config
except ImportError:
    asyncpg = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("compute_promotion")


OBJECT_TYPES = ["segments", "claims", "entities", "summaries"]


async def get_db_connection():
    """Get database connection."""
    return await asyncpg.connect(
        host=os.environ.get("GAMI_DB_HOST", "localhost"),
        port=int(os.environ.get("GAMI_DB_PORT", "5433")),
        user=os.environ.get("GAMI_DB_USER", "gami"),
        password=os.environ.get("GAMI_DB_PASSWORD", ""),
        database=os.environ.get("GAMI_DB_NAME", "gami"),
    )


async def get_total_sources(conn) -> int:
    """Get total number of sources in corpus."""
    result = await conn.fetchval("SELECT COUNT(*) FROM sources")
    return result or 1


async def get_object_metrics(
    conn,
    object_type: str,
    object_id: str,
    total_sources: int,
) -> PromotionFactors:
    """Compute promotion factors for an object."""

    # Get retrieval/citation counts from query logs
    retrieval_count = await conn.fetchval(
        """
        SELECT COUNT(*) FROM query_logs
        WHERE results @> $1::jsonb
        AND created_at > NOW() - INTERVAL '30 days'
        """,
        f'[{{"id": "{object_id}"}}]',
    ) or 0

    citation_count = await conn.fetchval(
        """
        SELECT citation_count FROM segments WHERE id = $1
        UNION ALL
        SELECT citation_count FROM claims WHERE id = $1
        UNION ALL
        SELECT citation_count FROM entities WHERE id = $1
        """,
        object_id,
    ) or 0

    # Get recency
    last_accessed = await conn.fetchval(
        """
        SELECT COALESCE(
            (SELECT MAX(created_at) FROM query_logs WHERE results @> $1::jsonb),
            (SELECT created_at FROM segments WHERE id = $1),
            NOW() - INTERVAL '90 days'
        )
        """,
        f'[{{"id": "{object_id}"}}]',
    )
    recency_days = (datetime.utcnow() - last_accessed.replace(tzinfo=None)).days if last_accessed else 90

    # Get source authority
    source_authority = await conn.fetchval(
        """
        SELECT COALESCE(s.importance_score, 0.5)
        FROM segments seg
        JOIN sources s ON seg.source_id = s.id
        WHERE seg.id = $1
        """,
        object_id,
    ) or 0.5

    # Compute importance
    importance = compute_importance(
        retrieval_count=retrieval_count,
        citation_count=citation_count,
        recency_days=recency_days,
        source_authority=source_authority,
    )

    # Get graph metrics (if entity or claim with relations)
    in_degree = await conn.fetchval(
        """
        SELECT COUNT(*) FROM relations WHERE target_id = $1
        """,
        object_id,
    ) or 0

    out_degree = await conn.fetchval(
        """
        SELECT COUNT(*) FROM relations WHERE source_id = $1
        """,
        object_id,
    ) or 0

    centrality = compute_graph_centrality(in_degree, out_degree)

    # Get source diversity
    source_ids = await conn.fetch(
        """
        SELECT DISTINCT source_id FROM segments
        WHERE id = $1
        OR id IN (SELECT segment_id FROM claims WHERE id = $1)
        OR id IN (SELECT source_segment_id FROM entities WHERE id = $1)
        """,
        object_id,
    )
    diversity = compute_source_diversity(
        [r["source_id"] for r in source_ids],
        total_sources,
    )

    # Get confidence
    confidence = await conn.fetchval(
        """
        SELECT COALESCE(
            (SELECT confidence FROM claims WHERE id = $1),
            (SELECT confidence FROM entities WHERE id = $1),
            0.5
        )
        """,
        object_id,
    ) or 0.5

    # Novelty: inverse of similarity to existing promoted objects
    # Simplified: check if similar objects exist
    novelty = 0.5  # Placeholder - would need embedding comparison

    # User relevance: based on tenant-specific retrieval patterns
    user_relevance = min(1.0, retrieval_count / 10) if retrieval_count > 0 else 0.3

    return PromotionFactors(
        importance=importance,
        retrieval_frequency=min(1.0, retrieval_count / 20),
        source_diversity=diversity,
        confidence=confidence,
        novelty=novelty,
        graph_centrality=centrality,
        user_relevance=user_relevance,
    )


async def get_objects_batch(
    conn,
    object_type: str,
    offset: int,
    limit: int,
) -> list:
    """Get batch of objects to score."""
    table = object_type  # segments, claims, entities, summaries
    rows = await conn.fetch(
        f"""
        SELECT id FROM {table}
        ORDER BY created_at
        OFFSET $1 LIMIT $2
        """,
        offset,
        limit,
    )
    return [r["id"] for r in rows]


async def upsert_promotion_score(
    conn,
    object_id: str,
    object_type: str,
    score: float,
    factors: PromotionFactors,
    should_promote_flag: bool,
):
    """Insert or update promotion score."""
    await conn.execute(
        """
        INSERT INTO promotion_scores (
            object_id, object_type, score,
            importance, retrieval_frequency, source_diversity,
            confidence, novelty, graph_centrality, user_relevance,
            should_promote, computed_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
        ON CONFLICT (object_id, object_type) DO UPDATE SET
            score = EXCLUDED.score,
            importance = EXCLUDED.importance,
            retrieval_frequency = EXCLUDED.retrieval_frequency,
            source_diversity = EXCLUDED.source_diversity,
            confidence = EXCLUDED.confidence,
            novelty = EXCLUDED.novelty,
            graph_centrality = EXCLUDED.graph_centrality,
            user_relevance = EXCLUDED.user_relevance,
            should_promote = EXCLUDED.should_promote,
            computed_at = EXCLUDED.computed_at
        """,
        object_id,
        object_type,
        score,
        factors.importance,
        factors.retrieval_frequency,
        factors.source_diversity,
        factors.confidence,
        factors.novelty,
        factors.graph_centrality,
        factors.user_relevance,
        should_promote_flag,
        datetime.utcnow(),
    )


async def process_batch(
    conn,
    object_type: str,
    object_ids: list,
    total_sources: int,
) -> tuple:
    """Process a batch of objects. Returns (success, errors, promoted)."""
    success = 0
    errors = 0
    promoted = 0

    for object_id in object_ids:
        try:
            factors = await get_object_metrics(conn, object_type, object_id, total_sources)
            score = compute_promotion_score(factors)
            promote = should_promote(score)

            await upsert_promotion_score(
                conn, object_id, object_type, score, factors, promote
            )

            success += 1
            if promote:
                promoted += 1

        except Exception as e:
            logger.error(f"Error processing {object_type}/{object_id}: {e}")
            errors += 1

    return success, errors, promoted


async def main(
    batch_size: int = 500,
    object_type: str = None,
):
    """Main scoring loop."""
    if asyncpg is None:
        logger.error("asyncpg not installed. Run: pip install asyncpg")
        return

    types_to_process = [object_type] if object_type else OBJECT_TYPES

    conn = await get_db_connection()
    total_sources = await get_total_sources(conn)

    try:
        for obj_type in types_to_process:
            logger.info(f"Computing promotion scores for {obj_type}")

            total_success = 0
            total_errors = 0
            total_promoted = 0
            offset = 0

            while True:
                object_ids = await get_objects_batch(conn, obj_type, offset, batch_size)

                if not object_ids:
                    break

                logger.info(f"Processing {obj_type} batch at offset {offset} ({len(object_ids)} objects)")

                success, errors, promoted = await process_batch(
                    conn, obj_type, object_ids, total_sources
                )

                total_success += success
                total_errors += errors
                total_promoted += promoted
                offset += batch_size

            logger.info(
                f"{obj_type} complete: {total_success} scored, "
                f"{total_promoted} above threshold ({PROMOTION_THRESHOLD}), "
                f"{total_errors} errors"
            )

    finally:
        await conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute promotion scores")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Number of objects per batch",
    )
    parser.add_argument(
        "--object-type",
        choices=OBJECT_TYPES,
        help="Process only this object type",
    )
    args = parser.parse_args()

    asyncio.run(main(batch_size=args.batch_size, object_type=args.object_type))
