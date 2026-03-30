"""Embedding worker — Celery task that embeds segments via Ollama.

Fetches segments without embeddings, embeds them in batches of 50,
and updates the segments table. Rate-limited to not overwhelm Ollama.
"""
import logging
import time
from datetime import datetime, timezone

from sqlalchemy import text

from workers.celery_app import celery_app

logger = logging.getLogger("gami.workers.embedder")

BATCH_SIZE = 50
# Pause between batches (seconds) to avoid overwhelming Ollama
BATCH_DELAY = 0.5


@celery_app.task(name="gami.embed_segments", bind=True, max_retries=3)
def embed_segments(self, tenant_id: str = None, batch_size: int = BATCH_SIZE):
    """
    Embed segments that have no embedding yet.

    Steps:
    1. Fetch segments with embedding IS NULL
    2. Embed in batches of 50
    3. UPDATE segments SET embedding = ... WHERE segment_id = ...
    4. Also update lexical_tsv if NULL
    """
    import os
    import sys

    gami_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if gami_root not in sys.path:
        sys.path.insert(0, gami_root)

    from api.llm.embeddings import embed_text_sync
    from api.services.db import get_sync_db

    db_gen = get_sync_db()
    db = next(db_gen)

    try:
        # Build query
        where_clause = "WHERE embedding IS NULL"
        params = {}
        if tenant_id:
            where_clause += " AND owner_tenant_id = :tid"
            params["tid"] = tenant_id

        # Count total
        count_result = db.execute(
            text(f"SELECT count(*) FROM segments {where_clause}"), params
        )
        total = count_result.scalar()
        logger.info("Found %d segments to embed (tenant=%s)", total, tenant_id or "all")

        if total == 0:
            return {"status": "completed", "embedded": 0, "total": 0}

        embedded = 0
        errors = 0
        offset = 0

        while offset < total:
            # Fetch batch
            batch_result = db.execute(
                text(
                    f"SELECT segment_id, text FROM segments {where_clause} "
                    f"ORDER BY created_at LIMIT :lim OFFSET :off"
                ),
                {**params, "lim": batch_size, "off": 0},
                # Always offset 0 since we're updating rows each batch
            )
            rows = batch_result.fetchall()
            if not rows:
                break

            for row in rows:
                seg_id = row[0]
                seg_text = row[1]

                try:
                    # Truncate very long text for embedding (nomic max ~8192 tokens)
                    embed_input = seg_text[:16000] if len(seg_text) > 16000 else seg_text
                    embedding = embed_text_sync(embed_input)

                    # Format as pgvector string
                    vec_str = "[" + ",".join(str(v) for v in embedding) + "]"

                    db.execute(
                        text(
                            "UPDATE segments SET embedding = CAST(:vec AS vector), "
                            "lexical_tsv = COALESCE(lexical_tsv, to_tsvector('english', text)) "
                            "WHERE segment_id = :sid"
                        ),
                        {"vec": vec_str, "sid": seg_id},
                    )
                    embedded += 1

                    if embedded % 10 == 0:
                        logger.info("Embedded %d / %d segments", embedded, total)
                        db.commit()

                except Exception as exc:
                    logger.warning(
                        "Failed to embed segment %s: %s", seg_id, exc
                    )
                    errors += 1

            db.commit()
            offset += len(rows)

            # Rate limit
            time.sleep(BATCH_DELAY)

        db.commit()
        logger.info(
            "Embedding complete: %d embedded, %d errors out of %d total",
            embedded, errors, total,
        )
        return {
            "status": "completed",
            "embedded": embedded,
            "errors": errors,
            "total": total,
        }

    except Exception as exc:
        logger.error("Embedding task failed: %s", exc, exc_info=True)
        raise self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))

    finally:
        try:
            next(db_gen)
        except StopIteration:
            pass
