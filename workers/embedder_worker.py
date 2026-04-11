"""Embedding worker — Celery task that embeds segments via Ollama.

Fetches segments without embeddings, embeds them in batches of 50,
and updates the segments table. Rate-limited to not overwhelm Ollama.

Also generates multi-manifold embeddings (TOPIC, CLAIM, PROCEDURE, TIME, EVIDENCE)
for each segment to support manifold-aware retrieval.
"""
import logging
import time
from datetime import datetime, timezone

import numpy as np
from sqlalchemy import text

from workers.celery_app import celery_app

logger = logging.getLogger("gami.workers.embedder")

BATCH_SIZE = 50
# Pause between batches (seconds) to avoid overwhelming Ollama
BATCH_DELAY = 0.5
# Whether to generate manifold embeddings alongside standard embeddings
GENERATE_MANIFOLD_EMBEDDINGS = True


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

                    # Generate manifold embeddings for this segment
                    if GENERATE_MANIFOLD_EMBEDDINGS and len(seg_text) > 50:
                        _generate_manifold_embeddings(db, seg_id, seg_text, embedding)

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


def _generate_manifold_embeddings(db, segment_id: str, text: str, base_embedding: list):
    """Generate and store manifold embeddings for a segment.

    Creates embeddings for TOPIC, CLAIM, PROCEDURE, TIME, and EVIDENCE manifolds.
    RELATION manifold requires graph data and is handled separately.
    """
    if not GENERATE_MANIFOLD_EMBEDDINGS:
        return

    try:
        from manifold.embeddings.multi_embedder import MultiManifoldEmbedder
        from manifold.embeddings.projectors import (
            compute_evidence_factors,
            get_time_projector,
            get_evidence_projector,
        )
        from api.llm.embeddings import embed_text_sync

        embedder = MultiManifoldEmbedder()
        time_projector = get_time_projector()
        evidence_projector = get_evidence_projector()

        def save_manifold(manifold_type: str, embedding, canonical: str = None):
            """Save a manifold embedding to the database."""
            emb_str = "[" + ",".join(str(x) for x in np.array(embedding).flatten()) + "]"
            db.execute(text(f"""
                INSERT INTO manifold_embeddings
                (target_id, target_type, manifold_type, embedding, canonical_form, updated_at)
                VALUES (:tid, 'segment', :mtype, '{emb_str}'::vector, :canonical, NOW())
                ON CONFLICT (target_id, target_type, manifold_type)
                DO UPDATE SET embedding = EXCLUDED.embedding, updated_at = NOW()
            """), {"tid": segment_id, "mtype": manifold_type, "canonical": canonical})

        # TOPIC: Use the base embedding we already computed
        save_manifold("TOPIC", base_embedding, text[:500])

        # CLAIM: Extract claims and embed
        claims = embedder._extract_claims(text)
        if claims:
            claim_text = " | ".join(claims)
            claim_emb = embed_text_sync(claim_text[:8000])
            save_manifold("CLAIM", claim_emb, claim_text[:500])
        else:
            # Use base embedding if no claims extracted
            save_manifold("CLAIM", base_embedding, text[:500])

        # PROCEDURE: Extract steps and embed
        steps = embedder._extract_procedure_steps(text)
        if steps:
            proc_text = " → ".join(steps)
            proc_emb = embed_text_sync(proc_text[:8000])
            save_manifold("PROCEDURE", proc_emb, proc_text[:500])

        # TIME: Extract temporal features and project
        time_features = embedder._extract_temporal_features(text)
        if time_features is not None:
            time_emb = time_projector.project(time_features)
            save_manifold("TIME", time_emb, None)

        # EVIDENCE: Use default evidence factors (can be updated later with real data)
        evidence_factors = compute_evidence_factors(
            authority_score=0.5,
            corroboration_count=0,
            recency_days=7.0,  # Assume recent for new segments
            specificity_score=0.5,
            contradiction_ratio=0.0,
        )
        evidence_emb = evidence_projector.project(evidence_factors)
        save_manifold("EVIDENCE", evidence_emb, None)

    except Exception as e:
        logger.warning(f"Manifold embedding failed for {segment_id}: {e}")


@celery_app.task(name="gami.embed_manifolds", bind=True, max_retries=2)
def embed_manifolds_batch(self, segment_ids: list, batch_size: int = 20):
    """Generate manifold embeddings for a batch of segments that already have base embeddings.

    This is a separate task that can be called to backfill manifold embeddings
    for segments that were embedded before manifold support was added.
    """
    import os
    import sys

    gami_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if gami_root not in sys.path:
        sys.path.insert(0, gami_root)

    from api.services.db import get_sync_db

    db_gen = get_sync_db()
    db = next(db_gen)

    try:
        processed = 0
        errors = 0

        for seg_id in segment_ids:
            try:
                # Get segment text and embedding
                result = db.execute(
                    text("SELECT text, embedding::text FROM segments WHERE segment_id = :sid"),
                    {"sid": seg_id}
                )
                row = result.fetchone()
                if not row or not row[0] or not row[1]:
                    continue

                seg_text = row[0]
                # Parse embedding from pgvector string
                emb_str = row[1].strip("[]")
                base_embedding = [float(x) for x in emb_str.split(",")]

                _generate_manifold_embeddings(db, seg_id, seg_text, base_embedding)
                processed += 1

                if processed % 10 == 0:
                    db.commit()
                    logger.info(f"Manifold embeddings: {processed}/{len(segment_ids)}")

            except Exception as e:
                logger.warning(f"Failed manifold embedding for {seg_id}: {e}")
                errors += 1

        db.commit()
        logger.info(f"Manifold batch complete: {processed} processed, {errors} errors")
        return {"processed": processed, "errors": errors}

    except Exception as exc:
        logger.error("Manifold embedding task failed: %s", exc, exc_info=True)
        raise self.retry(exc=exc, countdown=60)

    finally:
        try:
            next(db_gen)
        except StopIteration:
            pass
