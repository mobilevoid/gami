"""Extraction worker — Celery task that runs structured extraction on segments.

Rate-limited to max 2 concurrent tasks to avoid overwhelming vLLM.
"""
import json
import logging
import time
from datetime import datetime, timezone

from celery import current_app
from sqlalchemy import text

from workers.celery_app import celery_app

logger = logging.getLogger("gami.workers.extractor")


@celery_app.task(
    name="gami.extract_from_segment",
    bind=True,
    max_retries=2,
    rate_limit="2/m",  # Max 2 per minute per worker to pace vLLM usage
    soft_time_limit=300,
    time_limit=360,
)
def extract_from_segment(self, segment_id: str, tenant_id: str = None, job_id: str = None):
    """
    Run all extractors (entities, claims, relations, events) on a single segment.

    Rate-limited to avoid overwhelming vLLM.
    """
    import os
    import sys

    gami_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if gami_root not in sys.path:
        sys.path.insert(0, gami_root)

    from api.services.db import get_sync_db
    from api.services.extraction import extract_all_from_segment

    db_gen = get_sync_db()
    db = next(db_gen)

    try:
        # Update job status
        if job_id:
            now = datetime.now(timezone.utc)
            db.execute(
                text(
                    "UPDATE jobs SET status = 'running', "
                    "started_at = :now, worker_id = :wid, "
                    "attempt_count = attempt_count + 1 "
                    "WHERE job_id = :jid"
                ),
                {
                    "now": now,
                    "wid": f"celery-{self.request.hostname or 'local'}",
                    "jid": job_id,
                },
            )
            db.commit()

        # Fetch segment
        row = db.execute(
            text(
                "SELECT segment_id, source_id, owner_tenant_id, text, token_count "
                "FROM segments WHERE segment_id = :sid"
            ),
            {"sid": segment_id},
        ).fetchone()

        if not row:
            logger.error("Segment %s not found", segment_id)
            return {"status": "failed", "error": "segment not found"}

        seg_id, source_id, seg_tenant, seg_text, token_count = row
        effective_tenant = tenant_id or seg_tenant

        # Skip very short segments (tool calls, single-word results)
        if not seg_text or len(seg_text.strip()) < 50:
            logger.info("Skipping short segment %s (%d chars)", segment_id, len(seg_text or ""))
            return {"status": "skipped", "reason": "too_short"}

        # Run extraction
        result = extract_all_from_segment(
            db=db,
            segment_id=seg_id,
            text_content=seg_text,
            source_id=source_id,
            tenant_id=effective_tenant,
        )

        # Update job to completed
        if job_id:
            now = datetime.now(timezone.utc)
            result_data = json.dumps({
                "entities": result["entities"],
                "claims": result["claims"],
                "relations": result["relations"],
                "events": result["events"],
            })
            db.execute(
                text(
                    "UPDATE jobs SET status = 'completed', "
                    "completed_at = :now, result_json = CAST(:result AS jsonb) "
                    "WHERE job_id = :jid"
                ),
                {"now": now, "result": result_data, "jid": job_id},
            )
            db.commit()

        logger.info(
            "Extracted from segment %s: %d entities, %d claims, %d relations, %d events",
            segment_id, result["entities"], result["claims"],
            result["relations"], result["events"],
        )

        return {
            "status": "completed",
            "segment_id": segment_id,
            "entities": result["entities"],
            "claims": result["claims"],
            "relations": result["relations"],
            "events": result["events"],
        }

    except Exception as exc:
        logger.error("Extraction failed for segment %s: %s", segment_id, exc, exc_info=True)

        if job_id:
            try:
                error_data = json.dumps({"error": str(exc), "type": type(exc).__name__})
                db.execute(
                    text(
                        "UPDATE jobs SET status = 'failed', "
                        "error_json = CAST(:err AS jsonb), "
                        "completed_at = :now "
                        "WHERE job_id = :jid"
                    ),
                    {"err": error_data, "now": datetime.now(timezone.utc), "jid": job_id},
                )
                db.commit()
            except Exception:
                logger.error("Failed to update job failure status", exc_info=True)

        raise self.retry(exc=exc, countdown=120 * (2 ** self.request.retries))

    finally:
        try:
            next(db_gen)
        except StopIteration:
            pass
