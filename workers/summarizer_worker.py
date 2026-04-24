"""Summarizer worker — Celery task that generates hierarchical summaries.

For documents: chunk summaries -> section summaries -> source summary
For conversations: message summaries -> topic episode summaries -> session summary
"""
import json
import logging
from datetime import datetime, timezone

from sqlalchemy import text

from workers.celery_app import celery_app

logger = logging.getLogger("gami.workers.summarizer")


@celery_app.task(
    name="gami.summarize_source",
    bind=True,
    max_retries=2,
    rate_limit="2/m",
    soft_time_limit=600,
    time_limit=660,
)
def summarize_source_task(self, source_id: str, tenant_id: str = None, job_id: str = None):
    """
    Generate hierarchical summaries for a source.
    """
    import os
    import sys

    gami_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if gami_root not in sys.path:
        sys.path.insert(0, gami_root)

    from api.services.db import get_sync_db
    from api.services.summarizer import summarize_source

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

        # Get tenant if not provided
        if not tenant_id:
            row = db.execute(
                text("SELECT owner_tenant_id FROM sources WHERE source_id = :sid"),
                {"sid": source_id},
            ).fetchone()
            if row:
                tenant_id = row[0]
            else:
                logger.error("Source %s not found", source_id)
                return {"status": "failed", "error": "source not found"}

        # Run summarization
        result = summarize_source(db, source_id, tenant_id)

        # Update job to completed
        if job_id:
            now = datetime.now(timezone.utc)
            result_data = json.dumps(result, default=str)
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
            "Summarized source %s: %d summaries generated",
            source_id, result.get("summaries_generated", 0),
        )
        return result

    except Exception as exc:
        logger.error("Summarization failed for source %s: %s", source_id, exc, exc_info=True)

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
