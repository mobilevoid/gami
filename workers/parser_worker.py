"""Parser worker — Celery task that parses source files and stores segments.

This runs in a sync context (Celery worker), so it uses the sync DB session
and sync segment storage.
"""
import json
import logging
from datetime import datetime, timezone

from sqlalchemy import text

from workers.celery_app import celery_app

logger = logging.getLogger("gami.workers.parser")


@celery_app.task(name="gami.parse_source", bind=True, max_retries=3)
def parse_source(
    self,
    source_id: str,
    file_path: str,
    source_type: str,
    tenant_id: str,
    job_id: str = None,
):
    """
    Parse a source file and store segments.

    Steps:
    1. Get the appropriate parser
    2. Parse the file
    3. Store segments via segment_service (sync)
    4. Update source parse_status
    5. Update job status
    """
    import os
    import sys

    # Ensure GAMI root is on path for imports
    gami_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if gami_root not in sys.path:
        sys.path.insert(0, gami_root)

    from api.services.db import get_sync_db
    from api.services.segment_service import store_segments_sync
    from parsers import get_parser, get_parser_for_file

    logger.info(
        "Parsing source %s (type=%s, tenant=%s, job=%s)",
        source_id, source_type, tenant_id, job_id,
    )

    db_gen = get_sync_db()
    db = next(db_gen)

    try:
        # Update job to running
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

        # Update source status
        db.execute(
            text("UPDATE sources SET parse_status = 'parsing' WHERE source_id = :sid"),
            {"sid": source_id},
        )
        db.commit()

        # Get parser
        try:
            parser = get_parser(source_type)
        except ValueError:
            parser = get_parser_for_file(file_path)

        # Parse
        parse_result = parser.parse(file_path)

        # Store segments
        segment_ids = store_segments_sync(
            db=db,
            source_id=source_id,
            tenant_id=tenant_id,
            parsed_segments=parse_result.segments,
        )

        # Update source to parsed
        db.execute(
            text(
                "UPDATE sources SET parse_status = 'parsed', "
                "parser_version = '1.0.0' WHERE source_id = :sid"
            ),
            {"sid": source_id},
        )
        db.commit()

        # Update job to completed
        if job_id:
            now = datetime.now(timezone.utc)
            result_data = json.dumps({
                "segments_created": len(segment_ids),
                "source_type": source_type,
                "parser": type(parser).__name__,
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
            "Parsed source %s: %d segments created", source_id, len(segment_ids)
        )
        return {
            "source_id": source_id,
            "segments_created": len(segment_ids),
            "status": "completed",
        }

    except Exception as exc:
        logger.error("Parse failed for %s: %s", source_id, exc, exc_info=True)

        # Update source to failed
        try:
            db.execute(
                text(
                    "UPDATE sources SET parse_status = 'failed' "
                    "WHERE source_id = :sid"
                ),
                {"sid": source_id},
            )
            if job_id:
                error_data = json.dumps({
                    "error": str(exc),
                    "type": type(exc).__name__,
                })
                db.execute(
                    text(
                        "UPDATE jobs SET status = 'failed', "
                        "error_json = CAST(:err AS jsonb), "
                        "completed_at = :now "
                        "WHERE job_id = :jid"
                    ),
                    {
                        "err": error_data,
                        "now": datetime.now(timezone.utc),
                        "jid": job_id,
                    },
                )
            db.commit()
        except Exception:
            logger.error("Failed to update failure status", exc_info=True)

        # Retry with exponential backoff
        raise self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))

    finally:
        try:
            next(db_gen)
        except StopIteration:
            pass
