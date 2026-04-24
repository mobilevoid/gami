#!/usr/bin/env python3
"""GAMI Cold Storage Archival Cron — archive stale segments and entities.

Runs daily to move old, low-importance data to cold storage.
Segments older than 90 days and entities older than 180 days are archived.
"""
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("gami.cold.cron")

GAMI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if GAMI_ROOT not in sys.path:
    sys.path.insert(0, GAMI_ROOT)


def main():
    try:
        from storage.cold.archiver import archive_stale
        from api.services.db import SyncSessionLocal
    except ImportError as e:
        logger.error("Failed to import GAMI modules: %s", e)
        sys.exit(1)

    db = SyncSessionLocal()
    try:
        # Archive segments older than 90 days
        logger.info("Archiving stale segments (>90 days)...")
        seg_result = archive_stale(db, "segment", days_stale=90, limit=200)
        logger.info("Segments: archived=%d, skipped=%d, failed=%d",
                     seg_result.get("archived", 0),
                     seg_result.get("skipped", 0),
                     seg_result.get("failed", 0))

        # Archive entities older than 180 days
        logger.info("Archiving stale entities (>180 days)...")
        ent_result = archive_stale(db, "entity", days_stale=180, limit=200)
        logger.info("Entities: archived=%d, skipped=%d, failed=%d",
                     ent_result.get("archived", 0),
                     ent_result.get("skipped", 0),
                     ent_result.get("failed", 0))

    finally:
        db.close()

    logger.info("Cold archival cron complete.")


if __name__ == "__main__":
    main()
