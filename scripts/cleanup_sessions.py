#!/usr/bin/env python3
"""Cleanup stale sessions and old retrieval logs.

This script should be run periodically (e.g., daily via cron) to:
1. Mark stale sessions as ended
2. Archive old retrieval logs
3. Clean up orphaned hot context in Redis

Usage:
    python scripts/cleanup_sessions.py [--dry-run] [--session-ttl-hours 24] [--log-retention-days 30]
"""
import argparse
import logging
import os
import sys
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text
from api.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("gami.cleanup")


def cleanup_sessions(engine, ttl_hours: int, dry_run: bool) -> int:
    """Mark stale sessions as ended."""
    cutoff = datetime.utcnow() - timedelta(hours=ttl_hours)

    with engine.connect() as conn:
        if dry_run:
            result = conn.execute(text("""
                SELECT COUNT(*) FROM sessions
                WHERE ended_at IS NULL
                AND last_activity_at < :cutoff
            """), {"cutoff": cutoff})
            count = result.scalar()
            logger.info(f"[DRY RUN] Would end {count} stale sessions")
            return count

        result = conn.execute(text("""
            UPDATE sessions SET
                ended_at = NOW(),
                conversation_state = 'expired'
            WHERE ended_at IS NULL
            AND last_activity_at < :cutoff
            RETURNING session_id
        """), {"cutoff": cutoff})
        count = result.rowcount
        conn.commit()
        logger.info(f"Ended {count} stale sessions (inactive > {ttl_hours}h)")
        return count


def cleanup_retrieval_logs(engine, retention_days: int, dry_run: bool) -> int:
    """Archive old retrieval logs (move processed ones to archive)."""
    cutoff = datetime.utcnow() - timedelta(days=retention_days)

    with engine.connect() as conn:
        if dry_run:
            result = conn.execute(text("""
                SELECT COUNT(*) FROM retrieval_logs
                WHERE created_at < :cutoff
                AND processed_in_dream = true
            """), {"cutoff": cutoff})
            count = result.scalar()
            logger.info(f"[DRY RUN] Would archive {count} old retrieval logs")
            return count

        # For now, just delete very old processed logs
        # In production, you might want to move to an archive table
        result = conn.execute(text("""
            DELETE FROM retrieval_logs
            WHERE created_at < :cutoff
            AND processed_in_dream = true
            RETURNING log_id
        """), {"cutoff": cutoff})
        count = result.rowcount
        conn.commit()
        logger.info(f"Archived {count} old retrieval logs (> {retention_days} days)")
        return count


def cleanup_redis_hot_context(ttl_hours: int, dry_run: bool) -> int:
    """Clean up orphaned hot context keys in Redis."""
    try:
        import redis

        r = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
        )

        # Find all hot context keys
        keys = list(r.scan_iter("gami:session:*:hot"))

        if dry_run:
            logger.info(f"[DRY RUN] Found {len(keys)} hot context keys")
            return len(keys)

        # Redis keys should have TTL set automatically
        # This just reports on any without TTL
        no_ttl = 0
        for key in keys:
            if r.ttl(key) == -1:  # No TTL set
                r.expire(key, ttl_hours * 3600)
                no_ttl += 1

        logger.info(f"Set TTL on {no_ttl} hot context keys without expiry")
        r.close()
        return no_ttl

    except Exception as e:
        logger.warning(f"Redis cleanup failed: {e}")
        return 0


def cleanup_memory_clusters(engine, dry_run: bool) -> int:
    """Remove fully decayed memory clusters."""
    with engine.connect() as conn:
        if dry_run:
            result = conn.execute(text("""
                SELECT COUNT(*) FROM memory_clusters
                WHERE current_decay <= 0.05
                AND status = 'active'
            """))
            count = result.scalar()
            logger.info(f"[DRY RUN] Would archive {count} fully decayed clusters")
            return count

        result = conn.execute(text("""
            UPDATE memory_clusters SET
                status = 'archived',
                updated_at = NOW()
            WHERE current_decay <= 0.05
            AND status = 'active'
            RETURNING cluster_id
        """))
        count = result.rowcount
        conn.commit()
        logger.info(f"Archived {count} fully decayed memory clusters")
        return count


def main():
    parser = argparse.ArgumentParser(description="Clean up stale GAMI data")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--session-ttl-hours", type=int, default=24, help="Session TTL in hours")
    parser.add_argument("--log-retention-days", type=int, default=30, help="Log retention in days")
    args = parser.parse_args()

    logger.info("Starting GAMI cleanup...")
    if args.dry_run:
        logger.info("DRY RUN MODE - no changes will be made")

    engine = create_engine(settings.DATABASE_URL_SYNC)

    stats = {
        "sessions_ended": cleanup_sessions(engine, args.session_ttl_hours, args.dry_run),
        "logs_archived": cleanup_retrieval_logs(engine, args.log_retention_days, args.dry_run),
        "clusters_archived": cleanup_memory_clusters(engine, args.dry_run),
        "redis_keys_fixed": cleanup_redis_hot_context(args.session_ttl_hours, args.dry_run),
    }

    logger.info(f"Cleanup complete: {stats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
