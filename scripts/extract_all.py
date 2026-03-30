#!/usr/bin/env python3
"""Batch extraction — run structured extraction on segments.

Selects segments that haven't been extracted yet, prioritizing:
1. Memory files (markdown docs)
2. Conversation summaries
3. Raw messages

Usage:
    python scripts/extract_all.py [--limit 100] [--tenant TENANT_ID] [--dry-run]
"""
import argparse
import json
import logging
import os
import sys
import time

# Ensure GAMI root is on path
gami_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, gami_root)

from sqlalchemy import text

from api.config import settings
from api.services.db import get_sync_db
from api.services.extraction import extract_all_from_segment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("extract_all")

# Delay between extraction calls to not overwhelm vLLM
INTER_SEGMENT_DELAY = 1.0


def get_unextracted_segments(db, tenant_id=None, limit=100):
    """Get segments not yet in provenance table, prioritized by type."""
    params = {"lim": limit}
    tenant_filter = ""
    if tenant_id:
        tenant_filter = "AND s.owner_tenant_id = :tid"
        params["tid"] = tenant_id

    # Priority: markdown/memory segments first, then sections/paragraphs, then messages
    query = f"""
        SELECT s.segment_id, s.source_id, s.owner_tenant_id, s.text,
               s.segment_type, s.token_count,
               src.source_type, src.title
        FROM segments s
        JOIN sources src ON s.source_id = src.source_id
        WHERE s.segment_id NOT IN (
            SELECT DISTINCT p.segment_id FROM provenance p WHERE p.segment_id IS NOT NULL
        )
        AND LENGTH(s.text) >= 50
        {tenant_filter}
        ORDER BY
            CASE src.source_type
                WHEN 'markdown' THEN 1
                WHEN 'sqlite_memory' THEN 2
                WHEN 'conversation_session' THEN 3
                ELSE 4
            END,
            CASE s.segment_type
                WHEN 'section' THEN 1
                WHEN 'paragraph' THEN 2
                WHEN 'chapter' THEN 3
                WHEN 'topic_episode' THEN 4
                WHEN 'message' THEN 5
                WHEN 'turn' THEN 6
                ELSE 7
            END,
            s.token_count DESC
        LIMIT :lim
    """

    rows = db.execute(text(query), params).fetchall()
    return [
        {
            "segment_id": r[0],
            "source_id": r[1],
            "tenant_id": r[2],
            "text": r[3],
            "segment_type": r[4],
            "token_count": r[5],
            "source_type": r[6],
            "source_title": r[7],
        }
        for r in rows
    ]


def main():
    parser = argparse.ArgumentParser(description="Batch extract from segments")
    parser.add_argument("--limit", type=int, default=100, help="Max segments to process")
    parser.add_argument("--tenant", default=None, help="Filter by tenant ID")
    parser.add_argument("--dry-run", action="store_true", help="Just show what would be processed")
    parser.add_argument("--delay", type=float, default=INTER_SEGMENT_DELAY, help="Seconds between segments")
    args = parser.parse_args()

    db_gen = get_sync_db()
    db = next(db_gen)

    try:
        segments = get_unextracted_segments(db, args.tenant, args.limit)
        logger.info("Found %d segments to extract", len(segments))

        if not segments:
            logger.info("Nothing to extract")
            return

        if args.dry_run:
            for seg in segments[:20]:
                logger.info(
                    "  [%s] %s / %s: %s (%d tokens)",
                    seg["source_type"], seg["source_title"],
                    seg["segment_type"], seg["segment_id"],
                    seg["token_count"] or 0,
                )
            if len(segments) > 20:
                logger.info("  ... and %d more", len(segments) - 20)
            return

        total_entities = 0
        total_claims = 0
        total_relations = 0
        total_events = 0
        errors = 0

        for i, seg in enumerate(segments):
            logger.info(
                "[%d/%d] Extracting from %s (%s / %s, %d tokens)",
                i + 1, len(segments), seg["segment_id"],
                seg["source_type"], seg["segment_type"],
                seg["token_count"] or 0,
            )

            try:
                result = extract_all_from_segment(
                    db=db,
                    segment_id=seg["segment_id"],
                    text_content=seg["text"],
                    source_id=seg["source_id"],
                    tenant_id=seg["tenant_id"],
                )

                total_entities += result["entities"]
                total_claims += result["claims"]
                total_relations += result["relations"]
                total_events += result["events"]

                logger.info(
                    "  → %d entities, %d claims, %d relations, %d events",
                    result["entities"], result["claims"],
                    result["relations"], result["events"],
                )

            except Exception as exc:
                logger.error("  → FAILED: %s", exc)
                errors += 1
                db.rollback()

            # Pace ourselves
            if i < len(segments) - 1:
                time.sleep(args.delay)

        logger.info(
            "Extraction complete: %d segments processed, %d errors",
            len(segments) - errors, errors,
        )
        logger.info(
            "Totals: %d entities, %d claims, %d relations, %d events",
            total_entities, total_claims, total_relations, total_events,
        )

    finally:
        try:
            next(db_gen)
        except StopIteration:
            pass


if __name__ == "__main__":
    main()
