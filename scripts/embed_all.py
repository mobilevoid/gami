#!/usr/bin/env python3
"""Batch embed all segments that lack embeddings.

Usage:
    python scripts/embed_all.py [--tenant TENANT_ID] [--batch-size 50]
"""
import argparse
import logging
import os
import sys
import time

# Ensure GAMI root is on path
gami_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, gami_root)

from api.config import settings
from api.llm.embeddings import embed_text_sync
from api.services.db import get_sync_db

from sqlalchemy import text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("embed_all")

BATCH_SIZE = 50
BATCH_DELAY = 0.3  # seconds between batches


def main():
    parser = argparse.ArgumentParser(description="Embed all segments")
    parser.add_argument("--tenant", default=None, help="Filter by tenant ID")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    db_gen = get_sync_db()
    db = next(db_gen)

    try:
        # Count
        where = "WHERE embedding IS NULL"
        params = {}
        if args.tenant:
            where += " AND owner_tenant_id = :tid"
            params["tid"] = args.tenant

        total = db.execute(
            text(f"SELECT count(*) FROM segments {where}"), params
        ).scalar()
        logger.info("Segments to embed: %d", total)

        if total == 0 or args.dry_run:
            return

        embedded = 0
        errors = 0

        while True:
            rows = db.execute(
                text(
                    f"SELECT segment_id, text FROM segments {where} "
                    f"ORDER BY created_at LIMIT :lim"
                ),
                {**params, "lim": args.batch_size},
            ).fetchall()

            if not rows:
                break

            for row in rows:
                seg_id, seg_text = row[0], row[1]
                try:
                    embed_input = seg_text[:16000] if len(seg_text) > 16000 else seg_text
                    embedding = embed_text_sync(embed_input)
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
                        db.commit()
                        logger.info("Progress: %d / %d embedded", embedded, total)

                except Exception as exc:
                    logger.warning("Failed to embed %s: %s", seg_id, exc)
                    errors += 1

            db.commit()
            time.sleep(BATCH_DELAY)

        db.commit()
        logger.info(
            "Done: %d embedded, %d errors, %d total", embedded, errors, total
        )

    finally:
        try:
            next(db_gen)
        except StopIteration:
            pass


if __name__ == "__main__":
    main()
