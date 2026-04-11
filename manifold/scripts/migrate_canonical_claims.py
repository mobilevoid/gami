#!/usr/bin/env python3
"""Migrate existing claims to canonical SPO form.

This script processes all claims in the database and:
1. Extracts subject-predicate-object structure
2. Normalizes to canonical form
3. Stores in canonical_claims table
4. Updates claim embeddings with SPO-aware vectors

Usage:
    python migrate_canonical_claims.py [--batch-size 100] [--dry-run]
"""
import argparse
import asyncio
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from datetime import datetime
from typing import List, Optional

# Import will work after GAMI is fully installed
try:
    import asyncpg
    from manifold.canonical.claim_normalizer import ClaimNormalizer
    from manifold.config import get_config
except ImportError:
    asyncpg = None
    ClaimNormalizer = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("migrate_claims")


async def get_db_connection():
    """Get database connection."""
    config = get_config()
    db_url = os.environ.get("DATABASE_URL", config.database_url)
    return await asyncpg.connect(db_url)


async def get_claims_batch(conn, offset: int, limit: int) -> List[dict]:
    """Fetch batch of claims from database."""
    rows = await conn.fetch(
        """
        SELECT
            c.id,
            c.text,
            c.confidence,
            c.source_segment_id,
            c.created_at
        FROM claims c
        LEFT JOIN canonical_claims cc ON c.id = cc.claim_id
        WHERE cc.claim_id IS NULL  -- Not yet migrated
        ORDER BY c.created_at
        OFFSET $1 LIMIT $2
        """,
        offset,
        limit,
    )
    return [dict(r) for r in rows]


async def insert_canonical_claim(
    conn,
    claim_id: str,
    subject: str,
    predicate: str,
    obj: str,
    modality: str,
    temporal_scope: Optional[str],
    qualifiers: dict,
    canonical_text: str,
) -> bool:
    """Insert canonical claim into database."""
    try:
        await conn.execute(
            """
            INSERT INTO canonical_claims (
                claim_id, subject, predicate, object,
                modality, temporal_scope, qualifiers,
                canonical_text, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (claim_id) DO UPDATE SET
                subject = EXCLUDED.subject,
                predicate = EXCLUDED.predicate,
                object = EXCLUDED.object,
                modality = EXCLUDED.modality,
                temporal_scope = EXCLUDED.temporal_scope,
                qualifiers = EXCLUDED.qualifiers,
                canonical_text = EXCLUDED.canonical_text,
                updated_at = NOW()
            """,
            claim_id,
            subject,
            predicate,
            obj,
            modality,
            temporal_scope,
            qualifiers,
            canonical_text,
            datetime.utcnow(),
        )
        return True
    except Exception as e:
        logger.error(f"Failed to insert canonical claim {claim_id}: {e}")
        return False


async def migrate_batch(
    conn,
    normalizer: "ClaimNormalizer",
    claims: List[dict],
    dry_run: bool = False,
) -> tuple:
    """Migrate a batch of claims. Returns (success_count, error_count)."""
    success = 0
    errors = 0

    for claim in claims:
        try:
            # Normalize claim to SPO form
            result = normalizer.normalize(claim["text"])

            if dry_run:
                logger.info(
                    f"[DRY RUN] Claim {claim['id']}: "
                    f"'{claim['text'][:50]}...' -> "
                    f"({result.subject}, {result.predicate}, {result.object})"
                )
                success += 1
                continue

            # Insert canonical form
            ok = await insert_canonical_claim(
                conn,
                claim_id=claim["id"],
                subject=result.subject,
                predicate=result.predicate,
                obj=result.object,
                modality=result.modality.value if hasattr(result.modality, "value") else str(result.modality),
                temporal_scope=result.temporal_scope,
                qualifiers=result.qualifiers,
                canonical_text=result.canonical_text,
            )

            if ok:
                success += 1
            else:
                errors += 1

        except Exception as e:
            logger.error(f"Error processing claim {claim['id']}: {e}")
            errors += 1

    return success, errors


async def main(batch_size: int = 100, dry_run: bool = False):
    """Main migration loop."""
    if asyncpg is None:
        logger.error("asyncpg not installed. Run: pip install asyncpg")
        return

    if ClaimNormalizer is None:
        logger.error("ClaimNormalizer not available. Check imports.")
        return

    logger.info(f"Starting canonical claims migration (batch_size={batch_size}, dry_run={dry_run})")

    conn = await get_db_connection()
    normalizer = ClaimNormalizer()

    total_success = 0
    total_errors = 0
    offset = 0

    try:
        while True:
            claims = await get_claims_batch(conn, offset, batch_size)

            if not claims:
                logger.info("No more claims to process")
                break

            logger.info(f"Processing batch at offset {offset} ({len(claims)} claims)")

            success, errors = await migrate_batch(conn, normalizer, claims, dry_run)
            total_success += success
            total_errors += errors

            offset += batch_size

            # Progress report
            logger.info(
                f"Progress: {total_success} migrated, {total_errors} errors"
            )

    finally:
        await conn.close()

    logger.info(
        f"Migration complete: {total_success} claims migrated, {total_errors} errors"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate claims to canonical form")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of claims per batch",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing to database",
    )
    args = parser.parse_args()

    asyncio.run(main(batch_size=args.batch_size, dry_run=args.dry_run))
