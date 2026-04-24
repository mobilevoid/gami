#!/usr/bin/env python3
"""Migrate existing embeddings to manifold_embeddings table.

This script copies existing embeddings from segments, entities, claims,
and memories into the manifold_embeddings table with manifold_type='topic'.

Usage:
    python migrate_topic_embeddings.py --tenant my-tenant
    python migrate_topic_embeddings.py --tenant shared --batch-size 1000
    python migrate_topic_embeddings.py --all --dry-run

Requirements:
    - manifold tables must exist (run 002_manifold_tables.sql first)
    - existing embeddings in source tables

Safety:
    - Non-destructive: does not modify source tables
    - Idempotent: can be run multiple times safely (uses ON CONFLICT)
    - Supports dry-run mode
"""
import argparse
import logging
import sys
from datetime import datetime

# NOTE: This script is a template. It will be connected to the
# actual database during manifold system activation.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [MIGRATE] %(message)s",
)
log = logging.getLogger("migrate_topic")


def migrate_segments(tenant_id: str, batch_size: int, dry_run: bool) -> int:
    """Migrate segment embeddings to manifold_embeddings.

    Args:
        tenant_id: Tenant to migrate.
        batch_size: Batch size for inserts.
        dry_run: If True, don't actually insert.

    Returns:
        Number of embeddings migrated.
    """
    log.info(f"Migrating segment embeddings for tenant '{tenant_id}'")

    # SQL to copy embeddings
    sql = """
        INSERT INTO manifold_embeddings (
            target_id, target_type, manifold_type,
            embedding, embedding_model, embedding_version,
            canonical_form, created_at, updated_at
        )
        SELECT
            segment_id, 'segment', 'topic',
            embedding, 'nomic-embed-text', 1,
            LEFT(text, 500), NOW(), NOW()
        FROM segments
        WHERE owner_tenant_id = :tenant_id
        AND embedding IS NOT NULL
        ON CONFLICT (target_id, target_type, manifold_type) DO NOTHING
    """

    if dry_run:
        log.info(f"[DRY RUN] Would migrate segments for {tenant_id}")
        return 0

    # STUB: Not connected to database in isolated module
    log.warning("Database not connected in isolated module")
    return 0


def migrate_entities(tenant_id: str, batch_size: int, dry_run: bool) -> int:
    """Migrate entity embeddings to manifold_embeddings."""
    log.info(f"Migrating entity embeddings for tenant '{tenant_id}'")

    sql = """
        INSERT INTO manifold_embeddings (
            target_id, target_type, manifold_type,
            embedding, embedding_model, embedding_version,
            canonical_form, created_at, updated_at
        )
        SELECT
            entity_id, 'entity', 'topic',
            embedding, 'nomic-embed-text', 1,
            canonical_name || ': ' || COALESCE(description, ''), NOW(), NOW()
        FROM entities
        WHERE owner_tenant_id = :tenant_id
        AND embedding IS NOT NULL
        ON CONFLICT (target_id, target_type, manifold_type) DO NOTHING
    """

    if dry_run:
        log.info(f"[DRY RUN] Would migrate entities for {tenant_id}")
        return 0

    log.warning("Database not connected in isolated module")
    return 0


def migrate_claims(tenant_id: str, batch_size: int, dry_run: bool) -> int:
    """Migrate claim embeddings to manifold_embeddings."""
    log.info(f"Migrating claim embeddings for tenant '{tenant_id}'")

    if dry_run:
        log.info(f"[DRY RUN] Would migrate claims for {tenant_id}")
        return 0

    log.warning("Database not connected in isolated module")
    return 0


def migrate_memories(tenant_id: str, batch_size: int, dry_run: bool) -> int:
    """Migrate memory embeddings to manifold_embeddings."""
    log.info(f"Migrating memory embeddings for tenant '{tenant_id}'")

    if dry_run:
        log.info(f"[DRY RUN] Would migrate memories for {tenant_id}")
        return 0

    log.warning("Database not connected in isolated module")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Migrate embeddings to manifold_embeddings")
    parser.add_argument("--tenant", type=str, help="Tenant ID to migrate")
    parser.add_argument("--all", action="store_true", help="Migrate all tenants")
    parser.add_argument("--batch-size", type=int, default=500, help="Batch size")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually insert")
    parser.add_argument("--type", type=str, default="all",
                       choices=["all", "segments", "entities", "claims", "memories"],
                       help="Object type to migrate")

    args = parser.parse_args()

    if not args.tenant and not args.all:
        parser.error("Must specify --tenant or --all")

    tenants = [args.tenant] if args.tenant else ["shared"]  # Add your tenant names here

    total = 0
    for tenant in tenants:
        log.info(f"\n{'='*60}")
        log.info(f"Processing tenant: {tenant}")
        log.info(f"{'='*60}")

        if args.type in ("all", "segments"):
            total += migrate_segments(tenant, args.batch_size, args.dry_run)

        if args.type in ("all", "entities"):
            total += migrate_entities(tenant, args.batch_size, args.dry_run)

        if args.type in ("all", "claims"):
            total += migrate_claims(tenant, args.batch_size, args.dry_run)

        if args.type in ("all", "memories"):
            total += migrate_memories(tenant, args.batch_size, args.dry_run)

    log.info(f"\nTotal embeddings migrated: {total}")

    if args.dry_run:
        log.info("(This was a dry run - no changes made)")


if __name__ == "__main__":
    main()
