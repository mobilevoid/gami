#!/usr/bin/env python3
"""Migrate existing procedures to workflow memories.

SAFETY: Original procedures are NOT deleted, only marked as migrated.
Run this after Phase B (memory primary) is stable.

Usage:
    python scripts/migrate_procedures_to_memories.py [--dry-run] [--tenant TENANT_ID]
"""
import sys
import argparse
sys.path.insert(0, '/opt/gami')

from sqlalchemy import create_engine, text
from api.config import settings
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
log = logging.getLogger(__name__)


def gen_id(prefix: str, seed: str) -> str:
    """Generate a deterministic ID."""
    import hashlib
    h = hashlib.sha256(seed.encode()).hexdigest()[:16]
    return f"{prefix}_{h}"


def migrate(dry_run: bool = False, tenant_id: str | None = None):
    """Convert procedures to workflow memories.

    Args:
        dry_run: If True, only report what would be done
        tenant_id: If provided, only migrate for this tenant
    """
    engine = create_engine(settings.DATABASE_URL_SYNC)

    with engine.connect() as conn:
        # Get all active procedures (optionally filtered by tenant)
        query = """
            SELECT procedure_id, owner_tenant_id, name, description, steps,
                   confidence, category
            FROM procedures
            WHERE status = 'active'
        """
        params = {}
        if tenant_id:
            query += " AND owner_tenant_id = :tid"
            params["tid"] = tenant_id

        procedures = conn.execute(text(query), params).fetchall()

        log.info(f"Found {len(procedures)} active procedures to migrate")

        if not procedures:
            log.info("Nothing to migrate")
            return {"migrated": 0, "skipped": 0}

        migrated = 0
        skipped = 0

        for proc in procedures:
            # Format procedure as workflow description
            steps_text = ""
            if proc.steps:
                steps = json.loads(proc.steps) if isinstance(proc.steps, str) else proc.steps
                if isinstance(steps, list):
                    steps_text = " → ".join(
                        s.get('action', str(s))[:50] for s in steps[:10]
                    )

            workflow_text = f"Workflow: {proc.name}. {proc.description or ''}"
            if steps_text:
                workflow_text += f" Steps: {steps_text}"

            if proc.category:
                workflow_text += f" Category: {proc.category}"

            # Check if already migrated
            memory_id = f"MEM_PROC_{proc.procedure_id}"
            existing = conn.execute(text("""
                SELECT memory_id FROM assistant_memories
                WHERE memory_id = :mid
            """), {"mid": memory_id}).fetchone()

            if existing:
                log.info(f"  Skipping {proc.name} - already migrated")
                skipped += 1
                continue

            if dry_run:
                log.info(f"  [DRY RUN] Would migrate: {proc.name}")
                log.info(f"    → {workflow_text[:100]}...")
                migrated += 1
                continue

            # Generate embedding
            try:
                from api.llm.embeddings import embed_text_sync
                emb = embed_text_sync(workflow_text[:2000])
                vec_str = "[" + ",".join(str(v) for v in emb) + "]"
            except Exception as e:
                log.warning(f"  Embedding failed for {proc.name}: {e}")
                # Skip this one, can retry later
                continue

            # Create workflow memory
            conn.execute(text("""
                INSERT INTO assistant_memories
                (memory_id, owner_tenant_id, normalized_text, memory_type,
                 embedding, importance_score, status, created_at)
                VALUES (:mid, :tid, :txt, 'workflow',
                        CAST(:vec AS vector), :conf, 'active', NOW())
                ON CONFLICT (memory_id) DO NOTHING
            """), {
                "mid": memory_id,
                "tid": proc.owner_tenant_id,
                "txt": workflow_text,
                "vec": vec_str,
                "conf": proc.confidence or 0.6,
            })

            # Mark original as migrated (NOT deleted!)
            conn.execute(text("""
                UPDATE procedures
                SET status = 'migrated',
                    updated_at = NOW()
                WHERE procedure_id = :pid
            """), {"pid": proc.procedure_id})

            migrated += 1
            log.info(f"  Migrated: {proc.name}")

        if not dry_run:
            conn.commit()

        log.info(f"\nMigration complete: {migrated} migrated, {skipped} skipped")
        if not dry_run:
            log.info("Original procedures preserved with status='migrated'")

        return {"migrated": migrated, "skipped": skipped}


def main():
    parser = argparse.ArgumentParser(description="Migrate procedures to workflow memories")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be done without making changes")
    parser.add_argument("--tenant", type=str, help="Only migrate for specific tenant")
    args = parser.parse_args()

    result = migrate(dry_run=args.dry_run, tenant_id=args.tenant)

    if args.dry_run:
        print(f"\n[DRY RUN] Would migrate {result['migrated']} procedures")
    else:
        print(f"\nMigrated {result['migrated']} procedures, skipped {result['skipped']}")


if __name__ == "__main__":
    main()
