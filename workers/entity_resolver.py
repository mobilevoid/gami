"""Entity Resolution Worker — finds and proposes merges for similar entities.

Runs periodically to find entities with similar names or embeddings within
a tenant, and proposes merges via the proposed_changes table. Never commits
direct changes — all merges require review.
"""
import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone

from sqlalchemy import text

from workers.celery_app import celery_app

logger = logging.getLogger("gami.workers.entity_resolver")

SIMILARITY_THRESHOLD = 0.85
MAX_PROPOSALS_PER_RUN = 50


def _make_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


@celery_app.task(
    name="gami.resolve_entities",
    bind=True,
    max_retries=1,
    soft_time_limit=600,
    time_limit=660,
)
def resolve_entities(self, tenant_id: str = None):
    """Find entities with similar names/embeddings and propose merges.

    Strategy:
    1. Name-based: Levenshtein distance on canonical_name within same type
    2. Embedding-based: cosine similarity > SIMILARITY_THRESHOLD
    3. Only propose when confidence > 0.85
    4. Check daily budget before creating proposals
    """
    gami_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if gami_root not in sys.path:
        sys.path.insert(0, gami_root)

    from api.services.db import get_sync_db

    db_gen = get_sync_db()
    db = next(db_gen)

    try:
        # Determine tenants to process
        if tenant_id:
            tenant_ids = [tenant_id]
        else:
            rows = db.execute(
                text("SELECT tenant_id FROM tenants WHERE status = 'active'")
            ).fetchall()
            tenant_ids = [r[0] for r in rows]

        total_proposals = 0

        for tid in tenant_ids:
            # Check daily budget
            today_start = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            budget_row = db.execute(
                text(
                    "SELECT daily_write_budget FROM tenants WHERE tenant_id = :tid"
                ),
                {"tid": tid},
            ).fetchone()
            daily_budget = budget_row[0] if budget_row and budget_row[0] else 10000

            existing_proposals = db.execute(
                text(
                    "SELECT COUNT(*) FROM proposed_changes "
                    "WHERE proposer_tenant_id = 'background-worker' "
                    "AND change_type = 'entity_merge' "
                    "AND created_at >= :today"
                ),
                {"today": today_start},
            ).scalar()

            remaining_budget = daily_budget - (existing_proposals or 0)
            if remaining_budget <= 0:
                logger.info(
                    "Daily budget exhausted for tenant %s (%d proposals today)",
                    tid, existing_proposals,
                )
                continue

            # --- Name-based resolution ---
            name_proposals = _find_name_duplicates(db, tid)

            # --- Embedding-based resolution ---
            embedding_proposals = _find_embedding_duplicates(db, tid)

            # Merge and deduplicate proposals (by entity pair)
            seen_pairs = set()
            all_proposals = []

            for prop in name_proposals + embedding_proposals:
                pair = tuple(sorted([prop["entity_a"], prop["entity_b"]]))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    all_proposals.append(prop)

            # Check for already-pending proposals for same pairs
            for prop in all_proposals:
                if total_proposals >= MAX_PROPOSALS_PER_RUN:
                    break
                if total_proposals >= remaining_budget:
                    break

                pair_a, pair_b = prop["entity_a"], prop["entity_b"]

                existing = db.execute(
                    text(
                        "SELECT 1 FROM proposed_changes "
                        "WHERE change_type = 'entity_merge' "
                        "AND status = 'pending' "
                        "AND ("
                        "  (proposed_state_json->>'merge_from' = :a "
                        "   AND proposed_state_json->>'merge_into' = :b) "
                        "  OR "
                        "  (proposed_state_json->>'merge_from' = :b "
                        "   AND proposed_state_json->>'merge_into' = :a) "
                        ") LIMIT 1"
                    ),
                    {"a": pair_a, "b": pair_b},
                ).fetchone()

                if existing:
                    continue

                # Determine which entity to keep (higher mention_count)
                counts = db.execute(
                    text(
                        "SELECT entity_id, mention_count, source_count "
                        "FROM entities WHERE entity_id IN (:a, :b)"
                    ),
                    {"a": pair_a, "b": pair_b},
                ).fetchall()

                if len(counts) < 2:
                    continue

                counts_map = {r[0]: (r[1], r[2]) for r in counts}
                a_score = counts_map.get(pair_a, (0, 0))
                b_score = counts_map.get(pair_b, (0, 0))

                if (a_score[0] + a_score[1]) >= (b_score[0] + b_score[1]):
                    merge_into, merge_from = pair_a, pair_b
                else:
                    merge_into, merge_from = pair_b, pair_a

                proposal_id = _make_id("PROP")
                now = datetime.now(timezone.utc)

                db.execute(
                    text(
                        "INSERT INTO proposed_changes "
                        "(proposal_id, proposer_tenant_id, change_type, "
                        "target_type, target_id, proposed_state_json, "
                        "reason, confidence, evidence_ids, status, created_at, "
                        "expires_at) "
                        "VALUES (:pid, 'background-worker', 'entity_merge', "
                        "'entity', :target, CAST(:state AS jsonb), "
                        ":reason, :conf, CAST(:evidence AS jsonb), "
                        "'pending', :now, :expires)"
                    ),
                    {
                        "pid": proposal_id,
                        "target": merge_into,
                        "state": json.dumps({
                            "merge_from": merge_from,
                            "merge_into": merge_into,
                            "name_a": prop["name_a"],
                            "name_b": prop["name_b"],
                            "type": prop["entity_type"],
                            "method": prop["method"],
                        }),
                        "reason": prop["reason"],
                        "conf": prop["confidence"],
                        "evidence": json.dumps(prop.get("evidence_ids", [])),
                        "now": now,
                        "expires": now.replace(
                            day=now.day + 7 if now.day <= 24 else 1,
                            month=now.month if now.day <= 24 else now.month + 1,
                        ) if False else None,  # Let it persist
                    },
                )
                total_proposals += 1

            db.commit()
            logger.info(
                "Entity resolution for tenant %s: %d proposals created",
                tid, total_proposals,
            )

        return {
            "status": "completed",
            "proposals_created": total_proposals,
            "tenants_processed": len(tenant_ids),
        }

    except Exception as exc:
        logger.error("Entity resolution failed: %s", exc, exc_info=True)
        db.rollback()
        raise self.retry(exc=exc, countdown=300)

    finally:
        try:
            next(db_gen)
        except StopIteration:
            pass


def _find_name_duplicates(db, tenant_id: str) -> list[dict]:
    """Find entities with similar canonical names using trigram similarity.

    Falls back to exact prefix matching if pg_trgm is not available.
    """
    proposals = []

    try:
        # Try trigram similarity (requires pg_trgm extension)
        rows = db.execute(
            text(
                "SELECT a.entity_id, a.canonical_name, a.entity_type, "
                "       b.entity_id, b.canonical_name, "
                "       similarity(LOWER(a.canonical_name), LOWER(b.canonical_name)) AS sim "
                "FROM entities a "
                "JOIN entities b ON a.entity_type = b.entity_type "
                "  AND a.entity_id < b.entity_id "
                "  AND a.owner_tenant_id = b.owner_tenant_id "
                "WHERE a.owner_tenant_id = :tid "
                "  AND a.status != 'archived' AND b.status != 'archived' "
                "  AND a.merged_into_id IS NULL AND b.merged_into_id IS NULL "
                "  AND similarity(LOWER(a.canonical_name), LOWER(b.canonical_name)) > :thresh "
                "ORDER BY sim DESC "
                "LIMIT 100"
            ),
            {"tid": tenant_id, "thresh": SIMILARITY_THRESHOLD},
        ).fetchall()

        for row in rows:
            proposals.append({
                "entity_a": row[0],
                "name_a": row[1],
                "entity_type": row[2],
                "entity_b": row[3],
                "name_b": row[4],
                "confidence": float(row[5]),
                "method": "name_trigram",
                "reason": (
                    f"Name similarity {row[5]:.2f}: "
                    f"'{row[1]}' and '{row[4]}' (type: {row[2]})"
                ),
            })

    except Exception:
        # pg_trgm not available — fall back to normalized prefix matching
        logger.info("pg_trgm not available, using prefix matching for tenant %s", tenant_id)
        rows = db.execute(
            text(
                "SELECT a.entity_id, a.canonical_name, a.entity_type, "
                "       b.entity_id, b.canonical_name "
                "FROM entities a "
                "JOIN entities b ON a.entity_type = b.entity_type "
                "  AND a.entity_id < b.entity_id "
                "  AND a.owner_tenant_id = b.owner_tenant_id "
                "WHERE a.owner_tenant_id = :tid "
                "  AND a.status != 'archived' AND b.status != 'archived' "
                "  AND a.merged_into_id IS NULL AND b.merged_into_id IS NULL "
                "  AND LOWER(a.canonical_name) = LOWER(b.canonical_name) "
                "LIMIT 100"
            ),
            {"tid": tenant_id},
        ).fetchall()

        for row in rows:
            proposals.append({
                "entity_a": row[0],
                "name_a": row[1],
                "entity_type": row[2],
                "entity_b": row[3],
                "name_b": row[4],
                "confidence": 1.0,
                "method": "name_exact",
                "reason": f"Exact name match: '{row[1]}' (type: {row[2]})",
            })

    return proposals


def _find_embedding_duplicates(db, tenant_id: str) -> list[dict]:
    """Find entities with highly similar embeddings."""
    proposals = []

    try:
        rows = db.execute(
            text(
                "SELECT a.entity_id, a.canonical_name, a.entity_type, "
                "       b.entity_id, b.canonical_name, "
                "       1 - (a.embedding <=> b.embedding) AS cosine_sim "
                "FROM entities a "
                "JOIN entities b ON a.entity_type = b.entity_type "
                "  AND a.entity_id < b.entity_id "
                "  AND a.owner_tenant_id = b.owner_tenant_id "
                "WHERE a.owner_tenant_id = :tid "
                "  AND a.embedding IS NOT NULL AND b.embedding IS NOT NULL "
                "  AND a.status != 'archived' AND b.status != 'archived' "
                "  AND a.merged_into_id IS NULL AND b.merged_into_id IS NULL "
                "  AND 1 - (a.embedding <=> b.embedding) > :thresh "
                "  AND LOWER(a.canonical_name) != LOWER(b.canonical_name) "
                "ORDER BY cosine_sim DESC "
                "LIMIT 100"
            ),
            {"tid": tenant_id, "thresh": SIMILARITY_THRESHOLD},
        ).fetchall()

        for row in rows:
            proposals.append({
                "entity_a": row[0],
                "name_a": row[1],
                "entity_type": row[2],
                "entity_b": row[3],
                "name_b": row[4],
                "confidence": float(row[5]),
                "method": "embedding_cosine",
                "reason": (
                    f"Embedding similarity {row[5]:.3f}: "
                    f"'{row[1]}' and '{row[4]}' (type: {row[2]})"
                ),
            })

    except Exception as exc:
        logger.warning("Embedding duplicate search failed: %s", exc)

    return proposals


# Beat schedule is centralized in workers/celery_app.py
