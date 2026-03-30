"""Importance Scoring Worker — periodically updates entity importance scores.

Computes importance_score based on:
- retrieval_count (how often queried)
- source_count (how many sources mention it)
- graph_centrality (how connected in the relation graph)
- mention_count (raw mention frequency)

Runs hourly via Celery Beat.
"""
import logging
import math
import os
import sys
from datetime import datetime, timezone

from sqlalchemy import text

from workers.celery_app import celery_app

logger = logging.getLogger("gami.workers.importance")

# Weight factors for importance score components
W_RETRIEVAL = 0.30
W_SOURCE = 0.25
W_CENTRALITY = 0.25
W_MENTION = 0.20

BATCH_SIZE = 500


@celery_app.task(
    name="gami.update_importance",
    bind=True,
    max_retries=1,
    soft_time_limit=300,
    time_limit=360,
)
def update_importance(self, tenant_id: str = None):
    """Recompute importance_score for all active entities.

    importance = W_RETRIEVAL * norm(retrieval_count)
               + W_SOURCE * norm(source_count)
               + W_CENTRALITY * centrality
               + W_MENTION * norm(mention_count)

    Where norm(x) = log(1 + x) / log(1 + max_x) to normalize to [0, 1].
    """
    gami_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if gami_root not in sys.path:
        sys.path.insert(0, gami_root)

    from api.services.db import get_sync_db

    db_gen = get_sync_db()
    db = next(db_gen)

    try:
        # Determine tenants
        if tenant_id:
            tenant_ids = [tenant_id]
        else:
            rows = db.execute(
                text("SELECT tenant_id FROM tenants WHERE status = 'active'")
            ).fetchall()
            tenant_ids = [r[0] for r in rows]

        total_updated = 0

        for tid in tenant_ids:
            # Step 1: Compute graph centrality (degree centrality)
            # Count inbound + outbound relations per entity
            centrality_rows = db.execute(
                text(
                    "SELECT node_id, COUNT(*) AS degree FROM ("
                    "  SELECT from_node_id AS node_id FROM relations "
                    "  WHERE owner_tenant_id = :tid AND status = 'active' "
                    "    AND from_node_type = 'entity' "
                    "  UNION ALL "
                    "  SELECT to_node_id AS node_id FROM relations "
                    "  WHERE owner_tenant_id = :tid AND status = 'active' "
                    "    AND to_node_type = 'entity' "
                    ") sub GROUP BY node_id"
                ),
                {"tid": tid},
            ).fetchall()

            degree_map = {r[0]: r[1] for r in centrality_rows}
            max_degree = max(degree_map.values()) if degree_map else 1

            # Step 2: Get max values for normalization
            maxes = db.execute(
                text(
                    "SELECT "
                    "  COALESCE(MAX(retrieval_count), 1), "
                    "  COALESCE(MAX(source_count), 1), "
                    "  COALESCE(MAX(mention_count), 1) "
                    "FROM entities "
                    "WHERE owner_tenant_id = :tid AND status != 'archived'"
                ),
                {"tid": tid},
            ).fetchone()

            max_retrieval = max(maxes[0], 1)
            max_source = max(maxes[1], 1)
            max_mention = max(maxes[2], 1)

            # Step 3: Fetch and update entities in batches
            offset = 0
            tenant_updated = 0

            while True:
                entities = db.execute(
                    text(
                        "SELECT entity_id, retrieval_count, source_count, "
                        "       mention_count "
                        "FROM entities "
                        "WHERE owner_tenant_id = :tid "
                        "  AND status != 'archived' "
                        "  AND merged_into_id IS NULL "
                        "ORDER BY entity_id "
                        "LIMIT :lim OFFSET :off"
                    ),
                    {"tid": tid, "lim": BATCH_SIZE, "off": offset},
                ).fetchall()

                if not entities:
                    break

                now = datetime.now(timezone.utc)

                for ent in entities:
                    eid = ent[0]
                    retrieval = ent[1] or 0
                    source = ent[2] or 0
                    mention = ent[3] or 0

                    # Normalize using log scale
                    norm_retrieval = math.log1p(retrieval) / math.log1p(max_retrieval)
                    norm_source = math.log1p(source) / math.log1p(max_source)
                    norm_mention = math.log1p(mention) / math.log1p(max_mention)

                    # Graph centrality (normalized degree)
                    degree = degree_map.get(eid, 0)
                    centrality = degree / max_degree if max_degree > 0 else 0

                    # Weighted importance score
                    importance = (
                        W_RETRIEVAL * norm_retrieval
                        + W_SOURCE * norm_source
                        + W_CENTRALITY * centrality
                        + W_MENTION * norm_mention
                    )

                    # Clamp to [0, 1]
                    importance = max(0.0, min(1.0, importance))

                    db.execute(
                        text(
                            "UPDATE entities SET "
                            "importance_score = :score, "
                            "graph_centrality = :centrality, "
                            "updated_at = :now "
                            "WHERE entity_id = :eid"
                        ),
                        {
                            "score": round(importance, 6),
                            "centrality": round(centrality, 6),
                            "now": now,
                            "eid": eid,
                        },
                    )
                    tenant_updated += 1

                db.commit()
                offset += len(entities)

                if len(entities) < BATCH_SIZE:
                    break

            total_updated += tenant_updated
            logger.info(
                "Importance scoring for tenant %s: %d entities updated",
                tid, tenant_updated,
            )

        return {
            "status": "completed",
            "entities_updated": total_updated,
            "tenants_processed": len(tenant_ids),
        }

    except Exception as exc:
        logger.error("Importance scoring failed: %s", exc, exc_info=True)
        db.rollback()
        raise self.retry(exc=exc, countdown=120)

    finally:
        try:
            next(db_gen)
        except StopIteration:
            pass


# Beat schedule is centralized in workers/celery_app.py
