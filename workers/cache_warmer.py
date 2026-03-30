"""Cache Warming Worker — precomputes hot entity neighborhoods in Redis.

Precomputes:
1. Entity neighborhood graphs (2-hop) for high-importance entities
2. Frequent query anchor sets based on recent retrieval patterns
3. Entity summaries with related claims

All cached with 6-hour TTL in Redis.
"""
import json
import logging
import os
import sys
from datetime import datetime, timezone

import redis
from sqlalchemy import text

from workers.celery_app import celery_app

logger = logging.getLogger("gami.workers.cache_warmer")

CACHE_TTL = 6 * 3600  # 6 hours
TOP_ENTITIES = 200  # Warm top N entities by importance
NEIGHBORHOOD_HOPS = 2
MAX_NEIGHBORS = 50


@celery_app.task(
    name="gami.warm_cache",
    bind=True,
    max_retries=1,
    soft_time_limit=300,
    time_limit=360,
)
def warm_cache(self, tenant_id: str = None):
    """Precompute and cache hot entity neighborhoods and query anchors.

    Steps:
    1. Fetch top entities by importance_score
    2. For each, compute 2-hop neighborhood (relations + connected entities)
    3. Cache in Redis with gami:neighborhood:{entity_id} key
    4. Cache frequent query patterns as anchor sets
    """
    gami_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if gami_root not in sys.path:
        sys.path.insert(0, gami_root)

    from api.config import settings
    from api.services.db import get_sync_db

    db_gen = get_sync_db()
    db = next(db_gen)

    r = redis.Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=settings.REDIS_DB,
    )

    try:
        # Determine tenants
        if tenant_id:
            tenant_ids = [tenant_id]
        else:
            rows = db.execute(
                text("SELECT tenant_id FROM tenants WHERE status = 'active'")
            ).fetchall()
            tenant_ids = [r_[0] for r_ in rows]

        total_cached = 0
        total_anchors = 0

        for tid in tenant_ids:
            # --- Entity Neighborhood Caching ---
            top_entities = db.execute(
                text(
                    "SELECT entity_id, canonical_name, entity_type, "
                    "       importance_score, description "
                    "FROM entities "
                    "WHERE owner_tenant_id = :tid "
                    "  AND status IN ('active', 'confirmed', 'provisional') "
                    "  AND merged_into_id IS NULL "
                    "ORDER BY importance_score DESC "
                    "LIMIT :lim"
                ),
                {"tid": tid, "lim": TOP_ENTITIES},
            ).fetchall()

            for ent in top_entities:
                eid = ent[0]
                ename = ent[1]
                etype = ent[2]
                importance = ent[3]
                description = ent[4]

                neighborhood = _compute_neighborhood(db, eid, tid)

                # Fetch related claims
                claims = db.execute(
                    text(
                        "SELECT claim_id, predicate, summary_text, confidence "
                        "FROM claims "
                        "WHERE owner_tenant_id = :tid "
                        "  AND status = 'active' "
                        "  AND (subject_entity_id = :eid OR object_entity_id = :eid) "
                        "ORDER BY confidence DESC "
                        "LIMIT 20"
                    ),
                    {"tid": tid, "eid": eid},
                ).fetchall()

                cache_data = {
                    "entity_id": eid,
                    "name": ename,
                    "type": etype,
                    "importance": importance,
                    "description": description or "",
                    "neighbors": neighborhood,
                    "claims": [
                        {
                            "claim_id": c[0],
                            "predicate": c[1],
                            "summary": c[2],
                            "confidence": c[3],
                        }
                        for c in claims
                    ],
                    "cached_at": datetime.now(timezone.utc).isoformat(),
                }

                cache_key = f"gami:neighborhood:{eid}"
                r.setex(
                    cache_key,
                    CACHE_TTL,
                    json.dumps(cache_data, default=str),
                )
                total_cached += 1

            # --- Anchor Set Caching ---
            # Cache entity lookup by name for fast resolution
            all_entities = db.execute(
                text(
                    "SELECT entity_id, canonical_name, entity_type, "
                    "       aliases_json, importance_score "
                    "FROM entities "
                    "WHERE owner_tenant_id = :tid "
                    "  AND status IN ('active', 'confirmed', 'provisional') "
                    "  AND merged_into_id IS NULL"
                ),
                {"tid": tid},
            ).fetchall()

            # Build name -> entity_id index
            name_index = {}
            for ent in all_entities:
                eid = ent[0]
                cname = ent[1].lower()
                etype = ent[2]
                aliases = ent[3] if isinstance(ent[3], list) else []
                importance = ent[4]

                entry = {
                    "entity_id": eid,
                    "type": etype,
                    "importance": importance,
                }
                name_index[cname] = entry
                for alias in aliases:
                    if alias and isinstance(alias, str):
                        name_index[alias.lower()] = entry

            if name_index:
                anchor_key = f"gami:anchors:{tid}"
                r.setex(
                    anchor_key,
                    CACHE_TTL,
                    json.dumps(name_index, default=str),
                )
                total_anchors += 1

            # Cache tenant stats
            stats = _compute_tenant_stats(db, tid)
            stats_key = f"gami:stats:{tid}"
            r.setex(stats_key, CACHE_TTL, json.dumps(stats, default=str))

            logger.info(
                "Cache warming for tenant %s: %d neighborhoods, anchors cached",
                tid, len(top_entities),
            )

        r.close()

        return {
            "status": "completed",
            "neighborhoods_cached": total_cached,
            "anchor_sets_cached": total_anchors,
            "tenants_processed": len(tenant_ids),
        }

    except Exception as exc:
        logger.error("Cache warming failed: %s", exc, exc_info=True)
        r.close()
        raise self.retry(exc=exc, countdown=120)

    finally:
        try:
            next(db_gen)
        except StopIteration:
            pass


def _compute_neighborhood(db, entity_id: str, tenant_id: str) -> list[dict]:
    """Compute 2-hop neighborhood for an entity."""
    neighbors = []
    visited = {entity_id}

    # Hop 1: Direct relations
    hop1_rows = db.execute(
        text(
            "SELECT r.relation_id, r.relation_type, r.confidence, r.weight, "
            "       CASE WHEN r.from_node_id = :eid THEN r.to_node_id "
            "            ELSE r.from_node_id END AS neighbor_id, "
            "       CASE WHEN r.from_node_id = :eid THEN 'outgoing' "
            "            ELSE 'incoming' END AS direction "
            "FROM relations r "
            "WHERE r.owner_tenant_id = :tid "
            "  AND r.status = 'active' "
            "  AND (r.from_node_id = :eid OR r.to_node_id = :eid) "
            "ORDER BY r.weight DESC "
            "LIMIT :lim"
        ),
        {"eid": entity_id, "tid": tenant_id, "lim": MAX_NEIGHBORS},
    ).fetchall()

    hop1_ids = set()
    for row in hop1_rows:
        nid = row[4]
        if nid in visited:
            continue
        visited.add(nid)
        hop1_ids.add(nid)

        neighbors.append({
            "entity_id": nid,
            "relation_type": row[1],
            "confidence": row[2],
            "weight": row[3],
            "direction": row[5],
            "hop": 1,
        })

    # Hop 2: Relations of hop-1 neighbors
    if hop1_ids:
        hop1_list = list(hop1_ids)
        hop2_rows = db.execute(
            text(
                "SELECT r.relation_type, r.confidence, "
                "       CASE WHEN r.from_node_id = ANY(:ids) THEN r.from_node_id "
                "            ELSE r.to_node_id END AS via_id, "
                "       CASE WHEN r.from_node_id = ANY(:ids) THEN r.to_node_id "
                "            ELSE r.from_node_id END AS neighbor_id "
                "FROM relations r "
                "WHERE r.owner_tenant_id = :tid "
                "  AND r.status = 'active' "
                "  AND (r.from_node_id = ANY(:ids) OR r.to_node_id = ANY(:ids)) "
                "  AND r.from_node_id != :eid AND r.to_node_id != :eid "
                "LIMIT :lim"
            ),
            {
                "ids": hop1_list,
                "tid": tenant_id,
                "eid": entity_id,
                "lim": MAX_NEIGHBORS,
            },
        ).fetchall()

        for row in hop2_rows:
            nid = row[3]
            if nid in visited:
                continue
            visited.add(nid)
            neighbors.append({
                "entity_id": nid,
                "relation_type": row[0],
                "confidence": row[1],
                "via": row[2],
                "hop": 2,
            })

    # Enrich with entity names
    if neighbors:
        nids = [n["entity_id"] for n in neighbors]
        # Fetch in chunks to avoid parameter limits
        name_map = {}
        for i in range(0, len(nids), 100):
            chunk = nids[i : i + 100]
            name_rows = db.execute(
                text(
                    "SELECT entity_id, canonical_name, entity_type "
                    "FROM entities WHERE entity_id = ANY(:ids)"
                ),
                {"ids": chunk},
            ).fetchall()
            for nr in name_rows:
                name_map[nr[0]] = {"name": nr[1], "type": nr[2]}

        for n in neighbors:
            info = name_map.get(n["entity_id"], {})
            n["name"] = info.get("name", "unknown")
            n["type"] = info.get("type", "unknown")

    return neighbors


def _compute_tenant_stats(db, tenant_id: str) -> dict:
    """Compute quick stats for a tenant."""
    row = db.execute(
        text(
            "SELECT "
            "  (SELECT COUNT(*) FROM entities WHERE owner_tenant_id = :tid AND status != 'archived'), "
            "  (SELECT COUNT(*) FROM claims WHERE owner_tenant_id = :tid AND status = 'active'), "
            "  (SELECT COUNT(*) FROM relations WHERE owner_tenant_id = :tid AND status = 'active'), "
            "  (SELECT COUNT(*) FROM segments WHERE owner_tenant_id = :tid), "
            "  (SELECT COUNT(*) FROM sources WHERE owner_tenant_id = :tid), "
            "  (SELECT COUNT(*) FROM events WHERE owner_tenant_id = :tid AND status = 'active')"
        ),
        {"tid": tenant_id},
    ).fetchone()

    return {
        "tenant_id": tenant_id,
        "entities": row[0] if row else 0,
        "claims": row[1] if row else 0,
        "relations": row[2] if row else 0,
        "segments": row[3] if row else 0,
        "sources": row[4] if row else 0,
        "events": row[5] if row else 0,
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }


# Beat schedule is centralized in workers/celery_app.py
