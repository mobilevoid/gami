"""Basic topic clustering for GAMI.

Groups sources by tenant + source_type for now.
Future: embedding-based clustering using pgvector similarity.
"""
import logging
from datetime import datetime, timezone

from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger("gami.services.clustering")


def cluster_sources_by_type(db: Session) -> list[dict]:
    """Group sources by (tenant, source_type). Returns cluster info."""
    rows = db.execute(
        text(
            "SELECT s.owner_tenant_id, s.source_type, "
            "COUNT(*) as source_count, "
            "SUM(seg.seg_count) as total_segments "
            "FROM sources s "
            "LEFT JOIN LATERAL ("
            "  SELECT COUNT(*) as seg_count FROM segments "
            "  WHERE segments.source_id = s.source_id"
            ") seg ON true "
            "GROUP BY s.owner_tenant_id, s.source_type "
            "ORDER BY source_count DESC"
        )
    ).fetchall()

    clusters = []
    for row in rows:
        clusters.append({
            "tenant_id": row[0],
            "source_type": row[1],
            "source_count": row[2],
            "total_segments": row[3] or 0,
        })

    logger.info("Found %d source clusters", len(clusters))
    return clusters


def get_sources_in_cluster(
    db: Session,
    tenant_id: str,
    source_type: str,
    limit: int = 100,
) -> list[dict]:
    """Get all sources in a cluster (tenant + source_type)."""
    rows = db.execute(
        text(
            "SELECT source_id, title, source_type "
            "FROM sources "
            "WHERE owner_tenant_id = :tid AND source_type = :stype "
            "ORDER BY created_at DESC "
            "LIMIT :lim"
        ),
        {"tid": tenant_id, "stype": source_type, "lim": limit},
    ).fetchall()

    return [{"source_id": r[0], "title": r[1], "source_type": r[2]} for r in rows]
