"""Admin API for GAMI.

Provides database statistics, contradiction listing, and stale memory detection.
"""
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import text

from api.services.db import AsyncSessionLocal

logger = logging.getLogger("gami.routers.admin")

router = APIRouter()


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class DBStats(BaseModel):
    segments_total: int = 0
    segments_embedded: int = 0
    segments_unembedded: int = 0
    entities_total: int = 0
    entities_active: int = 0
    claims_total: int = 0
    claims_active: int = 0
    relations_total: int = 0
    summaries_total: int = 0
    memories_total: int = 0
    memories_active: int = 0
    memories_provisional: int = 0
    memories_archived: int = 0
    sources_total: int = 0
    tenants: list[str] = []
    provenance_total: int = 0


class ContradictionEntry(BaseModel):
    claim_id: str
    summary_text: Optional[str] = None
    contradiction_group_id: Optional[str] = None
    confidence: float = 0.0
    status: str = ""
    created_at: Optional[str] = None


class StaleMemory(BaseModel):
    memory_id: str
    memory_type: str
    subject: str
    text: str
    importance: float
    status: str
    last_confirmed_at: Optional[str] = None
    confirmation_count: int = 0
    created_at: Optional[str] = None
    age_days: float = 0.0


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/stats", response_model=DBStats)
async def get_stats():
    """Get database statistics across all tables."""
    try:
        async with AsyncSessionLocal() as db:
            stats = DBStats()

            # Segments
            r = await db.execute(text("SELECT count(*) FROM segments"))
            stats.segments_total = r.scalar()

            r = await db.execute(text("SELECT count(*) FROM segments WHERE embedding IS NOT NULL"))
            stats.segments_embedded = r.scalar()
            stats.segments_unembedded = stats.segments_total - stats.segments_embedded

            # Entities
            r = await db.execute(text("SELECT count(*) FROM entities"))
            stats.entities_total = r.scalar()

            r = await db.execute(text("SELECT count(*) FROM entities WHERE status = 'active'"))
            stats.entities_active = r.scalar()

            # Claims
            r = await db.execute(text("SELECT count(*) FROM claims"))
            stats.claims_total = r.scalar()

            r = await db.execute(text("SELECT count(*) FROM claims WHERE status = 'active'"))
            stats.claims_active = r.scalar()

            # Relations
            r = await db.execute(text("SELECT count(*) FROM relations"))
            stats.relations_total = r.scalar()

            # Summaries
            r = await db.execute(text("SELECT count(*) FROM summaries"))
            stats.summaries_total = r.scalar()

            # Memories
            r = await db.execute(text("SELECT count(*) FROM assistant_memories"))
            stats.memories_total = r.scalar()

            r = await db.execute(text("SELECT count(*) FROM assistant_memories WHERE status = 'active'"))
            stats.memories_active = r.scalar()

            r = await db.execute(text("SELECT count(*) FROM assistant_memories WHERE status = 'provisional'"))
            stats.memories_provisional = r.scalar()

            r = await db.execute(text("SELECT count(*) FROM assistant_memories WHERE status = 'archived'"))
            stats.memories_archived = r.scalar()

            # Sources
            r = await db.execute(text("SELECT count(*) FROM sources"))
            stats.sources_total = r.scalar()

            # Tenants
            r = await db.execute(text("SELECT tenant_id FROM tenants ORDER BY tenant_id"))
            stats.tenants = [row[0] for row in r.fetchall()]

            # Provenance
            r = await db.execute(text("SELECT count(*) FROM provenance"))
            stats.provenance_total = r.scalar()

            return stats

    except Exception as exc:
        logger.error("Stats query failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Stats failed: {exc}")


@router.get("/contradictions")
async def list_contradictions(
    tenant_id: Optional[str] = None,
    limit: int = 50,
):
    """List claims that have been flagged as contradictions."""
    try:
        async with AsyncSessionLocal() as db:
            where_parts = ["contradiction_group_id IS NOT NULL"]
            params = {"lim": limit}

            if tenant_id:
                where_parts.append("owner_tenant_id = :tid")
                params["tid"] = tenant_id

            where = " AND ".join(where_parts)

            r = await db.execute(
                text(f"""
                    SELECT claim_id, summary_text, contradiction_group_id,
                           confidence, status, created_at
                    FROM claims
                    WHERE {where}
                    ORDER BY created_at DESC
                    LIMIT :lim
                """),
                params,
            )
            rows = r.fetchall()

            return {
                "contradictions": [
                    ContradictionEntry(
                        claim_id=row[0],
                        summary_text=row[1],
                        contradiction_group_id=row[2],
                        confidence=float(row[3]),
                        status=row[4],
                        created_at=row[5].isoformat() if row[5] else None,
                    ).model_dump()
                    for row in rows
                ],
                "total": len(rows),
            }

    except Exception as exc:
        logger.error("Contradictions query failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}")


@router.get("/stale")
async def list_stale_memories(
    tenant_id: Optional[str] = None,
    stale_days: int = 30,
    limit: int = 50,
):
    """List memories that haven't been confirmed recently and may be stale.

    A memory is considered stale if:
    - It's active/provisional and hasn't been confirmed in stale_days
    - It has low confirmation_count (< 3)
    - It was created more than stale_days ago
    """
    try:
        async with AsyncSessionLocal() as db:
            cutoff = datetime.now(timezone.utc) - timedelta(days=stale_days)
            params = {"cutoff": cutoff, "lim": limit}

            where_parts = [
                "status IN ('active', 'provisional')",
                "created_at < :cutoff",
                "confirmation_count < 3",
                "(last_confirmed_at IS NULL OR last_confirmed_at < :cutoff)",
            ]

            if tenant_id:
                where_parts.append("owner_tenant_id = :tid")
                params["tid"] = tenant_id

            where = " AND ".join(where_parts)
            now = datetime.now(timezone.utc)

            r = await db.execute(
                text(f"""
                    SELECT memory_id, memory_type, subject_id,
                           normalized_text, importance_score, status,
                           last_confirmed_at, confirmation_count, created_at
                    FROM assistant_memories
                    WHERE {where}
                    ORDER BY importance_score ASC, created_at ASC
                    LIMIT :lim
                """),
                params,
            )
            rows = r.fetchall()

            return {
                "stale_memories": [
                    StaleMemory(
                        memory_id=row[0],
                        memory_type=row[1],
                        subject=row[2],
                        text=row[3],
                        importance=float(row[4]),
                        status=row[5],
                        last_confirmed_at=row[6].isoformat() if row[6] else None,
                        confirmation_count=row[7],
                        created_at=row[8].isoformat() if row[8] else None,
                        age_days=round((now - row[8]).total_seconds() / 86400, 1) if row[8] else 0,
                    ).model_dump()
                    for row in rows
                ],
                "total": len(rows),
                "stale_days_threshold": stale_days,
            }

    except Exception as exc:
        logger.error("Stale memories query failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}")
