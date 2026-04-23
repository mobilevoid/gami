"""Contradiction-aware retrieval service.

Checks for unresolved contradictions in retrieved claims and surfaces
them in the recall response so users are aware of conflicting information.
"""
import logging
from dataclasses import dataclass, field
from typing import List, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger("gami.services.contradiction")


@dataclass
class ContradictionInfo:
    """Information about a contradiction involving returned results."""
    contradiction_group_id: str
    claim_ids: List[str]
    predicate: str
    conflicting_values: List[str]
    confidence_scores: List[float]
    status: str  # 'unresolved', 'proposed', 'resolved'
    created_dates: List[str] = field(default_factory=list)
    proposal_id: Optional[str] = None
    suggested_resolution: Optional[str] = None


class ContradictionChecker:
    """Check for contradictions involving retrieved claims."""

    async def check_for_contradictions(
        self,
        db: AsyncSession,
        claim_ids: List[str],
        segment_ids: List[str],
        tenant_ids: List[str],
    ) -> List[ContradictionInfo]:
        """Check if any returned claims/segments have unresolved contradictions.

        Args:
            db: Database session
            claim_ids: IDs of claims being returned in recall
            segment_ids: IDs of segments being returned
            tenant_ids: Tenant scope

        Returns:
            List of ContradictionInfo for any contradictions found
        """
        if not claim_ids and not segment_ids:
            return []

        contradictions = []

        # Check claims with contradiction groups
        if claim_ids:
            claim_contradictions = await self._check_claim_contradictions(
                db, claim_ids, tenant_ids
            )
            contradictions.extend(claim_contradictions)

        # Check segments that source contradicting claims
        if segment_ids:
            segment_contradictions = await self._check_segment_contradictions(
                db, segment_ids, tenant_ids
            )
            contradictions.extend(segment_contradictions)

        return contradictions

    async def _check_claim_contradictions(
        self,
        db: AsyncSession,
        claim_ids: List[str],
        tenant_ids: List[str],
    ) -> List[ContradictionInfo]:
        """Find claims in contradiction groups."""
        result = await db.execute(text("""
            WITH returned_claims AS (
                SELECT claim_id, contradiction_group_id
                FROM claims
                WHERE claim_id = ANY(:cids)
                AND contradiction_group_id IS NOT NULL
            )
            SELECT
                c.contradiction_group_id,
                array_agg(DISTINCT c.claim_id) as claim_ids,
                c.predicate,
                array_agg(COALESCE(c.summary_text, 'N/A') ORDER BY c.created_at) as values,
                array_agg(COALESCE(c.confidence, 0.5) ORDER BY c.created_at) as confidences,
                array_agg(c.created_at::text ORDER BY c.created_at) as dates,
                COALESCE(
                    (SELECT status FROM proposed_changes pc
                     WHERE pc.target_type = 'claim'
                     AND pc.change_type = 'contradiction_resolution'
                     AND (pc.proposed_state_json->>'contradiction_group_id') = c.contradiction_group_id
                     ORDER BY pc.created_at DESC LIMIT 1),
                    'unresolved'
                ) as resolution_status,
                (SELECT proposal_id FROM proposed_changes pc
                 WHERE pc.target_type = 'claim'
                 AND pc.change_type = 'contradiction_resolution'
                 AND (pc.proposed_state_json->>'contradiction_group_id') = c.contradiction_group_id
                 AND pc.status = 'pending'
                 ORDER BY pc.created_at DESC LIMIT 1) as proposal_id
            FROM claims c
            WHERE c.contradiction_group_id IN (
                SELECT contradiction_group_id FROM returned_claims
            )
            AND c.status = 'active'
            AND c.owner_tenant_id = ANY(:tids)
            GROUP BY c.contradiction_group_id, c.predicate
        """), {"cids": claim_ids, "tids": tenant_ids})

        contradictions = []
        for row in result.fetchall():
            contradictions.append(ContradictionInfo(
                contradiction_group_id=row.contradiction_group_id,
                claim_ids=list(row.claim_ids) if row.claim_ids else [],
                predicate=row.predicate or "unknown",
                conflicting_values=list(row.values) if row.values else [],
                confidence_scores=[float(c) if c else 0.5 for c in (row.confidences or [])],
                created_dates=list(row.dates) if row.dates else [],
                status=row.resolution_status or "unresolved",
                proposal_id=row.proposal_id,
            ))

        return contradictions

    async def _check_segment_contradictions(
        self,
        db: AsyncSession,
        segment_ids: List[str],
        tenant_ids: List[str],
    ) -> List[ContradictionInfo]:
        """Find segments that are sources for contradicting claims."""
        # Check if any segments are provenance sources for contradicting claims
        result = await db.execute(text("""
            WITH segment_claims AS (
                SELECT DISTINCT p.segment_id, c.claim_id, c.contradiction_group_id
                FROM provenance p
                JOIN claims c ON p.target_id = c.claim_id AND p.target_type = 'claim'
                WHERE p.segment_id = ANY(:sids)
                AND c.contradiction_group_id IS NOT NULL
                AND c.status = 'active'
            )
            SELECT
                c.contradiction_group_id,
                array_agg(DISTINCT c.claim_id) as claim_ids,
                c.predicate,
                array_agg(COALESCE(c.summary_text, 'N/A') ORDER BY c.created_at) as values,
                array_agg(COALESCE(c.confidence, 0.5) ORDER BY c.created_at) as confidences,
                'unresolved' as resolution_status
            FROM claims c
            WHERE c.contradiction_group_id IN (
                SELECT contradiction_group_id FROM segment_claims
            )
            AND c.status = 'active'
            AND c.owner_tenant_id = ANY(:tids)
            GROUP BY c.contradiction_group_id, c.predicate
        """), {"sids": segment_ids, "tids": tenant_ids})

        contradictions = []
        for row in result.fetchall():
            contradictions.append(ContradictionInfo(
                contradiction_group_id=row.contradiction_group_id,
                claim_ids=list(row.claim_ids) if row.claim_ids else [],
                predicate=row.predicate or "unknown",
                conflicting_values=list(row.values) if row.values else [],
                confidence_scores=[float(c) if c else 0.5 for c in (row.confidences or [])],
                status=row.resolution_status,
            ))

        return contradictions

    def format_contradiction_warning(
        self,
        contradictions: List[ContradictionInfo],
    ) -> str:
        """Format contradictions into a warning string for context."""
        if not contradictions:
            return ""

        parts = ["\n---\n**Warning: Conflicting Information Detected**\n"]
        for i, c in enumerate(contradictions, 1):
            parts.append(f"\n{i}. Contradiction on '{c.predicate}':")
            for j, (val, conf) in enumerate(zip(c.conflicting_values, c.confidence_scores)):
                val_preview = val[:150] + "..." if len(val) > 150 else val
                parts.append(f"   - {val_preview} (confidence: {conf:.0%})")
            if c.status == 'proposed':
                parts.append(f"   - *Resolution pending review*")
            elif c.status == 'unresolved':
                parts.append(f"   - *Needs resolution*")

        parts.append("\n---\n")
        return "\n".join(parts)

    def to_response_dict(
        self,
        contradictions: List[ContradictionInfo],
    ) -> List[dict]:
        """Convert contradictions to response-friendly dicts."""
        return [
            {
                "group_id": c.contradiction_group_id,
                "predicate": c.predicate,
                "claim_ids": c.claim_ids,
                "values": c.conflicting_values,
                "confidences": c.confidence_scores,
                "status": c.status,
                "proposal_id": c.proposal_id,
            }
            for c in contradictions
        ]
