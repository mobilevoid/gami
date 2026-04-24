"""Contradiction Detection Worker — finds conflicting claims.

Compares claims with overlapping subjects. When subject + predicate match
but objects differ, creates a contradiction_group and stores a proposal
in proposed_changes for review.
"""
import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone

from sqlalchemy import text

from workers.celery_app import celery_app

logger = logging.getLogger("gami.workers.contradiction")


def _make_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


@celery_app.task(
    name="gami.detect_contradictions",
    bind=True,
    max_retries=1,
    soft_time_limit=600,
    time_limit=660,
)
def detect_contradictions(self, tenant_id: str = None):
    """Detect contradictory claims within each tenant.

    A contradiction is when two active claims share the same
    subject_entity_id + predicate but have different objects.

    Steps:
    1. Find claim pairs with matching subject+predicate, differing objects
    2. Skip pairs already in a contradiction_group
    3. Create contradiction group IDs
    4. Store proposals in proposed_changes for human review
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

        total_contradictions = 0
        total_proposals = 0

        for tid in tenant_ids:
            # Find claim pairs with same subject+predicate but different objects
            # We compare both entity-linked objects and literal objects
            pairs = db.execute(
                text(
                    "SELECT a.claim_id, a.predicate, a.object_entity_id, "
                    "       a.object_literal_json, a.summary_text, a.confidence, "
                    "       b.claim_id, b.object_entity_id, "
                    "       b.object_literal_json, b.summary_text, b.confidence "
                    "FROM claims a "
                    "JOIN claims b ON a.subject_entity_id = b.subject_entity_id "
                    "  AND a.predicate = b.predicate "
                    "  AND a.claim_id < b.claim_id "
                    "  AND a.owner_tenant_id = b.owner_tenant_id "
                    "WHERE a.owner_tenant_id = :tid "
                    "  AND a.status = 'active' AND b.status = 'active' "
                    "  AND a.subject_entity_id IS NOT NULL "
                    "  AND a.contradiction_group_id IS NULL "
                    "  AND b.contradiction_group_id IS NULL "
                    "  AND ("
                    "    (a.object_entity_id IS NOT NULL "
                    "     AND b.object_entity_id IS NOT NULL "
                    "     AND a.object_entity_id != b.object_entity_id) "
                    "    OR "
                    "    (a.object_literal_json IS NOT NULL "
                    "     AND b.object_literal_json IS NOT NULL "
                    "     AND a.object_literal_json != b.object_literal_json) "
                    "  ) "
                    "LIMIT 200"
                ),
                {"tid": tid},
            ).fetchall()

            for row in pairs:
                claim_a_id = row[0]
                predicate = row[1]
                claim_a_obj_eid = row[2]
                claim_a_obj_lit = row[3]
                claim_a_summary = row[4]
                claim_a_conf = row[5]
                claim_b_id = row[6]
                claim_b_obj_eid = row[7]
                claim_b_obj_lit = row[8]
                claim_b_summary = row[9]
                claim_b_conf = row[10]

                # Create contradiction group
                group_id = _make_id("CGRP")
                now = datetime.now(timezone.utc)

                # Tag both claims with the contradiction group
                db.execute(
                    text(
                        "UPDATE claims SET contradiction_group_id = :gid, "
                        "updated_at = :now "
                        "WHERE claim_id IN (:a, :b)"
                    ),
                    {"gid": group_id, "now": now, "a": claim_a_id, "b": claim_b_id},
                )
                total_contradictions += 1

                # Create a proposal for resolution
                proposal_id = _make_id("PROP")

                # Describe the contradiction
                obj_a_desc = claim_a_summary or str(claim_a_obj_lit or claim_a_obj_eid)
                obj_b_desc = claim_b_summary or str(claim_b_obj_lit or claim_b_obj_eid)

                reason = (
                    f"Contradicting claims on predicate '{predicate}': "
                    f"Claim A ({claim_a_conf:.2f}): {obj_a_desc[:200]} vs "
                    f"Claim B ({claim_b_conf:.2f}): {obj_b_desc[:200]}"
                )

                # Suggest keeping the higher-confidence claim
                if claim_a_conf >= claim_b_conf:
                    suggested_keep = claim_a_id
                    suggested_supersede = claim_b_id
                else:
                    suggested_keep = claim_b_id
                    suggested_supersede = claim_a_id

                db.execute(
                    text(
                        "INSERT INTO proposed_changes "
                        "(proposal_id, proposer_tenant_id, change_type, "
                        "target_type, target_id, proposed_state_json, "
                        "reason, confidence, evidence_ids, status, created_at) "
                        "VALUES (:pid, 'background-worker', 'contradiction_resolution', "
                        "'claim', :target, CAST(:state AS jsonb), "
                        ":reason, :conf, CAST(:evidence AS jsonb), "
                        "'pending', :now)"
                    ),
                    {
                        "pid": proposal_id,
                        "target": suggested_keep,
                        "state": json.dumps({
                            "contradiction_group_id": group_id,
                            "claim_a": claim_a_id,
                            "claim_b": claim_b_id,
                            "predicate": predicate,
                            "suggested_keep": suggested_keep,
                            "suggested_supersede": suggested_supersede,
                            "action": "supersede_lower_confidence",
                        }),
                        "reason": reason[:2000],
                        "conf": max(claim_a_conf, claim_b_conf),
                        "evidence": json.dumps([claim_a_id, claim_b_id]),
                        "now": now,
                    },
                )
                total_proposals += 1

            db.commit()
            logger.info(
                "Contradiction detection for tenant %s: %d contradictions found",
                tid, total_contradictions,
            )

        return {
            "status": "completed",
            "contradictions_found": total_contradictions,
            "proposals_created": total_proposals,
            "tenants_processed": len(tenant_ids),
        }

    except Exception as exc:
        logger.error("Contradiction detection failed: %s", exc, exc_info=True)
        db.rollback()
        raise self.retry(exc=exc, countdown=300)

    finally:
        try:
            next(db_gen)
        except StopIteration:
            pass


# Beat schedule is centralized in workers/celery_app.py
