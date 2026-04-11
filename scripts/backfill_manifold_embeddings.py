#!/usr/bin/env python3
"""Backfill multi-manifold embeddings for segments.

Populates the manifold_embeddings table with embeddings for all 6 manifolds:
- TOPIC: Direct text embedding
- CLAIM: Extracted claims embedding
- PROCEDURE: Extracted steps embedding
- RELATION: Graph fingerprint projection (placeholder)
- TIME: Temporal features projection
- EVIDENCE: Evidence factors projection (placeholder)

Usage:
    python backfill_manifold_embeddings.py [--batch-size 50] [--limit 1000]
"""
import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from api.config import settings
from api.llm.embeddings import embed_texts_batch
from manifold.embeddings.multi_embedder import (
    MultiManifoldEmbedder,
    ManifoldType,
    ManifoldEmbeddings,
)
from manifold.embeddings.projectors import (
    compute_evidence_factors,
    get_time_projector,
    get_evidence_projector,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Database connection
DATABASE_URL = settings.DATABASE_URL_SYNC


def get_segments_needing_backfill(engine, limit: int = 100) -> list:
    """Get segments that need manifold embeddings."""
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT s.segment_id, s.text, s.owner_tenant_id
            FROM segments s
            WHERE s.embedding IS NOT NULL
            AND s.storage_tier != 'cold'
            AND LENGTH(s.text) > 50
            AND s.segment_id NOT IN (
                SELECT DISTINCT target_id
                FROM manifold_embeddings
                WHERE manifold_type = 'TOPIC'
            )
            ORDER BY s.created_at DESC
            LIMIT :lim
        """), {"lim": limit})
        return result.fetchall()


def get_segment_evidence_factors(engine, segment_id: str) -> dict:
    """Get evidence factors for a segment (placeholder)."""
    # In production, would query the evidence scoring system
    return {
        "authority_score": 0.5,
        "corroboration_count": 0,
        "recency_days": 30.0,
        "specificity_score": 0.5,
        "contradiction_ratio": 0.0,
    }


def get_segment_graph_fingerprint(conn, segment_id: str) -> np.ndarray:
    """Compute graph fingerprint for a segment from provenance data.

    Returns a 64-dimensional fingerprint encoding:
    - Entity types connected to this segment
    - Number of entities mentioned
    - Relation types involving these entities
    - Centrality approximation
    """
    from manifold.embeddings.projectors import compute_graph_fingerprint

    # Get entities mentioned in this segment
    result = conn.execute(text("""
        SELECT p.target_id, e.entity_type
        FROM provenance p
        LEFT JOIN entities e ON p.target_id = e.entity_id
        WHERE p.segment_id = :sid
        AND p.target_type = 'entity'
    """), {"sid": segment_id})
    entity_rows = result.fetchall()

    if not entity_rows:
        return None

    entity_ids = [r[0] for r in entity_rows]
    entity_types = [r[1] or "unknown" for r in entity_rows if r[1]]

    # Get relations involving these entities
    result = conn.execute(text("""
        SELECT relation_type,
               CASE WHEN source_entity_id = ANY(:eids) THEN 'out' ELSE 'in' END as direction
        FROM relations
        WHERE source_entity_id = ANY(:eids) OR target_entity_id = ANY(:eids)
        LIMIT 100
    """), {"eids": entity_ids})
    relation_rows = result.fetchall()

    # Build edge counts
    in_edges = {}
    out_edges = {}
    for rel_type, direction in relation_rows:
        rel_type = rel_type or "relates_to"
        if direction == "in":
            in_edges[rel_type] = in_edges.get(rel_type, 0) + 1
        else:
            out_edges[rel_type] = out_edges.get(rel_type, 0) + 1

    # Approximate centrality based on connection count
    total_connections = len(relation_rows)
    centrality = min(1.0, total_connections / 50.0)

    return compute_graph_fingerprint(
        in_edges=in_edges,
        out_edges=out_edges,
        connected_types=list(set(entity_types)),
        centrality=centrality,
    )


def save_manifold_embedding(
    conn,
    target_id: str,
    target_type: str,
    manifold_type: str,
    embedding: np.ndarray,
    canonical_form: str = None,
):
    """Save a manifold embedding to the database."""
    emb_str = "[" + ",".join(str(x) for x in embedding.flatten()) + "]"

    # Use direct SQL with string formatting for the vector cast
    # The vector needs to be cast directly, not as a parameter
    sql = f"""
        INSERT INTO manifold_embeddings
        (target_id, target_type, manifold_type, embedding, canonical_form, updated_at)
        VALUES (:tid, :ttype, :mtype, '{emb_str}'::vector, :canonical, NOW())
        ON CONFLICT (target_id, target_type, manifold_type)
        DO UPDATE SET
            embedding = EXCLUDED.embedding,
            canonical_form = EXCLUDED.canonical_form,
            updated_at = NOW()
    """
    conn.execute(text(sql), {
        "tid": target_id,
        "ttype": target_type,
        "mtype": manifold_type,
        "canonical": canonical_form,
    })


def process_batch(engine, segments: list, embedder: MultiManifoldEmbedder) -> int:
    """Process a batch of segments."""
    processed = 0

    # Collect texts for batch embedding
    texts_topic = []
    texts_claim = []
    texts_procedure = []
    segment_data = []

    for seg in segments:
        segment_id, text, tenant_id = seg
        if not text or len(text) < 10:
            continue

        texts_topic.append(text)

        # Extract claims
        claims = embedder._extract_claims(text)
        claim_text = " | ".join(claims) if claims else text[:500]
        texts_claim.append(claim_text)

        # Extract procedures
        steps = embedder._extract_procedure_steps(text)
        proc_text = " → ".join(steps) if steps else ""
        texts_procedure.append(proc_text if proc_text else text[:500])

        segment_data.append({
            "segment_id": segment_id,
            "text": text,
            "claim_text": claim_text,
            "proc_text": proc_text if proc_text else None,
        })

    if not texts_topic:
        return 0

    # Batch embed
    logger.info(f"Embedding {len(texts_topic)} segments...")

    try:
        emb_topic = embed_texts_batch(texts_topic)
        emb_claim = embed_texts_batch(texts_claim)
        emb_procedure = embed_texts_batch(texts_procedure)
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        return 0

    # Save to database
    time_projector = get_time_projector()
    evidence_projector = get_evidence_projector()

    with engine.begin() as conn:
        for i, seg in enumerate(segment_data):
            segment_id = seg["segment_id"]

            # TOPIC
            save_manifold_embedding(
                conn, segment_id, "segment", "TOPIC",
                np.array(emb_topic[i]), seg["text"][:500]
            )

            # CLAIM
            save_manifold_embedding(
                conn, segment_id, "segment", "CLAIM",
                np.array(emb_claim[i]), seg["claim_text"][:500]
            )

            # PROCEDURE (only if we extracted steps)
            if seg["proc_text"]:
                save_manifold_embedding(
                    conn, segment_id, "segment", "PROCEDURE",
                    np.array(emb_procedure[i]), seg["proc_text"][:500]
                )

            # TIME
            time_features = embedder._extract_temporal_features(seg["text"])
            if time_features is not None:
                time_emb = time_projector.project(time_features)
                save_manifold_embedding(
                    conn, segment_id, "segment", "TIME",
                    time_emb, None
                )

            # EVIDENCE
            evidence_factors = compute_evidence_factors(
                authority_score=0.5,
                corroboration_count=0,
                recency_days=30.0,
                specificity_score=0.5,
                contradiction_ratio=0.0,
            )
            evidence_emb = evidence_projector.project(evidence_factors)
            save_manifold_embedding(
                conn, segment_id, "segment", "EVIDENCE",
                evidence_emb, None
            )

            # RELATION (graph fingerprint)
            try:
                from manifold.embeddings.projectors import get_relation_projector
                relation_projector = get_relation_projector()

                graph_fingerprint = get_segment_graph_fingerprint(conn, segment_id)
                if graph_fingerprint is not None:
                    relation_emb = relation_projector.project(graph_fingerprint)
                    save_manifold_embedding(
                        conn, segment_id, "segment", "RELATION",
                        relation_emb, None
                    )
            except Exception as e:
                logger.debug(f"RELATION embedding skipped for {segment_id}: {e}")

            processed += 1

    return processed


def get_entities_needing_backfill(engine, limit: int = 100) -> list:
    """Get entities that need manifold embeddings."""
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT e.entity_id, e.canonical_name, e.entity_type, e.owner_tenant_id
            FROM entities e
            WHERE e.embedding IS NOT NULL
            AND e.status = 'active'
            AND e.entity_id NOT IN (
                SELECT DISTINCT target_id
                FROM manifold_embeddings
                WHERE target_type = 'entity' AND manifold_type = 'TOPIC'
            )
            ORDER BY e.created_at DESC
            LIMIT :lim
        """), {"lim": limit})
        return result.fetchall()


def process_entity_batch(engine, entities: list, embedder: MultiManifoldEmbedder) -> int:
    """Process a batch of entities for manifold embeddings."""
    from manifold.embeddings.projectors import get_relation_projector, compute_graph_fingerprint

    processed = 0
    texts = []
    entity_data = []

    for ent in entities:
        entity_id, canonical_name, entity_type, tenant_id = ent
        if not canonical_name:
            continue

        # Use canonical name as the text to embed
        texts.append(canonical_name)
        entity_data.append({
            "entity_id": entity_id,
            "canonical_name": canonical_name,
            "entity_type": entity_type or "unknown",
        })

    if not texts:
        return 0

    logger.info(f"Embedding {len(texts)} entities...")

    try:
        embeddings = embed_texts_batch(texts)
    except Exception as e:
        logger.error(f"Entity embedding failed: {e}")
        return 0

    relation_projector = get_relation_projector()

    with engine.begin() as conn:
        for i, ent in enumerate(entity_data):
            entity_id = ent["entity_id"]

            # TOPIC: Direct embedding of canonical name
            save_manifold_embedding(
                conn, entity_id, "entity", "TOPIC",
                np.array(embeddings[i]), ent["canonical_name"][:500]
            )

            # RELATION: Graph fingerprint based on entity's connections
            try:
                # Get relations for this entity
                rel_result = conn.execute(text("""
                    SELECT relation_type,
                        CASE WHEN source_entity_id = :eid THEN 'out' ELSE 'in' END as direction,
                        CASE WHEN source_entity_id = :eid THEN target_entity_id ELSE source_entity_id END as other_id
                    FROM relations
                    WHERE source_entity_id = :eid OR target_entity_id = :eid
                    LIMIT 100
                """), {"eid": entity_id})
                rel_rows = rel_result.fetchall()

                if rel_rows:
                    in_edges = {}
                    out_edges = {}
                    other_ids = []

                    for rel_type, direction, other_id in rel_rows:
                        rel_type = rel_type or "relates_to"
                        other_ids.append(other_id)
                        if direction == "in":
                            in_edges[rel_type] = in_edges.get(rel_type, 0) + 1
                        else:
                            out_edges[rel_type] = out_edges.get(rel_type, 0) + 1

                    # Get entity types of connected entities
                    type_result = conn.execute(text("""
                        SELECT DISTINCT entity_type FROM entities WHERE entity_id = ANY(:ids)
                    """), {"ids": other_ids[:20]})
                    connected_types = [r[0] for r in type_result.fetchall() if r[0]]

                    centrality = min(1.0, len(rel_rows) / 50.0)
                    fingerprint = compute_graph_fingerprint(
                        in_edges=in_edges,
                        out_edges=out_edges,
                        connected_types=connected_types + [ent["entity_type"]],
                        centrality=centrality,
                    )
                    relation_emb = relation_projector.project(fingerprint)
                    save_manifold_embedding(
                        conn, entity_id, "entity", "RELATION",
                        relation_emb, None
                    )
            except Exception as e:
                logger.debug(f"RELATION embedding skipped for entity {entity_id}: {e}")

            processed += 1

    return processed


def main():
    parser = argparse.ArgumentParser(description="Backfill manifold embeddings")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size")
    parser.add_argument("--limit", type=int, default=1000, help="Max items to process")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between batches")
    parser.add_argument("--entities", action="store_true", help="Process entities instead of segments")
    args = parser.parse_args()

    target_type = "entities" if args.entities else "segments"
    logger.info(f"Starting manifold embedding backfill for {target_type} (batch={args.batch_size}, limit={args.limit})")

    engine = create_engine(DATABASE_URL)
    embedder = MultiManifoldEmbedder()

    total_processed = 0

    if args.entities:
        # Process entities
        while total_processed < args.limit:
            entities = get_entities_needing_backfill(
                engine,
                limit=min(args.batch_size, args.limit - total_processed)
            )

            if not entities:
                logger.info("No more entities to process")
                break

            processed = process_entity_batch(engine, entities, embedder)
            total_processed += processed

            logger.info(f"Processed {processed} entities (total: {total_processed})")

            if processed > 0:
                time.sleep(args.delay)
    else:
        # Process segments
        while total_processed < args.limit:
            segments = get_segments_needing_backfill(
                engine,
                limit=min(args.batch_size, args.limit - total_processed)
            )

            if not segments:
                logger.info("No more segments to process")
                break

            processed = process_batch(engine, segments, embedder)
            total_processed += processed

            logger.info(f"Processed {processed} segments (total: {total_processed})")

            if processed > 0:
                time.sleep(args.delay)

    logger.info(f"Backfill complete. Total {target_type} processed: {total_processed}")

    # Show counts
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT manifold_type, COUNT(*)
            FROM manifold_embeddings
            GROUP BY manifold_type
            ORDER BY manifold_type
        """))
        logger.info("Manifold embedding counts:")
        for row in result:
            logger.info(f"  {row[0]}: {row[1]}")


if __name__ == "__main__":
    main()
