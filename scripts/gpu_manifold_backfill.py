#!/usr/bin/env python3
"""GPU-accelerated manifold embedding backfill.

Runs on GPU for maximum speed. Processes segments and entities in large batches.
Uses sentence-transformers with CUDA acceleration.

Usage:
    python gpu_manifold_backfill.py                    # Process all
    python gpu_manifold_backfill.py --segments-only    # Segments only
    python gpu_manifold_backfill.py --entities-only    # Entities only
    python gpu_manifold_backfill.py --limit 100000     # Limit items
"""
import argparse
import logging
import sys
import time
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer

from api.config import settings
from manifold.embeddings.multi_embedder import MultiManifoldEmbedder
from manifold.embeddings.projectors import (
    compute_evidence_factors,
    compute_graph_fingerprint,
    get_time_projector,
    get_evidence_projector,
    get_relation_projector,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Database connection
DATABASE_URL = settings.DATABASE_URL_SYNC

# Batch sizes for GPU processing - tuned for 48GB RTX 6000 Ada
SEGMENT_BATCH_SIZE = 2000  # Large DB batches
ENTITY_BATCH_SIZE = 3000
GPU_BATCH_SIZE = 512  # Maximize GPU utilization


class GPUEmbedder:
    """GPU-accelerated embedding using sentence-transformers."""

    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing embedder on {device}")

        self.model = SentenceTransformer(
            'nomic-ai/nomic-embed-text-v1.5',
            trust_remote_code=True,
            device=device
        )
        self.device = device

        # Multi-manifold helper
        self.manifold_helper = MultiManifoldEmbedder()

    def embed_batch(self, texts: list) -> np.ndarray:
        """Embed a batch of texts with maximum GPU parallelism."""
        # Truncate long texts
        truncated = [t[:8000] if len(t) > 8000 else t for t in texts]
        embeddings = self.model.encode(
            truncated,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(truncated) > 500,  # Show progress for large batches
            batch_size=GPU_BATCH_SIZE,  # Large batches for GPU efficiency
        )
        return embeddings

    def extract_claims(self, text: str) -> str:
        """Extract claims from text."""
        claims = self.manifold_helper._extract_claims(text)
        return " | ".join(claims) if claims else text[:500]

    def extract_procedures(self, text: str) -> str:
        """Extract procedures from text."""
        steps = self.manifold_helper._extract_procedure_steps(text)
        return " → ".join(steps) if steps else ""

    def extract_temporal(self, text: str):
        """Extract temporal features."""
        return self.manifold_helper._extract_temporal_features(text)


def save_manifold_embedding(conn, target_id: str, target_type: str,
                           manifold_type: str, embedding: np.ndarray,
                           canonical_form: str = None):
    """Save a manifold embedding to the database."""
    emb_str = "[" + ",".join(str(x) for x in embedding.flatten()) + "]"

    sql = f"""
        INSERT INTO manifold_embeddings
        (target_id, target_type, manifold_type, embedding, canonical_form, updated_at)
        VALUES (:tid, :ttype, :mtype, '{emb_str}'::vector, :canonical, NOW())
        ON CONFLICT (target_id, target_type, manifold_type)
        DO UPDATE SET embedding = EXCLUDED.embedding, canonical_form = EXCLUDED.canonical_form, updated_at = NOW()
    """
    conn.execute(text(sql), {
        "tid": target_id,
        "ttype": target_type,
        "mtype": manifold_type,
        "canonical": canonical_form[:500] if canonical_form else None,
    })


def save_segment_manifolds(engine, segment_id, emb_topic, emb_claim, emb_procedure,
                           text_content, claim_text, proc_text, text_for_temporal,
                           time_projector, evidence_projector, relation_projector, embedder):
    """Save all manifold embeddings for a single segment. Runs in thread pool."""
    try:
        with engine.begin() as conn:
            # TOPIC
            save_manifold_embedding(conn, segment_id, "segment", "TOPIC", emb_topic, text_content[:500])

            # CLAIM
            save_manifold_embedding(conn, segment_id, "segment", "CLAIM", emb_claim, claim_text[:500])

            # PROCEDURE (only if extracted)
            if proc_text:
                save_manifold_embedding(conn, segment_id, "segment", "PROCEDURE", emb_procedure, proc_text[:500])

            # TIME
            time_features = embedder.extract_temporal(text_for_temporal)
            if time_features is not None:
                time_emb = time_projector.project(time_features)
                save_manifold_embedding(conn, segment_id, "segment", "TIME", time_emb, None)

            # EVIDENCE
            evidence_factors = compute_evidence_factors(
                authority_score=0.5, corroboration_count=0, recency_days=30.0,
                specificity_score=0.5, contradiction_ratio=0.0,
            )
            evidence_emb = evidence_projector.project(evidence_factors)
            save_manifold_embedding(conn, segment_id, "segment", "EVIDENCE", evidence_emb, None)

            # RELATION (from provenance)
            try:
                prov_result = conn.execute(text("""
                    SELECT p.target_id, e.entity_type
                    FROM provenance p
                    LEFT JOIN entities e ON p.target_id = e.entity_id
                    WHERE p.segment_id = :sid AND p.target_type = 'entity'
                """), {"sid": segment_id})
                entity_rows = prov_result.fetchall()

                if entity_rows:
                    entity_ids = [r[0] for r in entity_rows]
                    entity_types = [r[1] or "unknown" for r in entity_rows if r[1]]

                    rel_result = conn.execute(text("""
                        SELECT relation_type,
                            CASE WHEN source_entity_id = ANY(:eids) THEN 'out' ELSE 'in' END
                        FROM relations
                        WHERE source_entity_id = ANY(:eids) OR target_entity_id = ANY(:eids)
                        LIMIT 50
                    """), {"eids": entity_ids})
                    rel_rows = rel_result.fetchall()

                    in_edges, out_edges = {}, {}
                    for rel_type, direction in rel_rows:
                        rel_type = rel_type or "relates_to"
                        if direction == "in":
                            in_edges[rel_type] = in_edges.get(rel_type, 0) + 1
                        else:
                            out_edges[rel_type] = out_edges.get(rel_type, 0) + 1

                    fingerprint = compute_graph_fingerprint(
                        in_edges=in_edges, out_edges=out_edges,
                        connected_types=list(set(entity_types)),
                        centrality=min(1.0, len(rel_rows) / 50.0),
                    )
                    relation_emb = relation_projector.project(fingerprint)
                    save_manifold_embedding(conn, segment_id, "segment", "RELATION", relation_emb, None)
            except Exception:
                pass
        return True
    except Exception as e:
        return False


def process_segments(engine, embedder: GPUEmbedder, limit: int = None):
    """Process segments in batches with parallel DB writes."""
    time_projector = get_time_projector()
    evidence_projector = get_evidence_projector()
    relation_projector = get_relation_projector()

    total_processed = 0
    batch_num = 0
    start_time = time.time()

    # Thread pool for parallel DB writes - 16 threads for high throughput
    db_executor = ThreadPoolExecutor(max_workers=16)

    while True:
        if limit and total_processed >= limit:
            break

        batch_num += 1
        batch_limit = min(SEGMENT_BATCH_SIZE, limit - total_processed) if limit else SEGMENT_BATCH_SIZE

        # Get segments needing manifold embeddings
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
            """), {"lim": batch_limit})
            segments = result.fetchall()

        if not segments:
            logger.info("No more segments to process")
            break

        logger.info(f"Batch {batch_num}: Processing {len(segments)} segments...")

        # Prepare texts for batch embedding
        texts_topic = []
        texts_claim = []
        texts_procedure = []
        segment_data = []

        for seg in segments:
            segment_id, seg_text, tenant_id = seg
            if not seg_text or len(seg_text) < 10:
                continue

            texts_topic.append(seg_text)

            # Extract claims
            claim_text = embedder.extract_claims(seg_text)
            texts_claim.append(claim_text)

            # Extract procedures
            proc_text = embedder.extract_procedures(seg_text)
            texts_procedure.append(proc_text if proc_text else seg_text[:500])

            segment_data.append({
                "segment_id": segment_id,
                "text": seg_text,
                "claim_text": claim_text,
                "proc_text": proc_text if proc_text else None,
            })

        if not texts_topic:
            continue

        # Batch embed on GPU
        t0 = time.time()
        emb_topic = embedder.embed_batch(texts_topic)
        emb_claim = embedder.embed_batch(texts_claim)
        emb_procedure = embedder.embed_batch(texts_procedure)
        embed_time = time.time() - t0

        # Submit all DB writes to thread pool in parallel
        futures = []
        for i, seg in enumerate(segment_data):
            future = db_executor.submit(
                save_segment_manifolds,
                engine, seg["segment_id"],
                emb_topic[i], emb_claim[i], emb_procedure[i],
                seg["text"], seg["claim_text"], seg["proc_text"], seg["text"],
                time_projector, evidence_projector, relation_projector, embedder
            )
            futures.append(future)

        # Wait for all DB writes to complete
        saved = 0
        for future in as_completed(futures):
            if future.result():
                saved += 1
                total_processed += 1

        elapsed = time.time() - start_time
        rate = total_processed / elapsed if elapsed > 0 else 0
        logger.info(f"  Batch done. Saved: {saved}/{len(segment_data)}, Total: {total_processed}, Rate: {rate:.1f} seg/s, Embed time: {embed_time:.2f}s")

    db_executor.shutdown(wait=True)
    return total_processed


def save_entity_manifolds(engine, entity_id, embedding, canonical_name, entity_type, relation_projector):
    """Save all manifold embeddings for a single entity. Runs in thread pool."""
    try:
        with engine.begin() as conn:
            # TOPIC
            save_manifold_embedding(conn, entity_id, "entity", "TOPIC", embedding, canonical_name)

            # RELATION (from graph structure)
            try:
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
                    in_edges, out_edges, other_ids = {}, {}, []
                    for rel_type, direction, other_id in rel_rows:
                        rel_type = rel_type or "relates_to"
                        other_ids.append(other_id)
                        if direction == "in":
                            in_edges[rel_type] = in_edges.get(rel_type, 0) + 1
                        else:
                            out_edges[rel_type] = out_edges.get(rel_type, 0) + 1

                    type_result = conn.execute(text("""
                        SELECT DISTINCT entity_type FROM entities WHERE entity_id = ANY(:ids)
                    """), {"ids": other_ids[:20]})
                    connected_types = [r[0] for r in type_result.fetchall() if r[0]]

                    fingerprint = compute_graph_fingerprint(
                        in_edges=in_edges, out_edges=out_edges,
                        connected_types=connected_types + [entity_type],
                        centrality=min(1.0, len(rel_rows) / 50.0),
                    )
                    relation_emb = relation_projector.project(fingerprint)
                    save_manifold_embedding(conn, entity_id, "entity", "RELATION", relation_emb, None)
            except Exception:
                pass
        return True
    except Exception:
        return False


def process_entities(engine, embedder: GPUEmbedder, limit: int = None):
    """Process entities in batches with parallel DB writes."""
    relation_projector = get_relation_projector()

    total_processed = 0
    batch_num = 0
    start_time = time.time()

    # Thread pool for parallel DB writes
    db_executor = ThreadPoolExecutor(max_workers=16)

    while True:
        if limit and total_processed >= limit:
            break

        batch_num += 1
        batch_limit = min(ENTITY_BATCH_SIZE, limit - total_processed) if limit else ENTITY_BATCH_SIZE

        # Get entities needing manifold embeddings
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
            """), {"lim": batch_limit})
            entities = result.fetchall()

        if not entities:
            logger.info("No more entities to process")
            break

        logger.info(f"Batch {batch_num}: Processing {len(entities)} entities...")

        # Prepare texts
        texts = []
        entity_data = []

        for ent in entities:
            entity_id, canonical_name, entity_type, tenant_id = ent
            if not canonical_name:
                continue

            texts.append(canonical_name)
            entity_data.append({
                "entity_id": entity_id,
                "canonical_name": canonical_name,
                "entity_type": entity_type or "unknown",
            })

        if not texts:
            continue

        # Batch embed on GPU
        embeddings = embedder.embed_batch(texts)

        # Submit all DB writes to thread pool in parallel
        futures = []
        for i, ent in enumerate(entity_data):
            future = db_executor.submit(
                save_entity_manifolds,
                engine, ent["entity_id"], embeddings[i],
                ent["canonical_name"], ent["entity_type"], relation_projector
            )
            futures.append(future)

        # Wait for all DB writes to complete
        saved = 0
        for future in as_completed(futures):
            if future.result():
                saved += 1
                total_processed += 1

        elapsed = time.time() - start_time
        rate = total_processed / elapsed if elapsed > 0 else 0
        logger.info(f"  Batch done. Saved: {saved}/{len(entity_data)}, Total: {total_processed}, Rate: {rate:.1f} ent/s")

    db_executor.shutdown(wait=True)
    return total_processed


def main():
    parser = argparse.ArgumentParser(description="GPU-accelerated manifold backfill")
    parser.add_argument("--segments-only", action="store_true", help="Process only segments")
    parser.add_argument("--entities-only", action="store_true", help="Process only entities")
    parser.add_argument("--limit", type=int, default=None, help="Limit total items to process")
    args = parser.parse_args()

    # Check GPU
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        logger.warning("No GPU available, running on CPU")

    engine = create_engine(DATABASE_URL, pool_size=10, max_overflow=20)
    embedder = GPUEmbedder()

    start_time = time.time()
    segment_count = 0
    entity_count = 0

    # Process segments
    if not args.entities_only:
        logger.info("=" * 60)
        logger.info("PROCESSING SEGMENTS")
        logger.info("=" * 60)
        segment_count = process_segments(engine, embedder, args.limit)

    # Process entities
    if not args.segments_only:
        logger.info("=" * 60)
        logger.info("PROCESSING ENTITIES")
        logger.info("=" * 60)
        entity_limit = args.limit - segment_count if args.limit else None
        entity_count = process_entities(engine, embedder, entity_limit if entity_limit and entity_limit > 0 else None)

    total_time = time.time() - start_time

    logger.info("=" * 60)
    logger.info("BACKFILL COMPLETE")
    logger.info(f"  Segments: {segment_count}")
    logger.info(f"  Entities: {entity_count}")
    logger.info(f"  Total time: {total_time/60:.1f} minutes")
    logger.info(f"  Average rate: {(segment_count + entity_count) / total_time:.1f} items/sec")
    logger.info("=" * 60)

    # Show final counts
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT target_type, manifold_type, COUNT(*)
            FROM manifold_embeddings
            GROUP BY target_type, manifold_type
            ORDER BY target_type, manifold_type
        """))
        logger.info("Manifold embedding counts:")
        for row in result:
            logger.info(f"  {row[0]}/{row[1]}: {row[2]}")


if __name__ == "__main__":
    main()
