#!/usr/bin/env python3
"""Fast GPU manifold backfill with pipelining and larger batches."""
import sys
import time
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import queue
import threading

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer

from api.config import settings
from manifold.embeddings.projectors import (
    compute_evidence_factors,
    get_time_projector,
    get_evidence_projector,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BATCH_SIZE = 300   # DB batch - smaller to avoid OOM
GPU_BATCH = 32     # GPU encoding batch - conservative for memory

def fetch_batch(engine, batch_size):
    """Fetch a batch of segments needing manifold embeddings."""
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT s.segment_id, s.text
            FROM segments s
            LEFT JOIN manifold_embeddings me ON s.segment_id = me.target_id AND me.manifold_type = 'TOPIC'
            WHERE s.embedding IS NOT NULL
            AND s.storage_tier != 'cold'
            AND LENGTH(s.text) > 50
            AND me.target_id IS NULL
            LIMIT :lim
        """), {"lim": batch_size}).fetchall()
    return rows

def save_batch(engine, values_list):
    """Bulk insert manifold embeddings."""
    if not values_list:
        return
    sql = f"""
        INSERT INTO manifold_embeddings (target_id, target_type, manifold_type, embedding, canonical_form, updated_at)
        VALUES {','.join(values_list)}
        ON CONFLICT (target_id, target_type, manifold_type) DO UPDATE SET
            embedding = EXCLUDED.embedding, updated_at = NOW()
    """
    with engine.begin() as conn:
        conn.execute(text(sql))

def main():
    engine = create_engine(settings.DATABASE_URL_SYNC, pool_size=10)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using {device}, loading model...")

    model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True, device=device)
    ev_proj = get_evidence_projector()

    # Pre-compute default evidence embedding
    default_ev = ev_proj.project(compute_evidence_factors(0.5, 0, 30.0, 0.5, 0.0))
    default_ev_str = "[" + ",".join(f"{x:.8f}" for x in default_ev) + "]"

    total = 0
    start = time.time()

    # Thread pool for async DB saves
    save_executor = ThreadPoolExecutor(max_workers=4)
    pending_saves = []

    # Pre-fetch first batch
    next_batch = fetch_batch(engine, BATCH_SIZE)

    while next_batch:
        rows = next_batch

        # Start fetching next batch in background
        fetch_future = save_executor.submit(fetch_batch, engine, BATCH_SIZE)

        # Process current batch
        segment_ids = []
        texts = []
        for sid, txt in rows:
            if txt and len(txt) > 10:
                segment_ids.append(sid)
                texts.append(txt[:8000])  # Truncate long texts

        if not texts:
            next_batch = fetch_future.result()
            continue

        # GPU embedding
        embeddings = model.encode(texts, batch_size=GPU_BATCH, normalize_embeddings=True, show_progress_bar=False)

        # Build bulk insert values
        values = []
        for i, sid in enumerate(segment_ids):
            emb = embeddings[i]
            emb_str = "[" + ",".join(f"{x:.8f}" for x in emb) + "]"

            # TOPIC + CLAIM (same for speed)
            values.append(f"('{sid}', 'segment', 'TOPIC', '{emb_str}'::vector, NULL, NOW())")
            values.append(f"('{sid}', 'segment', 'CLAIM', '{emb_str}'::vector, NULL, NOW())")
            values.append(f"('{sid}', 'segment', 'EVIDENCE', '{default_ev_str}'::vector, NULL, NOW())")

        # Wait for any pending saves to complete
        for future in pending_saves:
            future.result()
        pending_saves.clear()

        # Save current batch async
        save_future = save_executor.submit(save_batch, engine, values)
        pending_saves.append(save_future)

        total += len(segment_ids)

        # Get next batch (should already be ready)
        next_batch = fetch_future.result()

        elapsed = time.time() - start
        rate = total / elapsed
        remaining = 366000 - total
        eta_hrs = (remaining / rate) / 3600 if rate > 0 else 0

        logger.info(f"Done {total}, rate {rate:.1f}/s, ETA {eta_hrs:.1f}h")

    # Wait for final saves
    for future in pending_saves:
        future.result()

    save_executor.shutdown()
    logger.info(f"Complete! Total: {total}, Time: {(time.time()-start)/60:.1f}min")

if __name__ == "__main__":
    main()
