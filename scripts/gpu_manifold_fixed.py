#!/usr/bin/env python3
"""Fixed GPU manifold backfill - no race conditions."""
import sys
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer

from api.config import settings
from manifold.embeddings.projectors import (
    compute_evidence_factors,
    get_evidence_projector,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BATCH_SIZE = 500   # DB batch
GPU_BATCH = 64     # GPU encoding batch

def main():
    engine = create_engine(settings.DATABASE_URL_SYNC, pool_size=5)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using {device}, loading model...")

    model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True, device=device)
    ev_proj = get_evidence_projector()

    # Pre-compute default evidence embedding
    default_ev = ev_proj.project(compute_evidence_factors(0.5, 0, 30.0, 0.5, 0.0))
    default_ev_str = "[" + ",".join(f"{x:.8f}" for x in default_ev) + "]"

    total = 0
    start = time.time()

    while True:
        # Fetch batch
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
            """), {"lim": BATCH_SIZE}).fetchall()

        if not rows:
            logger.info("Done - no more segments!")
            break

        # Process
        segment_ids = []
        texts = []
        for sid, txt in rows:
            if txt and len(txt) > 10:
                segment_ids.append(sid)
                texts.append(txt[:8000])

        if not texts:
            continue

        # GPU embedding
        embeddings = model.encode(texts, batch_size=GPU_BATCH, normalize_embeddings=True, show_progress_bar=False)

        # Build bulk insert
        values = []
        for i, sid in enumerate(segment_ids):
            emb = embeddings[i]
            emb_str = "[" + ",".join(f"{x:.8f}" for x in emb) + "]"
            values.append(f"('{sid}', 'segment', 'TOPIC', '{emb_str}'::vector, NULL, NOW())")
            values.append(f"('{sid}', 'segment', 'CLAIM', '{emb_str}'::vector, NULL, NOW())")
            values.append(f"('{sid}', 'segment', 'EVIDENCE', '{default_ev_str}'::vector, NULL, NOW())")

        # Save SYNCHRONOUSLY - no race condition
        sql = f"""
            INSERT INTO manifold_embeddings (target_id, target_type, manifold_type, embedding, canonical_form, updated_at)
            VALUES {','.join(values)}
            ON CONFLICT (target_id, target_type, manifold_type) DO UPDATE SET
                embedding = EXCLUDED.embedding, updated_at = NOW()
        """
        with engine.begin() as conn:
            conn.execute(text(sql))

        total += len(segment_ids)
        elapsed = time.time() - start
        rate = total / elapsed

        # Get actual remaining count every 10 batches
        if total % 5000 == 0:
            with engine.connect() as conn:
                remaining = conn.execute(text("""
                    SELECT COUNT(*) FROM segments s
                    LEFT JOIN manifold_embeddings me ON s.segment_id = me.target_id AND me.manifold_type = 'TOPIC'
                    WHERE s.embedding IS NOT NULL AND s.storage_tier != 'cold'
                    AND LENGTH(s.text) > 50 AND me.target_id IS NULL
                """)).scalar()
            eta_hrs = (remaining / rate) / 3600 if rate > 0 else 0
            logger.info(f"Done {total}, rate {rate:.1f}/s, remaining {remaining}, ETA {eta_hrs:.1f}h")
        else:
            logger.info(f"Done {total}, rate {rate:.1f}/s")

    logger.info(f"Complete! Total: {total}, Time: {(time.time()-start)/60:.1f}min")

if __name__ == "__main__":
    main()
