#!/usr/bin/env python3
"""Fast GPU manifold backfill with bulk inserts."""
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
from manifold.embeddings.multi_embedder import MultiManifoldEmbedder
from manifold.embeddings.projectors import (
    compute_evidence_factors,
    get_time_projector,
    get_evidence_projector,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BATCH_SIZE = 200  # Smaller batches, faster commits
GPU_BATCH = 64    # Reduced for memory

def main():
    engine = create_engine(settings.DATABASE_URL_SYNC, pool_size=5)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using {device}, loading model...")

    model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True, device=device)
    helper = MultiManifoldEmbedder()
    time_proj = get_time_projector()
    ev_proj = get_evidence_projector()

    total = 0
    start = time.time()

    while True:
        # Get batch of segments
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT s.segment_id, s.text
                FROM segments s
                LEFT JOIN manifold_embeddings me ON s.segment_id = me.target_id AND me.manifold_type = 'TOPIC'
                WHERE s.embedding IS NOT NULL
                AND s.storage_tier != 'cold'
                AND LENGTH(s.text) > 50
                AND me.target_id IS NULL
                ORDER BY s.created_at DESC
                LIMIT :lim
            """), {"lim": BATCH_SIZE}).fetchall()

        if not rows:
            logger.info("Done!")
            break

        # Prepare texts
        segment_ids = []
        texts = []
        for sid, txt in rows:
            if txt and len(txt) > 10:
                segment_ids.append(sid)
                texts.append(txt)

        if not texts:
            continue

        # Embed all at once
        embeddings = model.encode(texts, batch_size=GPU_BATCH, normalize_embeddings=True, show_progress_bar=False)

        # Build bulk insert values
        values = []
        for i, sid in enumerate(segment_ids):
            emb = embeddings[i]
            emb_str = "[" + ",".join(f"{x:.8f}" for x in emb) + "]"

            # TOPIC
            values.append(f"('{sid}', 'segment', 'TOPIC', '{emb_str}'::vector, NULL, NOW())")

            # CLAIM - use same embedding for speed (can refine later)
            values.append(f"('{sid}', 'segment', 'CLAIM', '{emb_str}'::vector, NULL, NOW())")

            # EVIDENCE - default factors
            ev = ev_proj.project(compute_evidence_factors(0.5, 0, 30.0, 0.5, 0.0))
            ev_str = "[" + ",".join(f"{x:.8f}" for x in ev) + "]"
            values.append(f"('{sid}', 'segment', 'EVIDENCE', '{ev_str}'::vector, NULL, NOW())")

            # TIME - if extractable
            tf = helper._extract_temporal_features(texts[i])
            if tf is not None:
                t_emb = time_proj.project(tf)
                t_str = "[" + ",".join(f"{x:.8f}" for x in t_emb) + "]"
                values.append(f"('{sid}', 'segment', 'TIME', '{t_str}'::vector, NULL, NOW())")

        # Single bulk insert
        if values:
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
        remaining = 366549 - total  # approximate
        eta_hrs = (remaining / rate) / 3600 if rate > 0 else 0

        logger.info(f"Done {total}, rate {rate:.1f}/s, ETA {eta_hrs:.1f}h")

if __name__ == "__main__":
    main()
