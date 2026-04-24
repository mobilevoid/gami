#!/usr/bin/env python3
"""GPU entity manifold backfill."""
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BATCH_SIZE = 500
GPU_BATCH = 64

def main():
    engine = create_engine(settings.DATABASE_URL_SYNC, pool_size=5)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using {device}, loading model...")

    model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True, device=device)

    total = 0
    start = time.time()

    while True:
        # Fetch entities needing manifold embeddings
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT e.entity_id, e.canonical_name
                FROM entities e
                LEFT JOIN manifold_embeddings me ON e.entity_id = me.target_id AND me.manifold_type = 'TOPIC'
                WHERE e.embedding IS NOT NULL
                AND e.status = 'active'
                AND me.target_id IS NULL
                LIMIT :lim
            """), {"lim": BATCH_SIZE}).fetchall()

        if not rows:
            logger.info("Done - no more entities!")
            break

        # Process
        entity_ids = []
        texts = []
        for eid, name in rows:
            if name and len(name) > 0:
                entity_ids.append(eid)
                texts.append(name[:1000])

        if not texts:
            continue

        # GPU embedding
        embeddings = model.encode(texts, batch_size=GPU_BATCH, normalize_embeddings=True, show_progress_bar=False)

        # Build bulk insert - just TOPIC for entities
        values = []
        for i, eid in enumerate(entity_ids):
            emb = embeddings[i]
            emb_str = "[" + ",".join(f"{x:.8f}" for x in emb) + "]"
            values.append(f"('{eid}', 'entity', 'TOPIC', '{emb_str}'::vector, NULL, NOW())")

        # Save
        sql = f"""
            INSERT INTO manifold_embeddings (target_id, target_type, manifold_type, embedding, canonical_form, updated_at)
            VALUES {','.join(values)}
            ON CONFLICT (target_id, target_type, manifold_type) DO UPDATE SET
                embedding = EXCLUDED.embedding, updated_at = NOW()
        """
        with engine.begin() as conn:
            conn.execute(text(sql))

        total += len(entity_ids)
        elapsed = time.time() - start
        rate = total / elapsed
        logger.info(f"Done {total}, rate {rate:.1f}/s")

    logger.info(f"Complete! Total: {total}, Time: {(time.time()-start)/60:.1f}min")

if __name__ == "__main__":
    main()
