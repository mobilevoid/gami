#!/usr/bin/env python3
"""Fast population of product manifold coordinates.

This script efficiently populates the product_manifold_coords table
by batch-processing segments with existing embeddings.
"""
import sys
sys.path.insert(0, '/opt/gami')

import os
import time
import argparse
import logging
from tqdm import tqdm
import numpy as np
import torch
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

DB_URL = os.getenv("GAMI_DATABASE_URL")
if not DB_URL:
    raise ValueError("GAMI_DATABASE_URL environment variable is required")
engine = create_engine(DB_URL)


def parse_embedding(emb_str):
    """Parse embedding from database format."""
    if emb_str is None:
        return None
    if isinstance(emb_str, (list, tuple, np.ndarray)):
        return np.array(emb_str, dtype=np.float32)
    if isinstance(emb_str, str):
        import json
        try:
            return np.array(json.loads(emb_str), dtype=np.float32)
        except json.JSONDecodeError:
            clean = emb_str.strip('[]')
            return np.array([float(v) for v in clean.split(',') if v.strip()], dtype=np.float32)
    return np.array(list(emb_str), dtype=np.float32)


def populate_manifold_coords(batch_size=100, limit=None, target_type='segment'):
    """Populate product_manifold_coords for segments."""
    from api.llm.manifold_embeddings import get_manifold_encoder

    log.info("Loading manifold encoder...")
    encoder = get_manifold_encoder()
    encoder.eval()

    with engine.connect() as conn:
        # Count items needing coords
        if target_type == 'segment':
            count_result = conn.execute(text("""
                SELECT COUNT(*) FROM segments s
                LEFT JOIN product_manifold_coords pmc
                    ON s.segment_id = pmc.target_id AND pmc.target_type = 'segment'
                WHERE s.embedding IS NOT NULL
                AND pmc.target_id IS NULL
            """))
        else:
            count_result = conn.execute(text("""
                SELECT COUNT(*) FROM entities e
                LEFT JOIN product_manifold_coords pmc
                    ON e.entity_id = pmc.target_id AND pmc.target_type = 'entity'
                WHERE e.embedding IS NOT NULL
                AND pmc.target_id IS NULL
            """))

        total = count_result.scalar()
        if limit:
            total = min(total, limit)

        log.info(f"Found {total} {target_type}s needing manifold coordinates")

        if total == 0:
            log.info("Nothing to process!")
            return 0

        # Process in batches
        processed = 0
        offset = 0

        with tqdm(total=total, desc=f"Processing {target_type}s") as pbar:
            while processed < total:
                # Fetch batch
                if target_type == 'segment':
                    result = conn.execute(text("""
                        SELECT s.segment_id as id, s.embedding
                        FROM segments s
                        LEFT JOIN product_manifold_coords pmc
                            ON s.segment_id = pmc.target_id AND pmc.target_type = 'segment'
                        WHERE s.embedding IS NOT NULL
                        AND pmc.target_id IS NULL
                        ORDER BY s.created_at DESC
                        LIMIT :batch
                    """), {"batch": batch_size})
                else:
                    result = conn.execute(text("""
                        SELECT e.entity_id as id, e.embedding
                        FROM entities e
                        LEFT JOIN product_manifold_coords pmc
                            ON e.entity_id = pmc.target_id AND pmc.target_type = 'entity'
                        WHERE e.embedding IS NOT NULL
                        AND pmc.target_id IS NULL
                        ORDER BY e.created_at DESC
                        LIMIT :batch
                    """), {"batch": batch_size})

                rows = result.fetchall()
                if not rows:
                    break

                # Parse embeddings
                ids = []
                embeddings = []
                for row in rows:
                    emb = parse_embedding(row.embedding)
                    if emb is not None and len(emb) == 768:
                        ids.append(row.id)
                        embeddings.append(emb)

                if not embeddings:
                    break

                # Batch encode to manifold
                emb_tensor = torch.tensor(np.array(embeddings), dtype=torch.float32)

                with torch.no_grad():
                    coords_batch = encoder.encode_batch_from_embeddings(emb_tensor)

                # Insert into database
                for i, item_id in enumerate(ids):
                    coords = coords_batch[i]
                    h_list = coords.hyperbolic.tolist()
                    s_list = coords.spherical.tolist()
                    e_list = coords.euclidean.tolist()

                    # Format for pgvector
                    e_vec = "[" + ",".join(str(v) for v in e_list) + "]"

                    conn.execute(text("""
                        INSERT INTO product_manifold_coords
                            (target_id, target_type, hyperbolic_coords, spherical_coords, euclidean_coords)
                        VALUES (:tid, :ttype, :h, :s, CAST(:e AS vector))
                        ON CONFLICT (target_id, target_type) DO UPDATE SET
                            hyperbolic_coords = EXCLUDED.hyperbolic_coords,
                            spherical_coords = EXCLUDED.spherical_coords,
                            euclidean_coords = EXCLUDED.euclidean_coords,
                            computed_at = NOW()
                    """), {
                        "tid": item_id,
                        "ttype": target_type,
                        "h": h_list,
                        "s": s_list,
                        "e": e_vec,
                    })

                conn.commit()
                processed += len(ids)
                pbar.update(len(ids))

                if limit and processed >= limit:
                    break

        log.info(f"Processed {processed} {target_type}s")
        return processed


def main():
    parser = argparse.ArgumentParser(description='Populate product manifold coordinates')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size')
    parser.add_argument('--limit', type=int, default=None, help='Max items to process')
    parser.add_argument('--type', choices=['segment', 'entity', 'all'], default='all',
                        help='Type of items to process')
    args = parser.parse_args()

    start = time.time()
    total = 0

    if args.type in ('segment', 'all'):
        total += populate_manifold_coords(
            batch_size=args.batch_size,
            limit=args.limit,
            target_type='segment',
        )

    if args.type in ('entity', 'all'):
        total += populate_manifold_coords(
            batch_size=args.batch_size,
            limit=args.limit,
            target_type='entity',
        )

    elapsed = time.time() - start
    log.info(f"Total: {total} items in {elapsed:.1f}s ({total/elapsed:.1f} items/sec)")


if __name__ == "__main__":
    main()
