#!/usr/bin/env python3
"""Populate product manifold coordinates using GPU (gami-embed conda env).

Usage:
    conda run -n gami-embed python scripts/populate_product_manifold.py

This script uses the trained manifold projection heads to convert
768d embeddings to H^32 × S^16 × E^64 product manifold coordinates.
"""
import sys
sys.path.insert(0, '/opt/gami')

import os
import json
import time
import argparse
import logging
import numpy as np
import torch
from tqdm import tqdm
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

DB_URL = os.getenv("GAMI_DATABASE_URL")
if not DB_URL:
    raise ValueError("GAMI_DATABASE_URL environment variable is required")
engine = create_engine(DB_URL)

# Manifold projection head dimensions
H_DIM = 32   # Hyperbolic (Poincaré ball)
S_DIM = 16   # Spherical
E_DIM = 64   # Euclidean


class ManifoldProjector(torch.nn.Module):
    """Projection heads for product manifold H^32 × S^16 × E^64."""

    def __init__(self, input_dim=768, device='cuda'):
        super().__init__()
        self.device = device

        # Hyperbolic projection (Poincaré ball)
        self.hyperbolic_head = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.GELU(),
            torch.nn.Linear(256, H_DIM),
            torch.nn.Tanh(),  # Keep in (-1, 1) for Poincaré ball
        )

        # Spherical projection
        self.spherical_head = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, S_DIM),
        )

        # Euclidean projection
        self.euclidean_head = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, E_DIM),
        )

        self.to(device)

    def forward(self, x):
        """Project embeddings to product manifold."""
        h = self.hyperbolic_head(x)
        # Scale to stay inside Poincaré ball (norm < 1)
        h_norm = torch.norm(h, dim=-1, keepdim=True)
        h = h / (h_norm + 1e-5) * torch.tanh(h_norm) * 0.95

        s = self.spherical_head(x)
        # Normalize to unit sphere
        s = torch.nn.functional.normalize(s, p=2, dim=-1)

        e = self.euclidean_head(x)

        return h, s, e

    def load_weights(self, path='/opt/gami/api/llm/manifold_weights.pt'):
        """Load trained projection weights."""
        if os.path.exists(path):
            state = torch.load(path, map_location=self.device)

            # Map keys from full encoder to our projector
            new_state = {}
            for k, v in state.items():
                if 'hyperbolic_head' in k:
                    new_key = k.replace('encoder.', '')
                    new_state[new_key] = v
                elif 'spherical_head' in k:
                    new_key = k.replace('encoder.', '')
                    new_state[new_key] = v
                elif 'euclidean_head' in k:
                    new_key = k.replace('encoder.', '')
                    new_state[new_key] = v

            if new_state:
                self.load_state_dict(new_state, strict=False)
                log.info(f"Loaded weights from {path}")
            else:
                log.warning(f"No matching weights in {path}, using random init")
        else:
            log.warning(f"No weights at {path}, using random init")


def parse_embedding(emb_str):
    """Parse embedding from database format."""
    if emb_str is None:
        return None
    if isinstance(emb_str, (list, tuple, np.ndarray)):
        return np.array(emb_str, dtype=np.float32)
    if isinstance(emb_str, str):
        try:
            return np.array(json.loads(emb_str), dtype=np.float32)
        except json.JSONDecodeError:
            clean = emb_str.strip('[]')
            return np.array([float(v) for v in clean.split(',') if v.strip()], dtype=np.float32)
    return np.array(list(emb_str), dtype=np.float32)


def populate_manifold_coords(batch_size=512, limit=None, target_type='segment'):
    """Populate product_manifold_coords using GPU."""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log.info(f"Using device: {device}")

    if device == 'cuda':
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
        log.info(f"Free memory: {torch.cuda.mem_get_info()[0] / 1e9:.1f} GB")

    # Load projector with trained weights
    projector = ManifoldProjector(device=device)
    projector.load_weights()
    projector.eval()

    with engine.connect() as conn:
        # Count items needing coords
        if target_type == 'segment':
            count_sql = """
                SELECT COUNT(*) FROM segments s
                LEFT JOIN product_manifold_coords pmc
                    ON s.segment_id = pmc.target_id AND pmc.target_type = 'segment'
                WHERE s.embedding IS NOT NULL
                AND pmc.target_id IS NULL
            """
            fetch_sql = """
                SELECT s.segment_id as id, s.embedding
                FROM segments s
                LEFT JOIN product_manifold_coords pmc
                    ON s.segment_id = pmc.target_id AND pmc.target_type = 'segment'
                WHERE s.embedding IS NOT NULL
                AND pmc.target_id IS NULL
                ORDER BY s.created_at DESC
                LIMIT :batch
            """
        else:
            count_sql = """
                SELECT COUNT(*) FROM entities e
                LEFT JOIN product_manifold_coords pmc
                    ON e.entity_id = pmc.target_id AND pmc.target_type = 'entity'
                WHERE e.embedding IS NOT NULL
                AND pmc.target_id IS NULL
            """
            fetch_sql = """
                SELECT e.entity_id as id, e.embedding
                FROM entities e
                LEFT JOIN product_manifold_coords pmc
                    ON e.entity_id = pmc.target_id AND pmc.target_type = 'entity'
                WHERE e.embedding IS NOT NULL
                AND pmc.target_id IS NULL
                ORDER BY e.created_at DESC
                LIMIT :batch
            """

        total = conn.execute(text(count_sql)).scalar()
        if limit:
            total = min(total, limit)

        log.info(f"Found {total:,} {target_type}s needing manifold coordinates")

        if total == 0:
            return 0

        processed = 0
        start_time = time.time()

        with tqdm(total=total, desc=f"GPU {target_type}s") as pbar:
            while processed < total:
                # Fetch batch
                result = conn.execute(text(fetch_sql), {"batch": batch_size})
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

                # GPU batch projection
                emb_tensor = torch.tensor(np.array(embeddings), dtype=torch.float32, device=device)

                with torch.no_grad():
                    h_batch, s_batch, e_batch = projector(emb_tensor)

                # Move to CPU for database insert
                h_np = h_batch.cpu().numpy()
                s_np = s_batch.cpu().numpy()
                e_np = e_batch.cpu().numpy()

                # Batch insert
                for i, item_id in enumerate(ids):
                    h_list = h_np[i].tolist()
                    s_list = s_np[i].tolist()
                    e_vec = "[" + ",".join(str(v) for v in e_np[i]) + "]"

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

        elapsed = time.time() - start_time
        rate = processed / elapsed if elapsed > 0 else 0
        log.info(f"Processed {processed:,} {target_type}s in {elapsed:.1f}s ({rate:.0f}/sec)")

        return processed


def main():
    parser = argparse.ArgumentParser(description='GPU manifold coordinate population')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size')
    parser.add_argument('--limit', type=int, default=None, help='Max items')
    parser.add_argument('--type', choices=['segment', 'entity', 'all'], default='all')
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
    rate = total / elapsed if elapsed > 0 else 0
    log.info(f"TOTAL: {total:,} items in {elapsed:.1f}s ({rate:.0f}/sec)")


if __name__ == "__main__":
    main()
