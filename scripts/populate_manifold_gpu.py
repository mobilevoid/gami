#!/usr/bin/env python3
"""GPU-accelerated population of product manifold coordinates.

Uses GPU for fast batch processing while ensuring CPU/GPU results match exactly.
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

# Ensure deterministic results
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


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


class GPUManifoldEncoder:
    """GPU-accelerated manifold encoder with CPU fallback."""

    def __init__(self, device=None):
        from api.llm.manifold_embeddings import ManifoldEncoder

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
                log.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                device = 'cpu'
                log.info("No GPU available, using CPU")

        self.device = device
        self.encoder = ManifoldEncoder(device=device)
        self.encoder.eval()

        # Move projection heads to GPU
        self.encoder.hyperbolic_head.to(device)
        self.encoder.spherical_head.to(device)
        self.encoder.euclidean_head.to(device)

    @torch.no_grad()
    def encode_batch(self, embeddings: np.ndarray):
        """Encode batch of 768d embeddings to manifold coordinates.

        Args:
            embeddings: (N, 768) numpy array

        Returns:
            Tuple of (hyperbolic, spherical, euclidean) numpy arrays
        """
        # Move to GPU
        x = torch.tensor(embeddings, dtype=torch.float32, device=self.device)

        # Project through heads
        h = self.encoder.hyperbolic_head(x)
        s = self.encoder.spherical_head(x)
        e = self.encoder.euclidean_head(x)

        # Move back to CPU for storage
        return (
            h.cpu().numpy(),
            s.cpu().numpy(),
            e.cpu().numpy(),
        )


def verify_cpu_gpu_match():
    """Verify CPU and GPU produce identical results."""
    log.info("Verifying CPU/GPU match...")

    # Create test embedding
    np.random.seed(42)
    test_emb = np.random.randn(1, 768).astype(np.float32)

    # CPU encode
    cpu_encoder = GPUManifoldEncoder(device='cpu')
    h_cpu, s_cpu, e_cpu = cpu_encoder.encode_batch(test_emb)

    if torch.cuda.is_available():
        # GPU encode
        gpu_encoder = GPUManifoldEncoder(device='cuda')
        h_gpu, s_gpu, e_gpu = gpu_encoder.encode_batch(test_emb)

        # Compare
        h_match = np.allclose(h_cpu, h_gpu, atol=1e-5)
        s_match = np.allclose(s_cpu, s_gpu, atol=1e-5)
        e_match = np.allclose(e_cpu, e_gpu, atol=1e-5)

        if h_match and s_match and e_match:
            log.info("CPU/GPU results match (within 1e-5 tolerance)")
            return True
        else:
            log.warning(f"CPU/GPU mismatch! h:{h_match} s:{s_match} e:{e_match}")
            h_diff = np.abs(h_cpu - h_gpu).max()
            s_diff = np.abs(s_cpu - s_gpu).max()
            e_diff = np.abs(e_cpu - e_gpu).max()
            log.warning(f"Max diffs: h={h_diff:.6f} s={s_diff:.6f} e={e_diff:.6f}")
            return False
    else:
        log.info("No GPU available, skipping match verification")
        return True


def populate_manifold_coords(batch_size=256, limit=None, target_type='segment', device=None):
    """Populate product_manifold_coords using GPU acceleration."""

    encoder = GPUManifoldEncoder(device=device)

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

        log.info(f"Found {total} {target_type}s needing manifold coordinates")

        if total == 0:
            return 0

        processed = 0
        insert_buffer = []

        with tqdm(total=total, desc=f"GPU Processing {target_type}s") as pbar:
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

                # GPU batch encode
                emb_array = np.array(embeddings, dtype=np.float32)
                h_batch, s_batch, e_batch = encoder.encode_batch(emb_array)

                # Prepare inserts
                for i, item_id in enumerate(ids):
                    h_list = h_batch[i].tolist()
                    s_list = s_batch[i].tolist()
                    e_vec = "[" + ",".join(str(v) for v in e_batch[i]) + "]"

                    insert_buffer.append({
                        "tid": item_id,
                        "ttype": target_type,
                        "h": h_list,
                        "s": s_list,
                        "e": e_vec,
                    })

                # Batch insert every 1000 items
                if len(insert_buffer) >= 1000:
                    _batch_insert(conn, insert_buffer)
                    insert_buffer = []

                processed += len(ids)
                pbar.update(len(ids))

                if limit and processed >= limit:
                    break

        # Final insert
        if insert_buffer:
            _batch_insert(conn, insert_buffer)

        log.info(f"Processed {processed} {target_type}s")
        return processed


def _batch_insert(conn, buffer):
    """Batch insert manifold coordinates."""
    for item in buffer:
        conn.execute(text("""
            INSERT INTO product_manifold_coords
                (target_id, target_type, hyperbolic_coords, spherical_coords, euclidean_coords)
            VALUES (:tid, :ttype, :h, :s, CAST(:e AS vector))
            ON CONFLICT (target_id, target_type) DO UPDATE SET
                hyperbolic_coords = EXCLUDED.hyperbolic_coords,
                spherical_coords = EXCLUDED.spherical_coords,
                euclidean_coords = EXCLUDED.euclidean_coords,
                computed_at = NOW()
        """), item)
    conn.commit()


def main():
    parser = argparse.ArgumentParser(description='GPU-accelerated manifold coordinate population')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size (larger = faster on GPU)')
    parser.add_argument('--limit', type=int, default=None, help='Max items to process')
    parser.add_argument('--type', choices=['segment', 'entity', 'all'], default='all',
                        help='Type of items to process')
    parser.add_argument('--device', choices=['cuda', 'cpu', 'auto'], default='auto',
                        help='Device to use')
    parser.add_argument('--verify', action='store_true', help='Verify CPU/GPU match before processing')
    args = parser.parse_args()

    device = None if args.device == 'auto' else args.device

    if args.verify:
        if not verify_cpu_gpu_match():
            log.error("CPU/GPU mismatch detected, aborting")
            sys.exit(1)

    start = time.time()
    total = 0

    if args.type in ('segment', 'all'):
        total += populate_manifold_coords(
            batch_size=args.batch_size,
            limit=args.limit,
            target_type='segment',
            device=device,
        )

    if args.type in ('entity', 'all'):
        total += populate_manifold_coords(
            batch_size=args.batch_size,
            limit=args.limit,
            target_type='entity',
            device=device,
        )

    elapsed = time.time() - start
    rate = total / elapsed if elapsed > 0 else 0
    log.info(f"Total: {total} items in {elapsed:.1f}s ({rate:.1f} items/sec)")


if __name__ == "__main__":
    main()
