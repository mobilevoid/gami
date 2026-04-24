#!/usr/bin/env python3
"""Ultra-fast GPU embedding with COPY to temp table."""
import os, sys, time, logging, io
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import psycopg2
from sentence_transformers import SentenceTransformer
from api.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("embed_gpu_copy")

GPU_BATCH = 128  # Safe batch size to avoid OOM
DB_URL = settings.DATABASE_URL_SYNC

def get_conn():
    """Get raw psycopg2 connection for COPY support."""
    # Parse SQLAlchemy URL to psycopg2 format
    # postgresql+psycopg2://user:pass@host:port/db or postgresql://...
    import re
    m = re.match(r'postgresql(?:\+psycopg2)?://([^:]+):([^@]+)@([^:]+):(\d+)/(.+)', DB_URL)
    if not m:
        raise ValueError(f"Cannot parse DB URL: {DB_URL}")
    return psycopg2.connect(
        user=m.group(1), password=m.group(2),
        host=m.group(3), port=m.group(4), dbname=m.group(5)
    )

def main():
    log.info("Loading nomic-embed-text-v1.5 on GPU...")
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    model = model.to("cuda")
    log.info(f"Model loaded. GPU memory: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    conn = get_conn()
    conn.autocommit = False
    cur = conn.cursor()

    # Count total
    cur.execute("SELECT count(*) FROM segments WHERE embedding IS NULL")
    total = cur.fetchone()[0]
    log.info(f"Segments to embed: {total:,}")

    if total == 0:
        return

    # Create temp table once
    cur.execute("""
        CREATE TEMP TABLE IF NOT EXISTS embed_batch (
            segment_id TEXT PRIMARY KEY,
            embedding vector(768)
        ) ON COMMIT DELETE ROWS
    """)
    conn.commit()

    embedded = errors = 0
    start = time.time()

    while embedded < total:
        try:
            # Fetch batch
            cur.execute("""
                SELECT segment_id, text FROM segments
                WHERE embedding IS NULL
                LIMIT %s
            """, (GPU_BATCH,))
            rows = cur.fetchall()

            if not rows:
                break

            seg_ids = [r[0] for r in rows]
            texts = [r[1][:8000] if r[1] else "" for r in rows]

            # GPU batch encode
            prefixed = ["search_document: " + t for t in texts]
            embeddings = model.encode(
                prefixed,
                batch_size=GPU_BATCH,
                show_progress_bar=False,
                normalize_embeddings=True
            )

            # Build COPY data (tab-separated: segment_id \t [vector])
            buf = io.StringIO()
            for sid, emb in zip(seg_ids, embeddings):
                vec_str = "[" + ",".join(f"{float(v):.8f}" for v in emb) + "]"
                buf.write(f"{sid}\t{vec_str}\n")
            buf.seek(0)

            # COPY to temp table
            cur.copy_expert("COPY embed_batch (segment_id, embedding) FROM STDIN", buf)

            # Update main table from temp
            cur.execute("""
                UPDATE segments s
                SET embedding = e.embedding
                FROM embed_batch e
                WHERE s.segment_id = e.segment_id
            """)

            # Clear temp table for next batch
            cur.execute("DELETE FROM embed_batch")
            conn.commit()

            embedded += len(seg_ids)

            elapsed = time.time() - start
            rate = embedded / elapsed if elapsed > 0 else 0
            remaining = (total - embedded) / rate if rate > 0 else 0

            if embedded % 2048 == 0 or embedded < 2000:
                log.info(f"{embedded:,}/{total:,} ({rate:.0f}/s, ~{remaining/60:.1f}min left, {errors} err)")

        except Exception as e:
            log.warning(f"Batch failed: {e}")
            errors += len(rows) if rows else GPU_BATCH
            conn.rollback()
            # Clear CUDA cache on OOM
            if "CUDA out of memory" in str(e):
                torch.cuda.empty_cache()

    conn.close()
    elapsed = time.time() - start
    log.info(f"Done: {embedded:,} in {elapsed/60:.1f}min ({embedded/max(elapsed,1):.0f}/s). Errors: {errors}")

if __name__ == "__main__":
    main()
