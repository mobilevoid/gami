#!/usr/bin/env python3
"""Fast GPU batch embedding with bulk updates."""
import os, sys, time, logging, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.config import settings
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("embed_gpu_fast")

GPU_BATCH = 256      # GPU encoding batch
DB_BATCH = 256       # DB update batch (same as GPU for simplicity)

def main():
    log.info("Loading nomic-embed-text-v1.5 on GPU...")
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    model = model.to("cuda")
    log.info(f"Model loaded. GPU memory: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    engine = create_engine(settings.DATABASE_URL_SYNC, pool_size=5, pool_pre_ping=True)

    with engine.connect() as conn:
        total = conn.execute(text("""
            SELECT count(*) FROM segments
            WHERE embedding IS NULL
        """)).scalar()
        log.info(f"Segments to embed: {total:,}")

        if total == 0:
            return

        embedded = errors = 0
        start = time.time()

        while True:
            # Fetch batch
            rows = conn.execute(text("""
                SELECT segment_id, text FROM segments
                WHERE embedding IS NULL
                LIMIT :lim
            """), {"lim": GPU_BATCH}).fetchall()

            if not rows:
                break

            seg_ids = [r[0] for r in rows]
            texts = [r[1][:8000] if r[1] else "" for r in rows]

            # Prefix for nomic-embed-text
            prefixed = ["search_document: " + t for t in texts]

            try:
                # GPU batch encode
                embeddings = model.encode(
                    prefixed,
                    batch_size=GPU_BATCH,
                    show_progress_bar=False,
                    normalize_embeddings=True
                )

                # Build bulk update values
                values = []
                for sid, emb in zip(seg_ids, embeddings):
                    vec_str = "[" + ",".join(f"{float(v):.8f}" for v in emb) + "]"
                    # Escape single quotes in segment_id if any
                    safe_sid = sid.replace("'", "''")
                    values.append(f"('{safe_sid}', '{vec_str}'::vector)")

                # Bulk update using VALUES and JOIN
                if values:
                    update_sql = f"""
                        UPDATE segments s
                        SET embedding = v.emb
                        FROM (VALUES {','.join(values)}) AS v(sid, emb)
                        WHERE s.segment_id = v.sid
                    """
                    conn.execute(text(update_sql))
                    conn.commit()
                    embedded += len(seg_ids)

            except Exception as e:
                log.warning(f"Batch failed: {e}")
                errors += len(seg_ids)
                conn.rollback()

            elapsed = time.time() - start
            rate = embedded / elapsed if elapsed > 0 else 0
            remaining = (total - embedded) / rate if rate > 0 else 0

            if embedded % 1024 == 0 or embedded < 1000:
                log.info(f"{embedded:,}/{total:,} ({rate:.0f}/s, ~{remaining/60:.1f}min left, {errors} err)")

        elapsed = time.time() - start
        log.info(f"Done: {embedded:,} embedded in {elapsed/60:.1f}min ({embedded/max(elapsed,1):.0f}/s). Errors: {errors}")

if __name__ == "__main__":
    main()
