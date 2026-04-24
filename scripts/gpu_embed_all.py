#!/usr/bin/env python3
"""GPU batch embedding for GAMI — uses sentence-transformers on CUDA.

Run with: /home/ai/.conda/envs/Training/bin/python scripts/gpu_embed_all.py

Embeds all unembedded entities, claims, and segments using nomic-embed-text
on GPU. ~100x faster than Ollama CPU.
"""
import time, logging, sys
import torch
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                   handlers=[logging.FileHandler("/tmp/gpu_embed.log"), logging.StreamHandler()])
log = logging.getLogger("gpu_embed")

# Database connection
DB_URL = "postgresql://gami:GamiProd2026@localhost:5433/gami"
engine = create_engine(DB_URL)

# Load model on GPU
log.info("Loading nomic-embed-text on GPU...")
model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
model = model.to("cuda")
log.info(f"Model loaded. GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.mem_get_info()[0]/1e9:.1f}GB free")

BATCH_SIZE = 64  # GPU can handle large batches


def embed_batch(texts):
    """Embed a batch of texts using GPU. Returns list of embedding vectors."""
    # NO prefix, NO L2 normalization — must produce same scale as Ollama nomic-embed-text.
    # Ollama returns unnormalized vectors (norm ~20-25). GPU sentence-transformers
    # normalize by default. Setting normalize=False matches Ollama's output space.
    truncated = [t[:1500] for t in texts]
    embeddings = model.encode(truncated, batch_size=BATCH_SIZE, show_progress_bar=False, normalize_embeddings=False)
    return embeddings.tolist()


def embed_table(table, id_col, text_expr, where_extra="", label=""):
    """Embed all unembedded rows in a table."""
    with engine.connect() as conn:
        total = conn.execute(text(
            f"SELECT count(*) FROM {table} WHERE embedding IS NULL {where_extra}"
        )).scalar()

        if total == 0:
            log.info(f"  {label}: all embedded already")
            return 0

        log.info(f"  {label}: {total} to embed")
        done = 0
        start = time.time()

        while True:
            rows = conn.execute(text(
                f"SELECT {id_col}, {text_expr} as txt FROM {table} "
                f"WHERE embedding IS NULL {where_extra} LIMIT :batch"
            ), {"batch": BATCH_SIZE}).fetchall()

            if not rows:
                break

            ids = [r[0] for r in rows]
            texts = [r[1] or "" for r in rows]

            # Skip empty texts
            valid = [(i, t) for i, t in zip(ids, texts) if len(t) >= 10]
            if not valid:
                # Mark empties as having NULL embedding (skip them)
                done += len(rows)
                continue

            valid_ids, valid_texts = zip(*valid)

            try:
                embeddings = embed_batch(valid_texts)

                for eid, emb in zip(valid_ids, embeddings):
                    vec = "[" + ",".join(str(v) for v in emb) + "]"
                    conn.execute(text(
                        f"UPDATE {table} SET embedding = CAST(:v AS vector) WHERE {id_col} = :id"
                    ), {"v": vec, "id": eid})

                conn.commit()
                done += len(rows)

                if done % (BATCH_SIZE * 5) == 0:
                    elapsed = time.time() - start
                    rate = done / elapsed
                    remaining = (total - done) / max(rate, 0.1) / 60
                    log.info(f"  {label}: {done}/{total} ({rate:.0f}/s, ~{remaining:.1f}min left)")

            except Exception as e:
                log.warning(f"  Batch error: {e}")
                # Fallback: embed one at a time
                for eid, etxt in zip(valid_ids, valid_texts):
                    try:
                        emb = embed_batch([etxt])[0]
                        vec = "[" + ",".join(str(v) for v in emb) + "]"
                        conn.execute(text(
                            f"UPDATE {table} SET embedding = CAST(:v AS vector) WHERE {id_col} = :id"
                        ), {"v": vec, "id": eid})
                        done += 1
                    except:
                        pass
                conn.commit()

        elapsed = time.time() - start
        log.info(f"  {label}: DONE — {done} embedded in {elapsed:.0f}s ({done/max(elapsed,1):.0f}/s)")
        return done


def main():
    log.info("=" * 50)
    log.info("GPU BATCH EMBEDDING — ALL TABLES")
    log.info("=" * 50)

    t0 = time.time()

    # 1. Entities (highest priority)
    e = embed_table("entities", "entity_id",
                    "canonical_name || ': ' || COALESCE(description, '')",
                    "AND status = 'active'", "Entities")

    # 2. Claims
    c = embed_table("claims", "claim_id",
                    "COALESCE(summary_text, predicate)",
                    "AND status = 'active'", "Claims")

    # 3. Segments (operational — already mostly done)
    s1 = embed_table("segments", "segment_id", "text",
                     "AND owner_tenant_id <> 'books' AND length(text) >= 20 "
                     "AND (quality_flags_json->>'chunked' IS NULL OR quality_flags_json->>'chunked' <> 'true')",
                     "Op Segments")

    # 4. Book segments
    s2 = embed_table("segments", "segment_id", "text",
                     "AND owner_tenant_id = 'books' AND length(text) >= 20",
                     "Book Segments")

    # 5. Summaries
    sm = embed_table("summaries", "summary_id", "summary_text",
                     "", "Summaries")

    # 6. Memories
    m = embed_table("assistant_memories", "memory_id", "normalized_text",
                    "", "Memories")

    elapsed = time.time() - t0
    log.info(f"\nALL DONE in {elapsed/60:.1f}min")
    log.info(f"  Entities: {e}, Claims: {c}, Op Segments: {s1}, Book Segments: {s2}, Summaries: {sm}, Memories: {m}")


if __name__ == "__main__":
    main()
