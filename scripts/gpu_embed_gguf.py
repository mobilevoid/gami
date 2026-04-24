import os
#!/usr/bin/env python3
"""GPU embedding using Ollama's exact GGUF model via llama-cpp-python.

This loads the same nomic-embed-text GGUF that Ollama uses, directly on GPU.
Produces identical embeddings to Ollama but at GPU speed (~500-1000/s vs ~3/s CPU).

Run with: python scripts/gpu_embed_gguf.py
"""
import time, logging
import numpy as np
from llama_cpp import Llama
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                   handlers=[logging.FileHandler("/tmp/gpu_embed_gguf.log"), logging.StreamHandler()])
log = logging.getLogger("gpu_embed")

DB_URL = os.getenv("DATABASE_URL", "postgresql://gami:gami@localhost:5432/gami")
MODEL_PATH = "/opt/gami/models/nomic-embed-text.gguf"
BATCH_SIZE = 100  # DB fetch batch
engine = create_engine(DB_URL, pool_size=3)

# Load model on GPU
log.info("Loading GGUF model on GPU...")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_gpu_layers=-1,  # All layers on GPU
    embedding=True,
    verbose=False,
)
log.info("Model loaded on GPU")


def embed_batch(texts):
    """Embed multiple texts. Returns list of embedding vectors."""
    embeddings = []
    for t in texts:
        emb = llm.embed(t[:2000])
        embeddings.append(emb)
    return embeddings


def embed_table(table, id_col, text_expr, where_extra="", label=""):
    """Embed all unembedded rows."""
    with engine.connect() as conn:
        total = conn.execute(text(
            f"SELECT count(*) FROM {table} WHERE embedding IS NULL {where_extra}"
        )).scalar()

        if total == 0:
            log.info(f"  {label}: all embedded")
            return 0

        log.info(f"  {label}: {total} to embed")
        done = skip = 0
        start = time.time()

        while True:
            rows = conn.execute(text(
                f"SELECT {id_col}, {text_expr} as txt FROM {table} "
                f"WHERE embedding IS NULL {where_extra} LIMIT :batch"
            ), {"batch": BATCH_SIZE}).fetchall()

            if not rows:
                break

            for row in rows:
                rid, txt = row[0], row[1] or ""
                if len(txt) < 10:
                    skip += 1
                    continue
                try:
                    emb = llm.embed(txt[:2000])
                    vec = "[" + ",".join(str(v) for v in emb) + "]"
                    conn.execute(text(
                        f"UPDATE {table} SET embedding = CAST(:v AS vector) WHERE {id_col} = :id"
                    ), {"v": vec, "id": rid})
                    done += 1
                except Exception as e:
                    skip += 1
                    if skip <= 3:
                        log.warning(f"  Error: {e}")

            conn.commit()

            if done % 500 == 0 and done > 0:
                elapsed = time.time() - start
                rate = done / elapsed
                remaining = (total - done - skip) / max(rate, 0.1) / 60
                log.info(f"  {label}: {done}/{total} ({rate:.1f}/s, ~{remaining:.1f}min, {skip} skip)")

        elapsed = time.time() - start
        log.info(f"  {label}: DONE — {done} in {elapsed:.0f}s ({done/max(elapsed,1):.1f}/s, {skip} skip)")
        return done


def embed_query(text):
    """Embed a query for searching GPU tenants. Call this from tenant_search."""
    return llm.embed(text[:2000])


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tenant", default="books", help="Tenant to embed")
    args = parser.parse_args()

    log.info("=" * 50)
    log.info(f"GPU GGUF EMBEDDING — tenant: {args.tenant}")
    log.info("NOTE: GPU embeddings only search within the same tenant")
    log.info("=" * 50)

    t0 = time.time()

    s = embed_table("segments", "segment_id", "text",
                    f"AND owner_tenant_id = '{args.tenant}' AND length(text) >= 20 "
                    "AND (quality_flags_json->>'chunked' IS NULL OR quality_flags_json->>'chunked' <> 'true')",
                    f"{args.tenant} Segments")

    elapsed = time.time() - t0
    log.info(f"\nDONE in {elapsed/60:.1f}min — {s} segments for '{args.tenant}'")


if __name__ == "__main__":
    main()
