import os
#!/usr/bin/env python3
"""Fast Ollama embedding for ALL GAMI tables — parallel requests.

Uses concurrent.futures to send multiple Ollama requests at once.
Ollama handles parallelism internally with NUM_PARALLEL setting.

Run: python3 scripts/ollama_embed_all.py
"""
import time, logging, requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                   handlers=[logging.FileHandler("/tmp/ollama_embed_all.log"), logging.StreamHandler()])
log = logging.getLogger("embed")

DB_URL = os.getenv("DATABASE_URL", "postgresql://gami:gami@localhost:5432/gami")
OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL = "nomic-embed-text"
WORKERS = 12  # Parallel Ollama requests
BATCH_DB = 100  # DB batch size

engine = create_engine(DB_URL, pool_size=5)


def embed_one(txt):
    """Embed a single text via Ollama."""
    r = requests.post(OLLAMA_URL, json={"model": MODEL, "prompt": txt[:2000]}, timeout=30)
    r.raise_for_status()
    return r.json()["embedding"]


def embed_table(table, id_col, text_expr, where_extra="", label=""):
    """Embed all unembedded rows using parallel Ollama requests."""
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
            ), {"batch": BATCH_DB}).fetchall()

            if not rows:
                break

            # Parallel embed
            futures = {}
            with ThreadPoolExecutor(max_workers=WORKERS) as executor:
                for row in rows:
                    rid, txt = row[0], row[1] or ""
                    if len(txt) < 10:
                        skip += 1
                        continue
                    futures[executor.submit(embed_one, txt)] = rid

                for future in as_completed(futures):
                    rid = futures[future]
                    try:
                        emb = future.result()
                        vec = "[" + ",".join(str(v) for v in emb) + "]"
                        conn.execute(text(
                            f"UPDATE {table} SET embedding = CAST(:v AS vector) WHERE {id_col} = :id"
                        ), {"v": vec, "id": rid})
                        done += 1
                    except Exception:
                        skip += 1

            conn.commit()

            if done % 500 == 0 and done > 0:
                elapsed = time.time() - start
                rate = done / elapsed
                remaining = (total - done - skip) / max(rate, 0.1) / 60
                log.info(f"  {label}: {done}/{total} ({rate:.1f}/s, ~{remaining:.1f}min, {skip} skip)")

        elapsed = time.time() - start
        log.info(f"  {label}: DONE — {done} in {elapsed:.0f}s ({done/max(elapsed,1):.1f}/s, {skip} skip)")
        return done


def main():
    log.info("=" * 50)
    log.info("OLLAMA PARALLEL EMBEDDING — ALL TABLES")
    log.info(f"Workers: {WORKERS}, DB batch: {BATCH_DB}")
    log.info("=" * 50)

    t0 = time.time()

    # Priority order: operational data FIRST, then reference/books
    m = embed_table("assistant_memories", "memory_id", "normalized_text", "", "Memories")
    sm = embed_table("summaries", "summary_id", "summary_text", "", "Summaries")
    c = embed_table("claims", "claim_id", "COALESCE(summary_text, predicate)", "AND status='active'", "Claims")
    e = embed_table("entities", "entity_id",
                    "canonical_name || ': ' || COALESCE(description, '')",
                    "AND status='active'", "Entities")

    # Operational segments first (conversations, configs, infrastructure)
    s1 = embed_table("segments", "segment_id", "text",
                     "AND owner_tenant_id <> 'books' AND length(text) >= 20 "
                     "AND (quality_flags_json->>'chunked' IS NULL OR quality_flags_json->>'chunked' <> 'true')",
                     "Op Segments")

    # Book segments last (reference tier, lower priority)
    s2 = embed_table("segments", "segment_id", "text",
                     "AND owner_tenant_id = 'books' AND length(text) >= 20 "
                     "AND (quality_flags_json->>'chunked' IS NULL OR quality_flags_json->>'chunked' <> 'true')",
                     "Book Segments")

    elapsed = time.time() - t0
    log.info(f"\nALL DONE in {elapsed/60:.1f}min")
    log.info(f"  Memories: {m}, Summaries: {sm}, Claims: {c}, Entities: {e}, Op Segs: {s1}, Book Segs: {s2}")


if __name__ == "__main__":
    main()
