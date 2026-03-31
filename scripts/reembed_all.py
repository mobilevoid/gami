#!/usr/bin/env python3
"""Re-embed everything with sentence-transformers nomic-embed-text-v1.5.

Run with: /home/ai/.conda/envs/gami-embed/bin/python scripts/reembed_all.py

Identical on CPU and GPU. Replaces all Ollama embeddings.
"""
import time, logging
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                   handlers=[logging.FileHandler("/tmp/reembed_all.log"), logging.StreamHandler()])
log = logging.getLogger("reembed")

DB_URL = "postgresql://gami:GamiProd2026@localhost:5433/gami"
MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
BATCH_SIZE = 128

engine = create_engine(DB_URL, pool_size=3)

device = "cuda" if torch.cuda.is_available() else "cpu"
log.info(f"Loading {MODEL_NAME} on {device}...")
model = SentenceTransformer(MODEL_NAME, trust_remote_code=True, device=device)
log.info("Model loaded")


def embed_batch(texts):
    truncated = [t[:2000] for t in texts]
    return model.encode(truncated, normalize_embeddings=False, show_progress_bar=False, batch_size=BATCH_SIZE)


def reembed_table(table, id_col, text_expr, where_extra="", label=""):
    with engine.connect() as conn:
        # NULL all existing embeddings first
        conn.execute(text(f"UPDATE {table} SET embedding = NULL {where_extra.replace('AND', 'WHERE', 1) if where_extra else ''}"))
        conn.commit()

        total = conn.execute(text(
            f"SELECT count(*) FROM {table} WHERE embedding IS NULL {where_extra}"
        )).scalar()

        if total == 0:
            log.info(f"  {label}: nothing to embed")
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

            ids = [r[0] for r in rows]
            texts_list = [r[1] or "" for r in rows]

            valid = [(i, t) for i, t in zip(ids, texts_list) if len(t) >= 10]
            if not valid:
                skip += len(rows)
                # Mark short texts so we don't loop forever
                for rid, _ in zip(ids, texts_list):
                    conn.execute(text(
                        f"UPDATE {table} SET embedding = CAST(:v AS vector) WHERE {id_col} = :id"
                    ), {"v": "[" + ",".join(["0"] * 768) + "]", "id": rid})
                conn.commit()
                continue

            valid_ids, valid_texts = zip(*valid)
            embs = embed_batch(valid_texts)

            for eid, emb in zip(valid_ids, embs):
                vec = "[" + ",".join(str(v) for v in emb) + "]"
                conn.execute(text(
                    f"UPDATE {table} SET embedding = CAST(:v AS vector) WHERE {id_col} = :id"
                ), {"v": vec, "id": eid})
            conn.commit()
            done += len(valid_ids)
            skip += len(rows) - len(valid_ids)

            if done % (BATCH_SIZE * 4) == 0 and done > 0:
                elapsed = time.time() - start
                rate = done / elapsed
                remaining = (total - done - skip) / max(rate, 0.1) / 60
                log.info(f"  {label}: {done}/{total} ({rate:.0f}/s, ~{remaining:.1f}min, {skip} skip)")

        elapsed = time.time() - start
        log.info(f"  {label}: DONE — {done} in {elapsed:.0f}s ({done/max(elapsed,1):.0f}/s)")
        return done


def main():
    log.info("=" * 60)
    log.info(f"RE-EMBEDDING EVERYTHING — {MODEL_NAME} on {device}")
    log.info(f"Batch size: {BATCH_SIZE}")
    log.info("=" * 60)

    t0 = time.time()

    m = reembed_table("assistant_memories", "memory_id", "normalized_text", "", "Memories")
    sm = reembed_table("summaries", "summary_id", "summary_text", "", "Summaries")
    c = reembed_table("claims", "claim_id", "COALESCE(summary_text, predicate)",
                      "AND status='active'", "Claims")
    e = reembed_table("entities", "entity_id",
                      "canonical_name || ': ' || COALESCE(description, '')",
                      "AND status='active'", "Entities")

    # Ops segments first
    s1 = reembed_table("segments", "segment_id", "text",
                       "AND owner_tenant_id <> 'books' AND length(text) >= 20 "
                       "AND (quality_flags_json->>'chunked' IS NULL OR quality_flags_json->>'chunked' <> 'true')",
                       "Op Segments")

    # Book segments
    s2 = reembed_table("segments", "segment_id", "text",
                       "AND owner_tenant_id = 'books' AND length(text) >= 20 "
                       "AND (quality_flags_json->>'chunked' IS NULL OR quality_flags_json->>'chunked' <> 'true')",
                       "Book Segments")

    elapsed = time.time() - t0
    log.info(f"\nALL DONE in {elapsed/60:.1f}min")
    log.info(f"  Memories: {m}, Summaries: {sm}, Claims: {c}, Entities: {e}")
    log.info(f"  Op Segments: {s1}, Book Segments: {s2}")
    log.info(f"  Total: {m+sm+c+e+s1+s2} embeddings at {(m+sm+c+e+s1+s2)/max(elapsed,1):.0f}/s")


if __name__ == "__main__":
    main()
