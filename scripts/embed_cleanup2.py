#!/usr/bin/env python3
"""Robust embedding pass — skips permanently failing segments."""
import os, sys, time, logging, requests as req
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.config import settings
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                   handlers=[logging.FileHandler("/tmp/embed_cleanup2.log"), logging.StreamHandler()])
log = logging.getLogger("embed")
engine = create_engine(settings.DATABASE_URL_SYNC)

FAILED_IDS = set()

def embed(txt):
    # Truncate to 2000 chars max for Ollama stability
    r = req.post(f"{settings.OLLAMA_URL}/api/embeddings",
                 json={"model": settings.EMBEDDING_MODEL, "prompt": txt[:2000]}, timeout=60)
    r.raise_for_status()
    return r.json()["embedding"]

def embed_table(conn, table, id_col, text_col, label):
    total = conn.execute(text(
        f"SELECT count(*) FROM {table} WHERE embedding IS NULL AND length({text_col}) >= 20"
    )).scalar()
    log.info(f"{label}: {total} to embed")
    done = errors = skipped = 0
    while True:
        rows = conn.execute(text(
            f"SELECT {id_col}, {text_col} FROM {table} WHERE embedding IS NULL AND length({text_col}) >= 20 LIMIT 10"
        )).fetchall()
        if not rows:
            break
        all_failed = True
        for rid, txt in rows:
            if rid in FAILED_IDS:
                skipped += 1
                continue
            time.sleep(0.3)
            try:
                emb = embed(txt)
                vec = "[" + ",".join(str(v) for v in emb) + "]"
                conn.execute(text(f"UPDATE {table} SET embedding = CAST(:v AS vector) WHERE {id_col} = :s"),
                           {"v": vec, "s": rid})
                done += 1
                all_failed = False
            except Exception as e:
                errors += 1
                FAILED_IDS.add(rid)
                if errors <= 10:
                    log.warning(f"Skipping {rid} ({len(txt)} chars): {str(e)[:80]}")
                time.sleep(1)
        conn.commit()
        if all_failed and len(FAILED_IDS) > 200:
            log.error("Too many permanent failures, stopping")
            break
        if done % 100 == 0 and done > 0:
            log.info(f"{label}: {done}/{total} done, {errors} failed, {skipped} skipped")
    log.info(f"{label} DONE: {done}/{total}, {errors} failed, {skipped} skipped")
    return done, errors

with engine.connect() as conn:
    start = time.time()
    s_done, s_err = embed_table(conn, "segments", "segment_id", "text", "Segments")
    e_done, e_err = embed_table(conn, "entities", "entity_id", "description", "Entities")
    elapsed = time.time() - start
    log.info(f"ALL DONE in {elapsed:.0f}s: segments={s_done}, entities={e_done}")
