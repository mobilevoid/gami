#!/usr/bin/env python3
"""Final embedding pass — short timeout, skip failures, process small first."""
import os, sys, time, logging, requests
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.config import settings
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                   handlers=[logging.FileHandler("/tmp/embed_v2.log"), logging.StreamHandler()])
log = logging.getLogger("embed")
engine = create_engine(settings.DATABASE_URL_SYNC)

OLLAMA_URL = settings.OLLAMA_URL + "/api/embeddings"
MODEL = settings.EMBEDDING_MODEL

def embed_one(txt):
    """Embed with 15s timeout, truncate to 1500 chars."""
    prompt = txt[:1500].strip()
    if not prompt:
        return None
    r = requests.post(OLLAMA_URL, json={"model": MODEL, "prompt": prompt}, timeout=15)
    r.raise_for_status()
    return r.json()["embedding"]

def run():
    with engine.connect() as conn:
        # Get all unembedded segment IDs, ordered smallest first
        rows = conn.execute(text(
            "SELECT segment_id, text FROM segments "
            "WHERE embedding IS NULL AND length(text) >= 20 "
            "ORDER BY length(text) ASC"
        )).fetchall()

        total = len(rows)
        log.info(f"Segments to embed: {total}")
        done = skipped = 0
        start = time.time()

        for sid, txt in rows:
            try:
                time.sleep(0.2)
                emb = embed_one(txt)
                if emb:
                    vec = "[" + ",".join(str(v) for v in emb) + "]"
                    conn.execute(text("UPDATE segments SET embedding = CAST(:v AS vector) WHERE segment_id = :s"),
                               {"v": vec, "s": sid})
                    done += 1
                else:
                    skipped += 1
            except Exception as e:
                skipped += 1
                if skipped <= 5:
                    log.warning(f"Skip {sid} ({len(txt)}ch): {str(e)[:60]}")

            if (done + skipped) % 10 == 0:
                conn.commit()
            if done % 100 == 0 and done > 0:
                elapsed = time.time() - start
                rate = done / elapsed
                log.info(f"{done}/{total} ({rate:.1f}/s, {skipped} skipped)")

        conn.commit()

        # Now entities
        erows = conn.execute(text(
            "SELECT entity_id, description FROM entities "
            "WHERE embedding IS NULL AND length(description) >= 10"
        )).fetchall()
        log.info(f"Entities to embed: {len(erows)}")
        edone = 0
        for eid, desc in erows:
            try:
                time.sleep(0.2)
                emb = embed_one(desc)
                if emb:
                    vec = "[" + ",".join(str(v) for v in emb) + "]"
                    conn.execute(text("UPDATE entities SET embedding = CAST(:v AS vector) WHERE entity_id = :s"),
                               {"v": vec, "s": eid})
                    edone += 1
            except:
                pass
        conn.commit()

        elapsed = time.time() - start
        log.info(f"DONE in {elapsed:.0f}s: {done} segments, {edone} entities, {skipped} skipped")

if __name__ == "__main__":
    run()
