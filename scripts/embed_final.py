#!/usr/bin/env python3
"""Final embedding pass — simple filter, 0.3s delay."""
import os, sys, time, logging, requests as req
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.config import settings
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                   handlers=[logging.FileHandler("/tmp/embed_final.log"), logging.StreamHandler()])
log = logging.getLogger("embed")
engine = create_engine(settings.DATABASE_URL_SYNC)

with engine.connect() as conn:
    total = conn.execute(text("SELECT count(*) FROM segments WHERE embedding IS NULL AND length(text) >= 20")).scalar()
    log.info(f"To embed: {total}")
    
    done = errors = 0
    start = time.time()
    while True:
        rows = conn.execute(text(
            "SELECT segment_id, text FROM segments WHERE embedding IS NULL AND length(text) >= 20 LIMIT 10"
        )).fetchall()
        if not rows:
            break
        for sid, txt in rows:
            time.sleep(0.3)
            try:
                r = req.post(f"{settings.OLLAMA_URL}/api/embeddings",
                           json={"model": settings.EMBEDDING_MODEL, "prompt": txt[:3000]}, timeout=30)
                if r.status_code == 200:
                    emb = r.json()["embedding"]
                    vec = "[" + ",".join(str(v) for v in emb) + "]"
                    conn.execute(text("UPDATE segments SET embedding = CAST(:v AS vector) WHERE segment_id = :s"),
                               {"v": vec, "s": sid})
                    done += 1
                else:
                    errors += 1
                    time.sleep(1)
            except:
                errors += 1
                time.sleep(1)
        conn.commit()
        elapsed = time.time() - start
        rate = done / elapsed if elapsed > 0 else 0
        remain = total - done
        if done % 100 == 0:
            log.info(f"{done}/{total} ({rate:.1f}/s, ~{remain/max(rate,0.01)/60:.0f}min, {errors} err)")
    log.info(f"DONE: {done}/{total}, {errors} err, {time.time()-start:.0f}s")
