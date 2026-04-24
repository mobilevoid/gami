#!/usr/bin/env python3
"""Parallel embedding using ThreadPoolExecutor + Ollama single API."""
import os, sys, time, logging, requests
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.config import settings
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("embed")

def embed_one(seg_id, seg_text):
    try:
        t = seg_text[:8000] if len(seg_text) > 8000 else seg_text
        r = requests.post(f"{settings.OLLAMA_URL}/api/embeddings",
                         json={"model": settings.EMBEDDING_MODEL, "prompt": t}, timeout=30)
        if r.status_code == 200:
            return seg_id, r.json()["embedding"]
    except:
        pass
    return seg_id, None

def main():
    engine = create_engine(settings.DATABASE_URL_SYNC, pool_size=5)
    with engine.connect() as conn:
        total = conn.execute(text("SELECT count(*) FROM segments WHERE embedding IS NULL")).scalar()
        log.info(f"Segments to embed: {total}")
        if total == 0:
            return
        embedded = errors = 0
        start = time.time()
        while True:
            rows = conn.execute(text(
                "SELECT segment_id, text FROM segments WHERE embedding IS NULL LIMIT 200"
            )).fetchall()
            if not rows:
                break
            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = {pool.submit(embed_one, r[0], r[1]): r[0] for r in rows}
                for f in as_completed(futures):
                    sid, emb = f.result()
                    if emb:
                        vec = "[" + ",".join(str(v) for v in emb) + "]"
                        conn.execute(text("UPDATE segments SET embedding = CAST(:v AS vector) WHERE segment_id = :s"),
                                   {"v": vec, "s": sid})
                        embedded += 1
                    else:
                        errors += 1
            conn.commit()
            elapsed = time.time() - start
            rate = embedded / elapsed if elapsed > 0 else 0
            remaining = (total - embedded) / rate if rate > 0 else 0
            log.info(f"{embedded}/{total} ({rate:.0f}/s, ~{remaining/60:.1f}min left, {errors} err)")
        log.info(f"Done: {embedded} in {time.time()-start:.0f}s ({embedded/(time.time()-start):.0f}/s)")

if __name__ == "__main__":
    main()
