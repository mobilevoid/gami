#!/usr/bin/env python3
"""Fast parallel embedding using ThreadPoolExecutor."""
import os, sys, time, logging, requests
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.config import settings
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("embed_fast")

OLLAMA_URL = settings.OLLAMA_URL
MODEL = settings.EMBEDDING_MODEL
WORKERS = 8
BATCH_DB = 100  # commit every N

def embed_one(seg_id, seg_text):
    try:
        t = seg_text[:8000] if len(seg_text) > 8000 else seg_text
        r = requests.post(f"{OLLAMA_URL}/api/embeddings",
                         json={"model": MODEL, "prompt": t}, timeout=30)
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
        
        embedded = 0
        errors = 0
        start = time.time()
        
        while True:
            rows = conn.execute(text(
                "SELECT segment_id, text FROM segments WHERE embedding IS NULL ORDER BY created_at LIMIT 500"
            )).fetchall()
            
            if not rows:
                break
            
            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                futures = {pool.submit(embed_one, r[0], r[1]): r[0] for r in rows}
                batch_updates = []
                
                for f in as_completed(futures):
                    seg_id, emb = f.result()
                    if emb:
                        vec = "[" + ",".join(str(v) for v in emb) + "]"
                        batch_updates.append((seg_id, vec))
                        embedded += 1
                    else:
                        errors += 1
                    
                    if len(batch_updates) >= BATCH_DB:
                        for sid, vec in batch_updates:
                            conn.execute(text("UPDATE segments SET embedding = CAST(:v AS vector) WHERE segment_id = :s"),
                                       {"v": vec, "s": sid})
                        conn.commit()
                        batch_updates = []
                        
                        elapsed = time.time() - start
                        rate = embedded / elapsed if elapsed > 0 else 0
                        remaining = (total - embedded) / rate if rate > 0 else 0
                        log.info(f"Progress: {embedded}/{total} ({rate:.0f}/s, ~{remaining:.0f}s remaining)")
                
                # Flush remaining
                for sid, vec in batch_updates:
                    conn.execute(text("UPDATE segments SET embedding = CAST(:v AS vector) WHERE segment_id = :s"),
                               {"v": vec, "s": sid})
                conn.commit()
        
        elapsed = time.time() - start
        log.info(f"Done: {embedded} embedded, {errors} errors in {elapsed:.1f}s ({embedded/elapsed:.0f}/s)")

if __name__ == "__main__":
    main()
