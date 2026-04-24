#!/usr/bin/env python3
import os, sys, time, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.llm.embeddings import embed_text_sync
from api.config import settings
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("embed")
engine = create_engine(settings.DATABASE_URL_SYNC)

with engine.connect() as conn:
    total = conn.execute(text(
        "SELECT count(*) FROM segments WHERE embedding IS NULL AND "
        "(quality_flags_json->>'chunked' IS NULL OR quality_flags_json->>'chunked' != 'true')"
    )).scalar()
    log.info(f"Segments to embed: {total}")
    
    done = errors = 0
    start = time.time()
    while True:
        rows = conn.execute(text(
            "SELECT segment_id, text FROM segments WHERE embedding IS NULL AND "
            "(quality_flags_json->>'chunked' IS NULL OR quality_flags_json->>'chunked' != 'true') "
            "LIMIT 20"
        )).fetchall()
        if not rows:
            break
        for sid, txt in rows:
            try:
                time.sleep(0.2)
                emb = embed_text_sync(txt[:6000])
                vec = "[" + ",".join(str(v) for v in emb) + "]"
                conn.execute(text("UPDATE segments SET embedding = CAST(:v AS vector) WHERE segment_id = :s"),
                           {"v": vec, "s": sid})
                done += 1
            except Exception as e:
                errors += 1
                if errors <= 3:
                    log.warning(f"Error: {e}")
                if errors > 50:
                    log.error("Too many errors, stopping")
                    break
        conn.commit()
        elapsed = time.time() - start
        rate = done / elapsed if elapsed > 0 else 0
        remaining = (total - done - errors) / rate if rate > 0 else 999999
        if done % 100 == 0 or done < 30:
            log.info(f"{done}/{total} ({rate:.1f}/s, ~{remaining/60:.1f}min left, {errors} err)")
    
    log.info(f"DONE: {done} embedded, {errors} errors in {time.time()-start:.0f}s")
