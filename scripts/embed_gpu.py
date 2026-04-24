#!/usr/bin/env python3
"""Fast GPU batch embedding using sentence-transformers."""
import os, sys, time, logging, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.config import settings
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("embed_gpu")

BATCH_SIZE = 32  # GPU can handle large batches
DB_COMMIT_SIZE = 500

def main():
    log.info("Loading nomic-embed-text on GPU...")
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    model = model.to("cuda")
    log.info(f"Model loaded. GPU memory: {torch.cuda.memory_allocated()/1e9:.1f}GB")
    
    engine = create_engine(settings.DATABASE_URL_SYNC, pool_size=3)
    
    with engine.connect() as conn:
        total = conn.execute(text("""
            SELECT count(*) FROM segments 
            WHERE embedding IS NULL 
            AND (quality_flags_json->>'chunked' IS NULL OR quality_flags_json->>'chunked' != 'true')
        """)).scalar()
        log.info(f"Segments to embed: {total}")
        
        if total == 0:
            return
        
        embedded = errors = 0
        start = time.time()
        
        while True:
            rows = conn.execute(text("""
                SELECT segment_id, text FROM segments 
                WHERE embedding IS NULL 
                AND (quality_flags_json->>'chunked' IS NULL OR quality_flags_json->>'chunked' != 'true')
                LIMIT :lim
            """), {"lim": BATCH_SIZE}).fetchall()
            
            if not rows:
                break
            
            seg_ids = [r[0] for r in rows]
            texts = [r[1][:8000] for r in rows]  # nomic handles up to 8192 tokens
            
            # Prefix for nomic-embed-text search documents
            prefixed = ["search_document: " + t for t in texts]
            
            try:
                embeddings = model.encode(prefixed, batch_size=BATCH_SIZE, show_progress_bar=False, 
                                         normalize_embeddings=True)
                
                for sid, emb in zip(seg_ids, embeddings):
                    vec = "[" + ",".join(str(float(v)) for v in emb) + "]"
                    conn.execute(text(
                        "UPDATE segments SET embedding = CAST(:v AS vector) WHERE segment_id = :s"
                    ), {"v": vec, "s": sid})
                    embedded += 1
                
                conn.commit()
                
            except Exception as e:
                log.warning(f"Batch failed: {e}")
                errors += len(seg_ids)
                conn.rollback()
            
            elapsed = time.time() - start
            rate = embedded / elapsed if elapsed > 0 else 0
            remaining = (total - embedded) / rate if rate > 0 else 0
            log.info(f"{embedded}/{total} ({rate:.0f}/s, ~{remaining:.0f}s left, {errors} err)")
        
        elapsed = time.time() - start
        final = conn.execute(text("SELECT count(*) FROM segments WHERE embedding IS NOT NULL")).scalar()
        log.info(f"Done: {embedded} embedded in {elapsed:.0f}s ({embedded/max(elapsed,1):.0f}/s). Total embedded: {final}")

if __name__ == "__main__":
    main()
