#!/usr/bin/env python3
"""Chunk oversized segments, then embed all unembedded segments."""
import os, sys, time, logging, requests, uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.config import settings
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("chunk_embed")

MAX_EMBED_CHARS = 6000  # ~1500 tokens, safe for nomic-embed-text (8192 token limit)
WORKERS = 4

def chunk_text(full_text, max_chars=MAX_EMBED_CHARS):
    """Split text into chunks at paragraph boundaries."""
    if len(full_text) <= max_chars:
        return [full_text]
    
    chunks = []
    paragraphs = full_text.split('\n\n')
    current = ""
    
    for para in paragraphs:
        if len(current) + len(para) + 2 > max_chars and current:
            chunks.append(current.strip())
            current = para
        else:
            current = current + "\n\n" + para if current else para
    
    if current.strip():
        chunks.append(current.strip())
    
    # If any chunk is still too large, split on newlines
    final = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            final.append(chunk)
        else:
            lines = chunk.split('\n')
            sub = ""
            for line in lines:
                if len(sub) + len(line) + 1 > max_chars and sub:
                    final.append(sub.strip())
                    sub = line
                else:
                    sub = sub + "\n" + line if sub else line
            if sub.strip():
                final.append(sub.strip())
    
    return final if final else [full_text[:max_chars]]

def embed_one(seg_id, seg_text):
    try:
        r = requests.post(f"{settings.OLLAMA_URL}/api/embeddings",
                         json={"model": settings.EMBEDDING_MODEL, "prompt": seg_text}, timeout=60)
        if r.status_code == 200:
            return seg_id, r.json()["embedding"]
    except:
        pass
    return seg_id, None

def main():
    engine = create_engine(settings.DATABASE_URL_SYNC, pool_size=5)
    
    with engine.connect() as conn:
        # Step 1: Chunk oversized segments that don't have children yet
        log.info("Step 1: Chunking oversized segments...")
        oversized = conn.execute(text("""
            SELECT s.segment_id, s.text, s.source_id, s.owner_tenant_id, s.segment_type
            FROM segments s
            WHERE length(s.text) > :max_chars
            AND s.embedding IS NULL
            AND NOT EXISTS (
                SELECT 1 FROM segments c WHERE c.parent_segment_id = s.segment_id
            )
        """), {"max_chars": MAX_EMBED_CHARS}).fetchall()
        
        log.info(f"Found {len(oversized)} oversized segments to chunk")
        chunks_created = 0
        
        for row in oversized:
            seg_id, full_text, source_id, tenant_id, seg_type = row
            chunks = chunk_text(full_text)
            
            if len(chunks) <= 1:
                continue  # Already fits
            
            for i, chunk_text_str in enumerate(chunks):
                chunk_id = f"{seg_id}_chunk_{i}"
                char_start = full_text.find(chunk_text_str[:100])
                char_end = char_start + len(chunk_text_str) if char_start >= 0 else None
                
                conn.execute(text("""
                    INSERT INTO segments (segment_id, source_id, owner_tenant_id, parent_segment_id,
                        segment_type, ordinal, depth, text, token_count, char_start, char_end,
                        quality_flags_json, storage_tier)
                    VALUES (:sid, :src, :tid, :parent, 'chunk', :ord, 1, :txt, :tc,
                        :cs, :ce, '{"is_chunk": true}'::jsonb, 'hot')
                    ON CONFLICT (segment_id) DO NOTHING
                """), {
                    "sid": chunk_id, "src": source_id, "tid": tenant_id,
                    "parent": seg_id, "ord": i, "txt": chunk_text_str,
                    "tc": len(chunk_text_str) // 4,  # rough token estimate
                    "cs": char_start if char_start >= 0 else None,
                    "ce": char_end,
                })
                chunks_created += 1
            
            # Mark parent as "chunked" — we'll embed children instead
            conn.execute(text("""
                UPDATE segments SET quality_flags_json = quality_flags_json || '{"chunked": true}'::jsonb
                WHERE segment_id = :sid
            """), {"sid": seg_id})
        
        conn.commit()
        log.info(f"Created {chunks_created} chunk segments from {len(oversized)} parents")
        
        # Step 2: Embed all segments that need it (skip chunked parents, embed their children)
        total = conn.execute(text("""
            SELECT count(*) FROM segments 
            WHERE embedding IS NULL 
            AND (quality_flags_json->>'chunked' IS NULL OR quality_flags_json->>'chunked' = 'false')
        """)).scalar()
        log.info(f"Step 2: Embedding {total} segments...")
        
        embedded = errors = 0
        start = time.time()
        
        while True:
            rows = conn.execute(text("""
                SELECT segment_id, text FROM segments 
                WHERE embedding IS NULL 
                AND (quality_flags_json->>'chunked' IS NULL OR quality_flags_json->>'chunked' = 'false')
                LIMIT 100
            """)).fetchall()
            
            if not rows:
                break
            
            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                futures = {pool.submit(embed_one, r[0], r[1]): r[0] for r in rows}
                for f in as_completed(futures):
                    sid, emb = f.result()
                    if emb:
                        vec = "[" + ",".join(str(v) for v in emb) + "]"
                        conn.execute(text(
                            "UPDATE segments SET embedding = CAST(:v AS vector) WHERE segment_id = :s"
                        ), {"v": vec, "s": sid})
                        embedded += 1
                    else:
                        errors += 1
            
            conn.commit()
            elapsed = time.time() - start
            rate = embedded / elapsed if elapsed > 0 else 0
            remaining = (total - embedded) / rate if rate > 0 else 0
            if embedded % 500 == 0 or embedded < 200:
                log.info(f"{embedded}/{total} ({rate:.0f}/s, ~{remaining/60:.1f}min left, {errors} err)")
        
        elapsed = time.time() - start
        
        # Final stats
        final_embedded = conn.execute(text("SELECT count(*) FROM segments WHERE embedding IS NOT NULL")).scalar()
        final_total = conn.execute(text("SELECT count(*) FROM segments")).scalar()
        log.info(f"Done: {embedded} new embeds in {elapsed:.0f}s. Total: {final_embedded}/{final_total} embedded.")

if __name__ == "__main__":
    main()
