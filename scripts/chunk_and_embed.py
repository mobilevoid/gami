#!/usr/bin/env python3
"""Retroactively chunk large unembedded segments and embed the children.

Finds segments that are >1000 tokens, have no embedding, and aren't already
marked as chunked. Splits them into child segments, stores children in DB,
marks parent as chunked, and embeds the children.
"""
import os, sys, time, logging, requests, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.config import settings
from sqlalchemy import create_engine, text
from parsers.chunker import _count_tokens, _split_paragraphs, _split_sentences, _merge_chunks

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                   handlers=[logging.FileHandler("/tmp/chunk_embed.log"), logging.StreamHandler()])
log = logging.getLogger("chunk_embed")
engine = create_engine(settings.DATABASE_URL_SYNC)

OLLAMA_URL = settings.OLLAMA_URL + "/api/embeddings"
MODEL = settings.EMBEDDING_MODEL

def embed_one(txt):
    r = requests.post(OLLAMA_URL, json={"model": MODEL, "prompt": txt[:2000]}, timeout=15)
    r.raise_for_status()
    return r.json()["embedding"]


def chunk_text(text_content, threshold=1000):
    """Split text into chunks if over threshold tokens."""
    tokens = _count_tokens(text_content)
    if tokens <= threshold:
        return None  # No chunking needed

    pieces = _split_paragraphs(text_content)
    if len(pieces) <= 1:
        pieces = _split_sentences(text_content)
    if len(pieces) <= 1:
        return None  # Can't split

    merged = _merge_chunks(pieces, 600, 800, threshold)
    if len(merged) <= 1:
        return None
    return merged


def run():
    with engine.connect() as conn:
        # Find unchunked, unembedded segments
        rows = conn.execute(text("""
            SELECT segment_id, source_id, owner_tenant_id, text,
                   segment_type, title_or_heading, speaker_role, speaker_name,
                   message_timestamp, parent_segment_id, ordinal
            FROM segments
            WHERE embedding IS NULL
              AND length(text) >= 20
              AND (quality_flags_json->>'chunked' IS NULL
                   OR quality_flags_json->>'chunked' != 'true')
            ORDER BY length(text) DESC
        """)).fetchall()

        total = len(rows)
        log.info(f"Found {total} unchunked unembedded segments")

        chunked_count = 0
        children_created = 0
        children_embedded = 0
        direct_embedded = 0
        skipped = 0

        for row in rows:
            sid = row.segment_id
            txt = row.text
            tokens = _count_tokens(txt)

            if tokens <= 1000:
                # Small enough to embed directly
                try:
                    time.sleep(0.2)
                    emb = embed_one(txt)
                    vec = "[" + ",".join(str(v) for v in emb) + "]"
                    conn.execute(text(
                        "UPDATE segments SET embedding = CAST(:v AS vector) WHERE segment_id = :s"
                    ), {"v": vec, "s": sid})
                    direct_embedded += 1
                except Exception as e:
                    skipped += 1
                    if skipped <= 5:
                        log.warning(f"Embed failed {sid}: {str(e)[:60]}")
                continue

            # Chunk the segment
            chunks = chunk_text(txt)
            if not chunks:
                # Can't chunk, embed truncated version
                try:
                    time.sleep(0.2)
                    emb = embed_one(txt[:2000])
                    vec = "[" + ",".join(str(v) for v in emb) + "]"
                    conn.execute(text(
                        "UPDATE segments SET embedding = CAST(:v AS vector) WHERE segment_id = :s"
                    ), {"v": vec, "s": sid})
                    direct_embedded += 1
                except:
                    skipped += 1
                continue

            # Create child segments and embed them
            for idx, chunk in enumerate(chunks):
                child_id = f"{sid}_chunk_{idx}"

                # Check if child already exists
                exists = conn.execute(text(
                    "SELECT 1 FROM segments WHERE segment_id = :cid"
                ), {"cid": child_id}).fetchone()

                if not exists:
                    # Insert child segment
                    conn.execute(text("""
                        INSERT INTO segments (segment_id, source_id, owner_tenant_id, text,
                            segment_type, title_or_heading, speaker_role, speaker_name,
                            message_timestamp, parent_segment_id, ordinal, depth,
                            token_count, quality_flags_json)
                        VALUES (:cid, :src, :tenant, :txt, :stype, :heading, :srole, :sname,
                            :ts, :parent, :ord, 1, :tc, '{"is_chunk": true}'::jsonb)
                        ON CONFLICT (segment_id) DO NOTHING
                    """), {
                        "cid": child_id,
                        "src": row.source_id,
                        "tenant": row.owner_tenant_id,
                        "txt": chunk,
                        "stype": row.segment_type,
                        "heading": row.title_or_heading,
                        "srole": row.speaker_role,
                        "sname": row.speaker_name,
                        "ts": row.message_timestamp,
                        "parent": sid,
                        "ord": (row.ordinal or 0) * 1000 + idx,
                        "tc": _count_tokens(chunk),
                    })
                    children_created += 1

                # Embed the child
                try:
                    time.sleep(0.2)
                    emb = embed_one(chunk)
                    vec = "[" + ",".join(str(v) for v in emb) + "]"
                    conn.execute(text(
                        "UPDATE segments SET embedding = CAST(:v AS vector) WHERE segment_id = :cid"
                    ), {"v": vec, "cid": child_id})
                    children_embedded += 1
                except Exception as e:
                    if skipped <= 5:
                        log.warning(f"Child embed failed {child_id}: {str(e)[:60]}")
                    skipped += 1

            # Mark parent as chunked
            conn.execute(text("""
                UPDATE segments
                SET quality_flags_json = COALESCE(quality_flags_json, '{}'::jsonb) || '{"chunked": "true"}'::jsonb
                WHERE segment_id = :s
            """), {"s": sid})
            chunked_count += 1

            if (chunked_count + direct_embedded) % 20 == 0:
                conn.commit()
                log.info(f"Progress: {chunked_count} chunked ({children_created} children, "
                        f"{children_embedded} embedded), {direct_embedded} direct, {skipped} skipped")

        conn.commit()
        log.info(f"DONE: {chunked_count} chunked -> {children_created} children ({children_embedded} embedded), "
                f"{direct_embedded} direct embedded, {skipped} skipped")


if __name__ == "__main__":
    run()
