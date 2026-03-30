"""Summary generation service for GAMI.

Generates summaries at various abstraction levels from segments.
Uses vLLM for summarization and Ollama for embeddings.
"""
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

from jinja2 import Environment, FileSystemLoader
from sqlalchemy import text
from sqlalchemy.orm import Session

from api.config import settings
from api.llm.client import call_vllm_sync, parse_json_from_llm
from api.llm.embeddings import embed_text_sync

logger = logging.getLogger("gami.services.summarizer")

PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "prompts")
jinja_env = Environment(loader=FileSystemLoader(PROMPTS_DIR))

SUMMARIZER_VERSION = "1.0.0"
# Max tokens to feed the summarizer (~10k tokens ~ 40k chars)
MAX_INPUT_CHARS = 40000


def _make_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def generate_summary(
    db: Session,
    scope_type: str,
    scope_id: str,
    segments: list[dict],
    tenant_id: str,
    abstraction_level: str = "source",
) -> Optional[str]:
    """Generate a summary from a list of segments.

    Args:
        db: Database session
        scope_type: "source", "topic", "session", "section"
        scope_id: ID of the scope (source_id, topic_id, etc.)
        segments: List of dicts with at least {"segment_id": ..., "text": ...}
        tenant_id: Owner tenant
        abstraction_level: "chunk", "section", "source", "topic", "session"

    Returns:
        summary_id on success, None on failure.
    """
    if not segments:
        logger.warning("No segments provided for summary generation")
        return None

    # Concatenate segment texts
    combined = "\n\n---\n\n".join(seg["text"] for seg in segments if seg.get("text"))
    if not combined.strip():
        return None

    # Truncate if needed
    if len(combined) > MAX_INPUT_CHARS:
        combined = combined[:MAX_INPUT_CHARS] + "\n\n[... truncated ...]"

    original_len = sum(len(seg.get("text", "")) for seg in segments)

    # Render prompt
    template = jinja_env.get_template("summarization.j2")
    prompt = template.render(text=combined)

    system_prompt = (
        "You are a precise summarizer. Preserve all technical details, "
        "IP addresses, credentials, version numbers, and specific identifiers. "
        "Be concise but complete."
    )

    try:
        summary_text = call_vllm_sync(
            prompt, system_prompt=system_prompt,
            max_tokens=2048, temperature=0.1,
        )
    except Exception as exc:
        logger.error("Summary generation LLM call failed: %s", exc)
        return None

    if not summary_text or not summary_text.strip():
        logger.warning("Summary generation returned empty text")
        return None

    summary_text = summary_text.strip()

    # Compute embedding
    try:
        embedding = embed_text_sync(summary_text[:8000])
        vec_str = "[" + ",".join(str(v) for v in embedding) + "]"
    except Exception as exc:
        logger.warning("Failed to embed summary: %s", exc)
        vec_str = None

    # Compression ratio
    compression = len(summary_text) / max(original_len, 1)

    # Store
    summary_id = _make_id("SUM")
    segment_ids = [seg["segment_id"] for seg in segments if seg.get("segment_id")]
    now = datetime.now(timezone.utc)

    try:
        if vec_str:
            db.execute(
                text(
                    "INSERT INTO summaries "
                    "(summary_id, owner_tenant_id, scope_type, scope_id, "
                    "abstraction_level, summary_text, embedding, "
                    "based_on_ids, quality_score, freshness_score, "
                    "compression_ratio, summarizer_version, created_at, updated_at) "
                    "VALUES (:sid, :tid, :stype, :scid, :alevel, :stext, "
                    "CAST(:vec AS vector), CAST(:based AS jsonb), "
                    "0.8, 1.0, :comp, :ver, :now, :now)"
                ),
                {
                    "sid": summary_id,
                    "tid": tenant_id,
                    "stype": scope_type,
                    "scid": scope_id,
                    "alevel": abstraction_level,
                    "stext": summary_text,
                    "vec": vec_str,
                    "based": json.dumps(segment_ids),
                    "comp": compression,
                    "ver": SUMMARIZER_VERSION,
                    "now": now,
                },
            )
        else:
            db.execute(
                text(
                    "INSERT INTO summaries "
                    "(summary_id, owner_tenant_id, scope_type, scope_id, "
                    "abstraction_level, summary_text, "
                    "based_on_ids, quality_score, freshness_score, "
                    "compression_ratio, summarizer_version, created_at, updated_at) "
                    "VALUES (:sid, :tid, :stype, :scid, :alevel, :stext, "
                    "CAST(:based AS jsonb), "
                    "0.8, 1.0, :comp, :ver, :now, :now)"
                ),
                {
                    "sid": summary_id,
                    "tid": tenant_id,
                    "stype": scope_type,
                    "scid": scope_id,
                    "alevel": abstraction_level,
                    "stext": summary_text,
                    "based": json.dumps(segment_ids),
                    "comp": compression,
                    "ver": SUMMARIZER_VERSION,
                    "now": now,
                },
            )
        db.commit()
    except Exception as exc:
        logger.error("Failed to store summary: %s", exc)
        db.rollback()
        return None

    logger.info(
        "Generated summary %s (%s/%s): %d chars → %d chars (%.1f%% compression)",
        summary_id, scope_type, scope_id,
        original_len, len(summary_text), compression * 100,
    )
    return summary_id


def summarize_source(
    db: Session,
    source_id: str,
    tenant_id: str,
    max_segments: int = 200,
) -> dict:
    """Generate hierarchical summaries for a source.

    For large sources:
    1. Group segments into chunks of ~20
    2. Summarize each chunk (abstraction_level='chunk')
    3. Summarize all chunk summaries (abstraction_level='source')

    For small sources:
    - Directly summarize all segments (abstraction_level='source')
    """
    # Fetch segments
    rows = db.execute(
        text(
            "SELECT segment_id, text, segment_type, ordinal "
            "FROM segments WHERE source_id = :sid "
            "ORDER BY ordinal NULLS LAST, created_at "
            "LIMIT :lim"
        ),
        {"sid": source_id, "lim": max_segments},
    ).fetchall()

    if not rows:
        return {"source_id": source_id, "summaries": 0, "error": "no segments found"}

    segments = [
        {"segment_id": r[0], "text": r[1], "segment_type": r[2], "ordinal": r[3]}
        for r in rows
    ]

    CHUNK_SIZE = 20
    summary_ids = []

    if len(segments) <= CHUNK_SIZE:
        # Small source — single summary
        sid = generate_summary(
            db, "source", source_id, segments, tenant_id,
            abstraction_level="source",
        )
        if sid:
            summary_ids.append(sid)
    else:
        # Large source — hierarchical
        chunk_summaries = []
        for i in range(0, len(segments), CHUNK_SIZE):
            chunk = segments[i:i + CHUNK_SIZE]
            chunk_num = i // CHUNK_SIZE
            sid = generate_summary(
                db, "source_chunk", f"{source_id}_chunk_{chunk_num}",
                chunk, tenant_id, abstraction_level="chunk",
            )
            if sid:
                summary_ids.append(sid)
                # Fetch the summary text for the next level
                row = db.execute(
                    text("SELECT summary_text FROM summaries WHERE summary_id = :sid"),
                    {"sid": sid},
                ).fetchone()
                if row:
                    chunk_summaries.append({
                        "segment_id": sid,  # Use summary_id as pseudo-segment
                        "text": row[0],
                    })

        # Source-level summary from chunk summaries
        if chunk_summaries:
            top_sid = generate_summary(
                db, "source", source_id, chunk_summaries, tenant_id,
                abstraction_level="source",
            )
            if top_sid:
                summary_ids.append(top_sid)

    return {
        "source_id": source_id,
        "segments_count": len(segments),
        "summaries_generated": len(summary_ids),
        "summary_ids": summary_ids,
    }
