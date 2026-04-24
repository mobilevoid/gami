"""Segment storage and retrieval service for GAMI.

Bridges between the parser layer (parsers.base.ParsedSegment) and the
database segments table.  Handles token counting, parent-child hierarchy,
and both async (FastAPI) and sync (Celery) DB sessions.
"""
import json
import logging
from datetime import datetime, timezone
from typing import Optional

import tiktoken
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from parsers.base import ParsedSegment

logger = logging.getLogger("gami.segment_service")

# Global encoder — cl100k_base is a reasonable proxy for token counts.
_encoder: Optional[tiktoken.Encoding] = None


def _get_encoder() -> tiktoken.Encoding:
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding("cl100k_base")
    return _encoder


def _count_tokens(text_str: str) -> int:
    """Count tokens using tiktoken cl100k_base."""
    return len(_get_encoder().encode(text_str))


def _make_segment_id(source_id: str, ordinal: int) -> str:
    """Generate segment ID: SEG_{source_id}_{ordinal}."""
    return f"SEG_{source_id}_{ordinal}"


def _build_insert_params(
    seg_id: str,
    source_id: str,
    tenant_id: str,
    seg: ParsedSegment,
    parent_id: Optional[str],
    now: datetime,
) -> dict:
    """Build the parameter dict for a single segment INSERT."""
    token_count = _count_tokens(seg.text)
    # Extract quality_flags from metadata if present
    quality_flags = seg.metadata.get("quality_flags", {}) if seg.metadata else {}

    return {
        "seg_id": seg_id,
        "source_id": source_id,
        "tenant_id": tenant_id,
        "parent_id": parent_id,
        "seg_type": seg.segment_type,
        "ordinal": seg.ordinal,
        "depth": seg.depth,
        "title": seg.title_or_heading,
        "text": seg.text,
        "tokens": token_count,
        "char_start": seg.char_start,
        "char_end": seg.char_end,
        "line_start": seg.line_start,
        "line_end": seg.line_end,
        "speaker_role": seg.speaker_role,
        "speaker_name": seg.speaker_name,
        "msg_ts": seg.message_timestamp,
        "lang": "en",
        "qf": json.dumps(quality_flags),
        "now": now,
    }


_INSERT_SQL = text("""
    INSERT INTO segments (
        segment_id, source_id, owner_tenant_id,
        parent_segment_id, segment_type, ordinal, depth,
        title_or_heading, text, token_count,
        char_start, char_end, line_start, line_end,
        speaker_role, speaker_name, message_timestamp,
        language, quality_flags_json, storage_tier, created_at
    ) VALUES (
        :seg_id, :source_id, :tenant_id,
        :parent_id, :seg_type, :ordinal, :depth,
        :title, :text, :tokens,
        :char_start, :char_end, :line_start, :line_end,
        :speaker_role, :speaker_name, :msg_ts,
        :lang, CAST(:qf AS jsonb), 'hot', :now
    )
""")


def _resolve_parents(
    segments: list[ParsedSegment],
    ordinal_to_id: dict[int, str],
) -> dict[int, Optional[str]]:
    """
    Map each segment's ordinal to its parent_segment_id (or None).

    Parent is determined by the 'parent_ordinal' key in segment metadata,
    which is set by parsers that track heading hierarchy.
    """
    result: dict[int, Optional[str]] = {}
    for seg in segments:
        parent_ord = None
        if seg.metadata:
            parent_ord = seg.metadata.get("parent_ordinal")
        if parent_ord is not None and parent_ord in ordinal_to_id:
            result[seg.ordinal] = ordinal_to_id[parent_ord]
        else:
            result[seg.ordinal] = None
    return result


# ---------------------------------------------------------------------------
# Async interface (FastAPI)
# ---------------------------------------------------------------------------

async def store_segments(
    db: AsyncSession,
    source_id: str,
    tenant_id: str,
    parsed_segments: list[ParsedSegment],
) -> list[str]:
    """
    Store parsed segments into the segments table (async).

    Returns list of segment_ids.
    """
    if not parsed_segments:
        return []

    now = datetime.now(timezone.utc)

    # Pre-compute all segment IDs
    ordinal_to_id: dict[int, str] = {}
    segment_ids: list[str] = []
    for seg in parsed_segments:
        sid = _make_segment_id(source_id, seg.ordinal)
        ordinal_to_id[seg.ordinal] = sid
        segment_ids.append(sid)

    parent_map = _resolve_parents(parsed_segments, ordinal_to_id)

    for i, seg in enumerate(parsed_segments):
        params = _build_insert_params(
            seg_id=segment_ids[i],
            source_id=source_id,
            tenant_id=tenant_id,
            seg=seg,
            parent_id=parent_map.get(seg.ordinal),
            now=now,
        )
        await db.execute(_INSERT_SQL, params)

    await db.commit()
    logger.info(
        "Stored %d segments for source %s (tenant %s)",
        len(segment_ids), source_id, tenant_id,
    )
    return segment_ids


# ---------------------------------------------------------------------------
# Sync interface (Celery workers)
# ---------------------------------------------------------------------------

def store_segments_sync(
    db: Session,
    source_id: str,
    tenant_id: str,
    parsed_segments: list[ParsedSegment],
) -> list[str]:
    """
    Store parsed segments into the segments table (sync, for Celery).

    Returns list of segment_ids.
    """
    if not parsed_segments:
        return []

    now = datetime.now(timezone.utc)

    ordinal_to_id: dict[int, str] = {}
    segment_ids: list[str] = []
    for seg in parsed_segments:
        sid = _make_segment_id(source_id, seg.ordinal)
        ordinal_to_id[seg.ordinal] = sid
        segment_ids.append(sid)

    parent_map = _resolve_parents(parsed_segments, ordinal_to_id)

    for i, seg in enumerate(parsed_segments):
        params = _build_insert_params(
            seg_id=segment_ids[i],
            source_id=source_id,
            tenant_id=tenant_id,
            seg=seg,
            parent_id=parent_map.get(seg.ordinal),
            now=now,
        )
        db.execute(_INSERT_SQL, params)

    db.commit()
    logger.info(
        "Stored %d segments for source %s (tenant %s) [sync]",
        len(segment_ids), source_id, tenant_id,
    )
    return segment_ids


# ---------------------------------------------------------------------------
# Query interface
# ---------------------------------------------------------------------------

async def get_segments(db: AsyncSession, source_id: str) -> list[dict]:
    """Get all segments for a source, ordered by ordinal."""
    result = await db.execute(
        text(
            "SELECT segment_id, segment_type, ordinal, depth, "
            "title_or_heading, text, token_count, parent_segment_id, "
            "speaker_role, speaker_name, char_start, char_end, "
            "line_start, line_end, created_at "
            "FROM segments WHERE source_id = :sid ORDER BY ordinal"
        ),
        {"sid": source_id},
    )
    return [dict(r._mapping) for r in result.fetchall()]


async def get_segment(db: AsyncSession, segment_id: str) -> Optional[dict]:
    """Get a single segment by ID."""
    result = await db.execute(
        text(
            "SELECT segment_id, source_id, owner_tenant_id, segment_type, "
            "ordinal, depth, title_or_heading, text, token_count, "
            "parent_segment_id, speaker_role, speaker_name, "
            "char_start, char_end, line_start, line_end, created_at "
            "FROM segments WHERE segment_id = :sid"
        ),
        {"sid": segment_id},
    )
    row = result.fetchone()
    if not row:
        return None
    return dict(row._mapping)
