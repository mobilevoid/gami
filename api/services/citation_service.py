"""Citation service for GAMI retrieval orchestrator.

Builds structured citations for evidence items, supporting brief,
full, and drill_down levels with provenance chain tracing.
"""
import logging
from enum import Enum
from typing import Optional

from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger("gami.services.citation_service")


class CitationLevel(str, Enum):
    BRIEF = "brief"
    FULL = "full"
    DRILL_DOWN = "drill_down"


class Citation(BaseModel):
    """Structured citation for an evidence item."""
    item_id: str
    item_type: str  # segment, claim, summary, entity, memory
    source_title: Optional[str] = None
    source_id: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    speaker_name: Optional[str] = None
    speaker_role: Optional[str] = None
    timestamp: Optional[str] = None
    confidence: Optional[float] = None
    provenance_chain: Optional[list[dict]] = None
    formatted: str = ""


async def build_citation(
    db: AsyncSession,
    item_id: str,
    item_type: str,
    level: CitationLevel = CitationLevel.BRIEF,
    metadata: Optional[dict] = None,
) -> Citation:
    """Build a citation for a single evidence item.

    Args:
        db: Async database session.
        item_id: The ID of the item (segment_id, claim_id, etc.).
        item_type: Type of evidence (segment, claim, summary, entity, memory).
        level: Citation detail level.
        metadata: Pre-fetched metadata to avoid extra DB calls.

    Returns:
        Citation with formatted text and structured fields.
    """
    metadata = metadata or {}
    citation = Citation(item_id=item_id, item_type=item_type)

    if item_type == "segment":
        await _cite_segment(db, citation, level, metadata)
    elif item_type == "claim":
        await _cite_claim(db, citation, level, metadata)
    elif item_type == "summary":
        await _cite_summary(db, citation, level, metadata)
    elif item_type == "entity":
        await _cite_entity(db, citation, level, metadata)
    elif item_type == "memory":
        await _cite_memory(db, citation, level, metadata)
    else:
        citation.formatted = f"[{item_type}:{item_id}]"

    return citation


async def build_citations(
    db: AsyncSession,
    items: list[dict],
    level: CitationLevel = CitationLevel.BRIEF,
) -> list[Citation]:
    """Build citations for a list of evidence items."""
    citations = []
    for item in items:
        cit = await build_citation(
            db,
            item_id=item["item_id"],
            item_type=item["item_type"],
            level=level,
            metadata=item.get("metadata", {}),
        )
        citations.append(cit)
    return citations


async def get_provenance_chain(
    db: AsyncSession,
    target_id: str,
    target_type: str,
    max_depth: int = 3,
) -> list[dict]:
    """Trace the provenance chain for an item: summary -> claim -> segment -> source."""
    chain = []
    current_id = target_id
    current_type = target_type

    for _ in range(max_depth):
        result = await db.execute(
            text("""
                SELECT p.provenance_id, p.source_id, p.segment_id,
                       p.extraction_method, p.confidence,
                       p.page_start, p.page_end, p.line_start, p.line_end,
                       p.speaker_name, p.quote_text,
                       s.title AS source_title, s.source_type
                FROM provenance p
                LEFT JOIN sources s ON s.source_id = p.source_id
                WHERE p.target_id = :tid AND p.target_type = :ttype
                ORDER BY p.created_at DESC
                LIMIT 1
            """),
            {"tid": current_id, "ttype": current_type},
        )
        row = result.fetchone()
        if not row:
            break

        link = {
            "provenance_id": row[0],
            "source_id": row[1],
            "segment_id": row[2],
            "extraction_method": row[3],
            "confidence": row[4],
            "page_start": row[5],
            "page_end": row[6],
            "line_start": row[7],
            "line_end": row[8],
            "speaker_name": row[9],
            "quote_text": row[10],
            "source_title": row[11],
            "source_type": row[12],
        }
        chain.append(link)

        # Follow to segment if we have one
        if row[2] and current_type != "segment":
            current_id = row[2]
            current_type = "segment"
        else:
            break

    return chain


# ---------------------------------------------------------------------------
# Internal citation builders
# ---------------------------------------------------------------------------

async def _cite_segment(
    db: AsyncSession,
    citation: Citation,
    level: CitationLevel,
    metadata: dict,
):
    """Build citation for a segment."""
    # Use pre-fetched metadata if available
    source_title = metadata.get("source_title")
    source_id = metadata.get("source_id")
    speaker = metadata.get("speaker_name") or metadata.get("speaker_role")
    timestamp = metadata.get("message_timestamp")
    page_start = metadata.get("page_start")
    line_start = metadata.get("line_start")

    if not source_title and source_id:
        result = await db.execute(
            text("SELECT title FROM sources WHERE source_id = :sid"),
            {"sid": source_id},
        )
        row = result.fetchone()
        if row:
            source_title = row[0]

    if not source_title:
        # Fetch from segment join
        result = await db.execute(
            text("""
                SELECT s.source_id, src.title, s.speaker_name, s.speaker_role,
                       s.message_timestamp, s.page_start, s.line_start
                FROM segments s
                LEFT JOIN sources src ON src.source_id = s.source_id
                WHERE s.segment_id = :sid
            """),
            {"sid": citation.item_id},
        )
        row = result.fetchone()
        if row:
            source_id = row[0]
            source_title = row[1]
            speaker = row[2] or row[3]
            timestamp = row[4].isoformat() if row[4] else None
            page_start = row[5]
            line_start = row[6]

    citation.source_title = source_title
    citation.source_id = source_id
    citation.speaker_name = speaker
    citation.timestamp = timestamp
    citation.page_start = page_start
    citation.line_start = line_start

    # Format
    parts = []
    if source_title:
        parts.append(source_title)
    if page_start:
        parts.append(f"p.{page_start}")
    if line_start:
        parts.append(f"L{line_start}")
    if speaker:
        parts.append(f"({speaker})")
    if timestamp and level != CitationLevel.BRIEF:
        parts.append(f"@{timestamp}")

    citation.formatted = " ".join(parts) if parts else f"[seg:{citation.item_id[:12]}]"

    if level == CitationLevel.DRILL_DOWN:
        citation.provenance_chain = await get_provenance_chain(
            db, citation.item_id, "segment"
        )


async def _cite_claim(
    db: AsyncSession,
    citation: Citation,
    level: CitationLevel,
    metadata: dict,
):
    """Build citation for a claim, tracing back to source segment."""
    result = await db.execute(
        text("""
            SELECT c.summary_text, c.confidence, c.modality,
                   p.source_id, p.segment_id, p.speaker_name,
                   src.title AS source_title
            FROM claims c
            LEFT JOIN provenance p ON p.target_id = c.claim_id AND p.target_type = 'claim'
            LEFT JOIN sources src ON src.source_id = p.source_id
            WHERE c.claim_id = :cid
            LIMIT 1
        """),
        {"cid": citation.item_id},
    )
    row = result.fetchone()
    if not row:
        citation.formatted = f"[claim:{citation.item_id[:12]}]"
        return

    citation.source_title = row[6]
    citation.source_id = row[3]
    citation.speaker_name = row[5]
    citation.confidence = row[1]

    parts = []
    if row[6]:
        parts.append(row[6])
    if row[5]:
        parts.append(f"({row[5]})")
    parts.append(f"[conf:{row[1]:.0%}]")

    citation.formatted = " ".join(parts) if parts else f"[claim:{citation.item_id[:12]}]"

    if level == CitationLevel.DRILL_DOWN:
        citation.provenance_chain = await get_provenance_chain(
            db, citation.item_id, "claim"
        )


async def _cite_summary(
    db: AsyncSession,
    citation: Citation,
    level: CitationLevel,
    metadata: dict,
):
    """Build citation for a summary."""
    result = await db.execute(
        text("""
            SELECT scope_type, scope_id, abstraction_level, quality_score
            FROM summaries WHERE summary_id = :sid
        """),
        {"sid": citation.item_id},
    )
    row = result.fetchone()
    if not row:
        citation.formatted = f"[summary:{citation.item_id[:12]}]"
        return

    citation.formatted = f"[Summary of {row[0]}:{row[1][:20]} ({row[2]})]"
    citation.confidence = row[3]


async def _cite_entity(
    db: AsyncSession,
    citation: Citation,
    level: CitationLevel,
    metadata: dict,
):
    """Build citation for an entity."""
    result = await db.execute(
        text("""
            SELECT canonical_name, entity_type, mention_count
            FROM entities WHERE entity_id = :eid
        """),
        {"eid": citation.item_id},
    )
    row = result.fetchone()
    if not row:
        citation.formatted = f"[entity:{citation.item_id[:12]}]"
        return

    citation.formatted = f"{row[0]} ({row[1]}, {row[2]} mentions)"


async def _cite_memory(
    db: AsyncSession,
    citation: Citation,
    level: CitationLevel,
    metadata: dict,
):
    """Build citation for an assistant memory."""
    result = await db.execute(
        text("""
            SELECT memory_type, subject_id, importance_score,
                   confirmation_count, created_at
            FROM assistant_memories WHERE memory_id = :mid
        """),
        {"mid": citation.item_id},
    )
    row = result.fetchone()
    if not row:
        citation.formatted = f"[memory:{citation.item_id[:12]}]"
        return

    citation.formatted = (
        f"[Memory: {row[0]}/{row[1]} "
        f"(importance:{row[2]:.1f}, confirmed:{row[3]}x)]"
    )
    citation.confidence = row[2]
    citation.timestamp = row[4].isoformat() if row[4] else None
