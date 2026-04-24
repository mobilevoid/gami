"""Chunk splitter for long segments.

Takes a ParsedSegment and splits it if token_count > 1000.
Splits on paragraph boundaries (double newline), falls back to
sentence boundaries. Target chunk size: 600-800 tokens.
"""
import logging
import re
from typing import Optional

import tiktoken

from parsers.base import ParsedSegment

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token counting (shared encoder)
# ---------------------------------------------------------------------------
_encoder: Optional[tiktoken.Encoding] = None


def _get_encoder() -> tiktoken.Encoding:
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding("cl100k_base")
    return _encoder


def _count_tokens(text: str) -> int:
    return len(_get_encoder().encode(text))


# ---------------------------------------------------------------------------
# Splitting helpers
# ---------------------------------------------------------------------------

_SENTENCE_RE = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z])"   # split after sentence-ending punct + space before capital
    r"|(?<=\n)\s*(?=\S)",        # or at line breaks
)


def _split_paragraphs(text: str) -> list[str]:
    """Split text on double-newline paragraph boundaries."""
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]


def _split_sentences(text: str) -> list[str]:
    """Split text on sentence boundaries."""
    parts = _SENTENCE_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _merge_chunks(
    pieces: list[str],
    target_min: int = 600,
    target_max: int = 800,
    hard_max: int = 1000,
) -> list[str]:
    """Greedily merge pieces into chunks within the token budget."""
    chunks: list[str] = []
    current_parts: list[str] = []
    current_tokens = 0

    for piece in pieces:
        piece_tokens = _count_tokens(piece)

        # If a single piece exceeds hard_max, it goes alone
        if piece_tokens > hard_max:
            if current_parts:
                chunks.append("\n\n".join(current_parts))
                current_parts = []
                current_tokens = 0
            chunks.append(piece)
            continue

        # Would adding this piece exceed target_max?
        combined_tokens = current_tokens + piece_tokens + (3 if current_parts else 0)
        if combined_tokens > target_max and current_parts:
            chunks.append("\n\n".join(current_parts))
            current_parts = [piece]
            current_tokens = piece_tokens
        else:
            current_parts.append(piece)
            current_tokens = combined_tokens

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

TARGET_MIN = 600
TARGET_MAX = 800
HARD_MAX = 1000


def chunk_segment(
    segment: ParsedSegment,
    threshold: int = 1000,
    target_min: int = TARGET_MIN,
    target_max: int = TARGET_MAX,
) -> list[ParsedSegment]:
    """Split a segment if its token count exceeds *threshold*.

    Returns a list of child ParsedSegment objects with correct
    char_start / char_end offsets preserved through the split.
    If the segment is small enough, returns it unchanged in a
    single-element list.
    """
    token_count = _count_tokens(segment.text)
    if token_count <= threshold:
        return [segment]

    text = segment.text

    # Try paragraph splitting first
    pieces = _split_paragraphs(text)
    if len(pieces) <= 1:
        # Fall back to sentence splitting
        pieces = _split_sentences(text)
    if len(pieces) <= 1:
        # Cannot split further — return as-is
        return [segment]

    merged = _merge_chunks(pieces, target_min, target_max, threshold)
    if len(merged) <= 1:
        return [segment]

    # Build child segments with offset tracking
    children: list[ParsedSegment] = []
    base_char = segment.char_start or 0
    base_line = segment.line_start or 0
    search_start = 0

    for idx, chunk_text in enumerate(merged):
        # Find the chunk's position in the original text
        pos = text.find(chunk_text[:80], search_start)
        if pos == -1:
            pos = search_start

        char_start = base_char + pos
        char_end = char_start + len(chunk_text)

        # Estimate line numbers
        lines_before = text[:pos].count("\n")
        lines_in_chunk = chunk_text.count("\n")

        child = ParsedSegment(
            text=chunk_text,
            segment_type=segment.segment_type,
            ordinal=segment.ordinal * 1000 + idx,  # sub-ordinals
            depth=segment.depth + 1,
            title_or_heading=segment.title_or_heading,
            char_start=char_start,
            char_end=char_end,
            line_start=base_line + lines_before,
            line_end=base_line + lines_before + lines_in_chunk,
            speaker_role=segment.speaker_role,
            speaker_name=segment.speaker_name,
            message_timestamp=segment.message_timestamp,
            metadata={
                **(segment.metadata or {}),
                "chunk_index": idx,
                "chunk_total": len(merged),
                "parent_ordinal": segment.ordinal,
                "is_chunk": True,
            },
        )
        children.append(child)
        search_start = pos + len(chunk_text[:80])

    logger.info(
        "Chunked segment ord=%d (%d tokens) into %d chunks",
        segment.ordinal,
        token_count,
        len(children),
    )
    return children


def chunk_segments(
    segments: list[ParsedSegment],
    threshold: int = 1000,
) -> list[ParsedSegment]:
    """Apply chunking to a list of segments, returning the expanded list."""
    result: list[ParsedSegment] = []
    for seg in segments:
        result.extend(chunk_segment(seg, threshold))
    return result
