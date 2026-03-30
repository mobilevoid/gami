"""Conversation topic detector for GAMI.

Groups conversation messages into topic episodes using simple heuristics:
- Long pauses between messages (>30 min gap)
- Explicit topic change markers
- Major entity change in consecutive messages
"""
import logging
import re
from datetime import datetime, timedelta
from typing import Optional

from parsers.base import ParsedSegment

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Gap in minutes that signals a topic shift
PAUSE_THRESHOLD_MINUTES = 30

# Phrases that explicitly signal a topic change
TOPIC_CHANGE_MARKERS = [
    r"\bnow let'?s\b",
    r"\bmoving on to\b",
    r"\bcan you look at\b",
    r"\blet'?s switch to\b",
    r"\blet'?s move on\b",
    r"\bnext,?\s+(let'?s|we|I)\b",
    r"\bchanging topic\b",
    r"\bon a different note\b",
    r"\bseparately\b",
    r"\banother thing\b",
    r"\blet'?s talk about\b",
    r"\bnew task\b",
    r"\blet me ask\b",
    r"\bactually,?\s+(can|could|let)\b",
    r"\bBuild Phase\b",
    r"\bPhase \d+\b",
    r"\bI need you to\b",
]

_TOPIC_PATTERN = re.compile(
    "|".join(TOPIC_CHANGE_MARKERS), re.IGNORECASE
)

# Simple entity extraction: capitalized multi-word names, paths, IPs
_ENTITY_RE = re.compile(
    r"\b(?:CT\d{3}|/[\w/.-]{4,}|(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+))\b"
    r"|(?:\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"
)


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def _extract_entities(text: str) -> set[str]:
    """Extract rough entity mentions from text."""
    return set(_ENTITY_RE.findall(text[:2000]))  # cap to avoid huge scans


def _has_topic_marker(text: str) -> bool:
    """Check if text contains an explicit topic-change phrase."""
    return bool(_TOPIC_PATTERN.search(text[:500]))  # only check start


def _time_gap_minutes(
    ts1: Optional[datetime], ts2: Optional[datetime]
) -> Optional[float]:
    """Return minutes between two timestamps, or None if either is missing."""
    if ts1 is None or ts2 is None:
        return None
    delta = abs((ts2 - ts1).total_seconds()) / 60.0
    return delta


def _entity_overlap(set_a: set[str], set_b: set[str]) -> float:
    """Jaccard similarity between two entity sets."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_topic_episodes(
    segments: list[ParsedSegment],
    pause_minutes: float = PAUSE_THRESHOLD_MINUTES,
    entity_threshold: float = 0.1,
) -> list[ParsedSegment]:
    """Group conversation segments into topic episodes.

    Returns a list of parent 'topic_episode' segments, each containing
    child message references in their metadata. The original segments
    are NOT modified.

    Parameters
    ----------
    segments : list[ParsedSegment]
        Conversation segments, ordered by ordinal.
    pause_minutes : float
        Minutes of silence that triggers a topic boundary.
    entity_threshold : float
        Jaccard similarity below this = major entity change.

    Returns
    -------
    list[ParsedSegment]
        Topic episode parent segments with metadata pointing to children.
    """
    if not segments:
        return []

    # Filter to content-bearing segments (messages, tool_calls)
    content_segs = [
        s for s in segments
        if s.segment_type in ("message", "tool_call", "tool_result", "thinking", "system_context")
    ]
    if not content_segs:
        return []

    # Detect boundaries
    boundaries: list[int] = [0]  # first segment always starts a topic
    prev_entities = _extract_entities(content_segs[0].text)

    for i in range(1, len(content_segs)):
        is_boundary = False
        prev_seg = content_segs[i - 1]
        curr_seg = content_segs[i]

        # Check 1: time gap
        gap = _time_gap_minutes(prev_seg.message_timestamp, curr_seg.message_timestamp)
        if gap is not None and gap > pause_minutes:
            is_boundary = True
            logger.debug(
                "Topic boundary at ord=%d: %.0f min gap", curr_seg.ordinal, gap
            )

        # Check 2: explicit topic marker (only in user messages)
        if not is_boundary and curr_seg.speaker_role in ("user", "human"):
            if _has_topic_marker(curr_seg.text):
                is_boundary = True
                logger.debug(
                    "Topic boundary at ord=%d: explicit marker", curr_seg.ordinal
                )

        # Check 3: major entity change
        if not is_boundary:
            curr_entities = _extract_entities(curr_seg.text)
            if prev_entities and curr_entities:
                overlap = _entity_overlap(prev_entities, curr_entities)
                if overlap < entity_threshold:
                    is_boundary = True
                    logger.debug(
                        "Topic boundary at ord=%d: entity overlap=%.2f",
                        curr_seg.ordinal,
                        overlap,
                    )
                prev_entities = curr_entities
            elif curr_entities:
                prev_entities = curr_entities

        if is_boundary:
            boundaries.append(i)

    # Build topic episode segments
    episodes: list[ParsedSegment] = []
    for ep_idx in range(len(boundaries)):
        start_idx = boundaries[ep_idx]
        end_idx = boundaries[ep_idx + 1] if ep_idx + 1 < len(boundaries) else len(content_segs)

        episode_segs = content_segs[start_idx:end_idx]
        if not episode_segs:
            continue

        # Build summary title from first user message in the episode
        title = _derive_episode_title(episode_segs, ep_idx)

        # Combine text for the episode (truncated for storage)
        combined_texts = []
        child_ordinals = []
        for s in episode_segs:
            role_prefix = f"[{s.speaker_role or 'unknown'}] " if s.speaker_role else ""
            combined_texts.append(f"{role_prefix}{s.text[:500]}")
            child_ordinals.append(s.ordinal)

        episode_text = "\n---\n".join(combined_texts)
        # Cap at ~4000 chars for the episode summary
        if len(episode_text) > 4000:
            episode_text = episode_text[:4000] + "\n[...truncated]"

        first_ts = episode_segs[0].message_timestamp
        last_ts = episode_segs[-1].message_timestamp

        episode = ParsedSegment(
            text=episode_text,
            segment_type="topic_episode",
            ordinal=ep_idx,
            depth=0,
            title_or_heading=title,
            char_start=episode_segs[0].char_start,
            char_end=episode_segs[-1].char_end,
            line_start=episode_segs[0].line_start,
            line_end=episode_segs[-1].line_end,
            speaker_role=None,
            speaker_name=None,
            message_timestamp=first_ts,
            metadata={
                "episode_index": ep_idx,
                "child_ordinals": child_ordinals,
                "message_count": len(episode_segs),
                "start_timestamp": first_ts.isoformat() if first_ts else None,
                "end_timestamp": last_ts.isoformat() if last_ts else None,
                "is_topic_episode": True,
            },
        )
        episodes.append(episode)

    logger.info(
        "Detected %d topic episodes from %d content segments",
        len(episodes),
        len(content_segs),
    )
    return episodes


def _derive_episode_title(segs: list[ParsedSegment], ep_idx: int) -> str:
    """Create a short title for a topic episode."""
    # Use first user message as basis
    for s in segs:
        if s.speaker_role in ("user", "human") and s.text:
            # Take first line, truncated
            first_line = s.text.split("\n")[0].strip()
            if len(first_line) > 80:
                first_line = first_line[:77] + "..."
            return first_line
    return f"Topic Episode {ep_idx + 1}"
