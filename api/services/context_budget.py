"""Context budget manager for GAMI retrieval orchestrator.

Manages token budgets for recall responses, prioritizing high-value
evidence and deduplicating across result types.
"""
import logging
from typing import Optional

import tiktoken

logger = logging.getLogger("gami.services.context_budget")

# Use cl100k_base (GPT-4 tokenizer) as a reasonable general-purpose counter
_enc: Optional[tiktoken.Encoding] = None


def _get_encoder() -> tiktoken.Encoding:
    global _enc
    if _enc is None:
        _enc = tiktoken.get_encoding("cl100k_base")
    return _enc


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken cl100k_base."""
    if not text:
        return 0
    try:
        return len(_get_encoder().encode(text))
    except Exception:
        # Fallback: ~4 chars per token
        return len(text) // 4


# Priority order for evidence types (lower = higher priority)
PRIORITY_ORDER = {
    "claim": 0,
    "summary": 1,
    "entity_brief": 2,
    "memory": 3,
    "segment": 4,
}

# Citation overhead per item (approximate tokens for header/metadata)
CITATION_OVERHEAD = 25


class EvidenceItem:
    """A single piece of evidence for the context budget."""

    __slots__ = (
        "item_id", "item_type", "text", "score", "importance",
        "recency_score", "token_count", "metadata",
    )

    def __init__(
        self,
        item_id: str,
        item_type: str,
        text: str,
        score: float = 0.0,
        importance: float = 0.5,
        recency_score: float = 0.5,
        token_count: Optional[int] = None,
        metadata: Optional[dict] = None,
    ):
        self.item_id = item_id
        self.item_type = item_type
        self.text = text
        self.score = score
        self.importance = importance
        self.recency_score = recency_score
        self.token_count = token_count or count_tokens(text)
        self.metadata = metadata or {}

    @property
    def priority(self) -> int:
        return PRIORITY_ORDER.get(self.item_type, 5)

    @property
    def effective_score(self) -> float:
        """Combined score weighting relevance, importance, and priority."""
        priority_boost = max(0, 5 - self.priority) * 0.1
        return (
            0.5 * self.score
            + 0.2 * self.importance
            + 0.1 * self.recency_score
            + 0.2 * priority_boost
        )


class ContextBudget:
    """Manages token budget for recall context assembly."""

    def __init__(self, max_tokens: int = 4000):
        self.max_tokens = max_tokens
        self.used_tokens = 0
        self.items: list[EvidenceItem] = []
        self._seen_ids: set[str] = set()
        self._seen_texts: set[int] = set()  # hash of text for dedup

    @property
    def remaining(self) -> int:
        return max(0, self.max_tokens - self.used_tokens)

    def deduplicate(self, candidates: list[EvidenceItem]) -> list[EvidenceItem]:
        """Remove duplicate items by ID and by near-identical text."""
        unique: list[EvidenceItem] = []
        for item in candidates:
            if item.item_id in self._seen_ids:
                continue
            # Simple text dedup: hash first 200 chars
            text_hash = hash(item.text[:200].strip().lower())
            if text_hash in self._seen_texts:
                continue
            unique.append(item)
        return unique

    def rank_and_fill(self, candidates: list[EvidenceItem]) -> list[EvidenceItem]:
        """Rank candidates by effective score, fill budget, return selected items.

        Items are sorted by: priority tier, then effective_score descending.
        If a segment is too large, it gets truncated to fit remaining budget.
        """
        # Deduplicate
        candidates = self.deduplicate(candidates)

        # Sort: primary by priority (lower = better), secondary by effective score (higher = better)
        candidates.sort(key=lambda x: (x.priority, -x.effective_score))

        selected: list[EvidenceItem] = []

        for item in candidates:
            cost = item.token_count + CITATION_OVERHEAD
            if cost <= self.remaining:
                # Fits fully
                self._accept(item)
                selected.append(item)
            elif self.remaining > CITATION_OVERHEAD + 50:
                # Truncate to fit
                avail = self.remaining - CITATION_OVERHEAD
                truncated = truncate_to_tokens(item.text, avail)
                item.text = truncated
                item.token_count = count_tokens(truncated)
                self._accept(item)
                selected.append(item)
                break  # Budget essentially full after truncation
            else:
                break  # No room

        return selected

    def _accept(self, item: EvidenceItem):
        """Accept an item into the budget."""
        self._seen_ids.add(item.item_id)
        self._seen_texts.add(hash(item.text[:200].strip().lower()))
        self.used_tokens += item.token_count + CITATION_OVERHEAD
        self.items.append(item)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to fit within max_tokens, preserving word boundaries."""
    if not text:
        return text
    enc = _get_encoder()
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    truncated_tokens = tokens[:max_tokens]
    result = enc.decode(truncated_tokens)
    # Try to end at a sentence or word boundary
    last_period = result.rfind(".")
    last_newline = result.rfind("\n")
    cut = max(last_period, last_newline)
    if cut > len(result) * 0.6:
        result = result[: cut + 1]
    else:
        last_space = result.rfind(" ")
        if last_space > len(result) * 0.8:
            result = result[:last_space]
        result += "..."
    return result
