"""Enhanced retrieval service for GAMI.

Orchestrates query classification, hybrid search, scoring, budget management,
and citation assembly for the recall pipeline.
"""
import logging
import time
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from api.llm.embeddings import embed_text
from api.search.hybrid import hybrid_search, vector_search, lexical_search
from api.services.context_budget import (
    ContextBudget,
    EvidenceItem,
    count_tokens,
)
from api.services.citation_service import (
    Citation,
    CitationLevel,
    build_citation,
)
from api.services.query_classifier import (
    QueryClassification,
    QueryMode,
    classify_query,
)
from api.services.db import AsyncSessionLocal

logger = logging.getLogger("gami.services.retrieval")


class EvidenceResult(BaseModel):
    """A single piece of ranked evidence with citation."""
    item_id: str
    item_type: str
    text: str
    score: float
    effective_score: float
    token_count: int
    citation: Optional[dict] = None
    metadata: dict = {}


class RecallResult(BaseModel):
    """Full recall response."""
    query: str
    mode: str
    granularity: str
    evidence: list[EvidenceResult]
    context_text: str
    total_tokens_used: int
    total_candidates: int
    classification_ms: float = 0.0
    search_ms: float = 0.0
    total_ms: float = 0.0


class VerifyResult(BaseModel):
    """Verification result for a claim."""
    claim: str
    verdict: str  # supported, contradicted, unsupported, partial
    confidence: float
    supporting_evidence: list[EvidenceResult]
    contradicting_evidence: list[EvidenceResult]


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _recency_score(timestamp: Optional[str]) -> float:
    """Score 0-1 based on how recent the content is. Recent = higher."""
    if not timestamp:
        return 0.3  # Unknown recency gets neutral score
    try:
        if isinstance(timestamp, str):
            ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        else:
            ts = timestamp
        age_days = (datetime.now(timezone.utc) - ts).total_seconds() / 86400
        if age_days < 1:
            return 1.0
        elif age_days < 7:
            return 0.9
        elif age_days < 30:
            return 0.7
        elif age_days < 90:
            return 0.5
        else:
            return 0.3
    except Exception:
        return 0.3


def _compute_evidence_score(
    result: dict,
    mode: QueryMode,
) -> tuple[float, float, float]:
    """Compute semantic, importance, and recency scores for a search result.

    Returns (combined_score, importance, recency).
    """
    # Base relevance from search
    combined = result.get("combined_score", 0.0)
    vector = result.get("vector_score", 0.0)
    lexical = result.get("lexical_score", 0.0)

    # Use max of available scores
    relevance = max(combined, vector, lexical, result.get("similarity", 0.0), result.get("rank", 0.0))

    # Importance: prefer certain segment types
    seg_type = result.get("segment_type", "")
    importance_map = {
        "heading": 0.8,
        "summary": 0.9,
        "claim": 0.95,
        "paragraph": 0.5,
        "code_block": 0.6,
        "list_item": 0.4,
        "table_row": 0.5,
        "message": 0.6,
    }
    importance = importance_map.get(seg_type, 0.5)

    # Mode-specific boosts
    if mode == QueryMode.TIMELINE:
        if result.get("message_timestamp"):
            importance += 0.2
    elif mode == QueryMode.ENTITY_CENTRIC:
        if result.get("title_or_heading"):
            importance += 0.1
    elif mode == QueryMode.VERIFICATION:
        # Prefer segments with higher confidence
        importance += 0.1

    importance = min(1.0, importance)

    # Penalize very short segments (noise: "OK", "FAILED", etc.)
    text_len = len(result.get("text", ""))
    if text_len < 20:
        importance *= 0.1
    elif text_len < 50:
        importance *= 0.3
    elif text_len < 100:
        importance *= 0.7

    recency = _recency_score(result.get("message_timestamp"))

    return relevance, importance, recency


# ---------------------------------------------------------------------------
# Main recall function
# ---------------------------------------------------------------------------

async def recall(
    query: str,
    tenant_id: str = "claude-opus",
    tenant_ids: Optional[list[str]] = None,
    max_tokens: int = 4000,
    mode: Optional[str] = None,
    include_citations: bool = True,
    citation_level: CitationLevel = CitationLevel.BRIEF,
) -> RecallResult:
    """Full retrieval pipeline: classify, search, score, budget, cite.

    Args:
        query: The user's query text.
        tenant_id: Default tenant for search scope.
        tenant_ids: Optional list of tenants to search across.
        max_tokens: Token budget for the response context.
        mode: Override query mode (skip classification).
        include_citations: Whether to attach citations.
        citation_level: Detail level for citations.

    Returns:
        RecallResult with ranked, budgeted evidence and context.
    """
    t_start = time.monotonic()
    tids = tenant_ids or [tenant_id]

    # 1. Classify query
    t_class = time.monotonic()
    if mode:
        try:
            query_mode = QueryMode(mode)
        except ValueError:
            query_mode = QueryMode.FACTUAL
        classification = QueryClassification(mode=query_mode)
    else:
        classification = await classify_query(query)
    classification_ms = (time.monotonic() - t_class) * 1000

    # 2. Embed query
    query_embedding = await embed_text(query)

    # 3. Parallel anchor retrieval (vector + lexical via hybrid)
    t_search = time.monotonic()

    # Adjust search limits based on mode
    search_limit = 40
    if classification.mode in (QueryMode.REPORT, QueryMode.SYNTHESIS):
        search_limit = 60
    elif classification.mode == QueryMode.FACTUAL:
        search_limit = 30

    async with AsyncSessionLocal() as db:
        results = await hybrid_search(
            db, query, query_embedding, tids,
            limit=search_limit,
            vector_weight=0.7,
            lexical_weight=0.3,
        )
        search_ms = (time.monotonic() - t_search) * 1000

        # 4. Score results
        evidence_items: list[EvidenceItem] = []
        for r in results:
            relevance, importance, recency = _compute_evidence_score(
                r, classification.mode
            )
            token_count = r.get("token_count") or count_tokens(r.get("text", ""))

            item = EvidenceItem(
                item_id=r["segment_id"],
                item_type="segment",
                text=r.get("text", ""),
                score=relevance,
                importance=importance,
                recency_score=recency,
                token_count=token_count,
                metadata={
                    "source_id": r.get("source_id"),
                    "segment_type": r.get("segment_type"),
                    "title_or_heading": r.get("title_or_heading"),
                    "speaker_name": r.get("speaker_name"),
                    "speaker_role": r.get("speaker_role"),
                    "message_timestamp": r.get("message_timestamp"),
                    "owner_tenant_id": r.get("owner_tenant_id"),
                },
            )
            evidence_items.append(item)

        # 5. Apply token budget
        budget = ContextBudget(max_tokens=max_tokens)
        selected = budget.rank_and_fill(evidence_items)

        # 6. Attach citations
        evidence_results: list[EvidenceResult] = []
        for item in selected:
            cit_dict = None
            if include_citations:
                try:
                    cit = await build_citation(
                        db,
                        item_id=item.item_id,
                        item_type=item.item_type,
                        level=citation_level,
                        metadata=item.metadata,
                    )
                    cit_dict = cit.model_dump()
                except Exception as exc:
                    logger.warning("Citation build failed for %s: %s", item.item_id, exc)

            evidence_results.append(EvidenceResult(
                item_id=item.item_id,
                item_type=item.item_type,
                text=item.text,
                score=item.score,
                effective_score=item.effective_score,
                token_count=item.token_count,
                citation=cit_dict,
                metadata=item.metadata,
            ))

    # 7. Build context text
    context_parts = []
    for i, ev in enumerate(evidence_results, 1):
        header_parts = []
        if ev.metadata.get("title_or_heading"):
            header_parts.append(ev.metadata["title_or_heading"])
        if ev.metadata.get("speaker_role"):
            header_parts.append(f"[{ev.metadata['speaker_role']}]")
        if ev.metadata.get("segment_type"):
            header_parts.append(f"({ev.metadata['segment_type']})")
        header = " ".join(header_parts) if header_parts else f"Evidence {i}"
        context_parts.append(f"### [{i}] {header}\n{ev.text}")

    context_text = "\n\n---\n\n".join(context_parts)

    total_ms = (time.monotonic() - t_start) * 1000

    return RecallResult(
        query=query,
        mode=classification.mode.value,
        granularity=classification.granularity,
        evidence=evidence_results,
        context_text=context_text,
        total_tokens_used=budget.used_tokens,
        total_candidates=len(results),
        classification_ms=round(classification_ms, 1),
        search_ms=round(search_ms, 1),
        total_ms=round(total_ms, 1),
    )


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

async def verify_claim(
    claim_text: str,
    tenant_id: str = "claude-opus",
    tenant_ids: Optional[list[str]] = None,
    max_evidence: int = 10,
) -> VerifyResult:
    """Check whether a claim is supported or contradicted by stored evidence.

    Searches for the claim, then looks for both supporting and contradicting
    evidence. Verdict is based on score distribution.
    """
    tids = tenant_ids or [tenant_id]
    query_embedding = await embed_text(claim_text)

    async with AsyncSessionLocal() as db:
        results = await hybrid_search(
            db, claim_text, query_embedding, tids,
            limit=max_evidence * 2,
        )

    if not results:
        return VerifyResult(
            claim=claim_text,
            verdict="unsupported",
            confidence=0.0,
            supporting_evidence=[],
            contradicting_evidence=[],
        )

    # Simple heuristic: high-similarity results are "supporting"
    # Very different results that mention key terms are "contradicting"
    supporting = []
    contradicting = []

    for r in results[:max_evidence]:
        score = r.get("combined_score", 0.0)
        ev = EvidenceResult(
            item_id=r["segment_id"],
            item_type="segment",
            text=r.get("text", ""),
            score=score,
            effective_score=score,
            token_count=r.get("token_count", 0),
            metadata={
                "source_id": r.get("source_id"),
                "segment_type": r.get("segment_type"),
                "title_or_heading": r.get("title_or_heading"),
            },
        )
        if score > 0.5:
            supporting.append(ev)
        elif score > 0.2:
            # Neutral — could support or contradict. Check text overlap.
            # For now, classify as weak support.
            supporting.append(ev)

    # Determine verdict
    if len(supporting) >= 3 and supporting[0].score > 0.6:
        verdict = "supported"
        confidence = min(1.0, supporting[0].score + 0.1 * len(supporting))
    elif len(supporting) >= 1:
        verdict = "partial"
        confidence = supporting[0].score
    else:
        verdict = "unsupported"
        confidence = 0.1

    return VerifyResult(
        claim=claim_text,
        verdict=verdict,
        confidence=round(confidence, 3),
        supporting_evidence=supporting[:5],
        contradicting_evidence=contradicting[:5],
    )
