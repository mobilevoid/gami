"""Enhanced retrieval service for GAMI.

Orchestrates query classification, hybrid search, scoring, budget management,
and citation assembly for the recall pipeline.
"""
import asyncio
import logging
import math
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
from api.services.hot_cache import search_hot_cache

logger = logging.getLogger("gami.services.retrieval")


# ---------------------------------------------------------------------------
# Access tracking
# ---------------------------------------------------------------------------

async def _track_access(segment_ids: list[str]):
    """Background task: increment retrieval_count and update last_retrieved_at."""
    if not segment_ids:
        return
    try:
        async with AsyncSessionLocal() as db:
            await db.execute(
                text("""
                    UPDATE segments
                    SET retrieval_count = COALESCE(retrieval_count, 0) + 1,
                        last_retrieved_at = NOW()
                    WHERE segment_id = ANY(:ids)
                """),
                {"ids": segment_ids},
            )
            await db.commit()
    except Exception as exc:
        logger.warning("Access tracking failed: %s", exc)


def _hot_score(retrieval_count: int, last_retrieved_at) -> float:
    """Compute hot score from access frequency and recency."""
    if retrieval_count is None or retrieval_count == 0:
        return 0.0
    days_since = 30.0  # default if no timestamp
    if last_retrieved_at:
        try:
            if isinstance(last_retrieved_at, str):
                ts = datetime.fromisoformat(last_retrieved_at.replace("Z", "+00:00"))
            else:
                ts = last_retrieved_at
            days_since = max(0.0, (datetime.now(timezone.utc) - ts).total_seconds() / 86400)
        except Exception:
            pass
    recency_factor = 1.0 / (1.0 + days_since * 0.1)
    return math.log(retrieval_count + 1) * recency_factor


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
    # Always include 'shared' tenant so shared knowledge is searchable
    if "shared" not in tids:
        tids = list(tids) + ["shared"]

    # Load knowledge tiers for all searched tenants
    _tenant_tiers = {}
    try:
        async with AsyncSessionLocal() as _db:
            _tier_rows = await _db.execute(
                text("SELECT tenant_id, config_json->>'knowledge_tier' as tier FROM tenants WHERE tenant_id = ANY(:tids)"),
                {"tids": tids},
            )
            for _r in _tier_rows:
                _tenant_tiers[_r.tenant_id] = _r.tier or "operational"
    except Exception:
        pass

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

    # 2. Check Redis hot cache first (fast path)
    hot_hits = []
    try:
        for tid in tids:
            hot_hits.extend(search_hot_cache(tid, query, limit=5))
    except Exception as exc:
        logger.debug("Hot cache check failed (non-critical): %s", exc)

    # 3. Embed query
    query_embedding = await embed_text(query, is_query=False)
    vec_str = "[" + ",".join(str(v) for v in query_embedding) + "]"

    # 4. Multi-source retrieval: memories + entities + claims + segments
    t_search = time.monotonic()

    search_limit = 40
    if classification.mode in (QueryMode.REPORT, QueryMode.SYNTHESIS):
        search_limit = 60
    elif classification.mode == QueryMode.FACTUAL:
        search_limit = 30

    evidence_items: list[EvidenceItem] = []

    async with AsyncSessionLocal() as db:
        # ---- 4a. Search assistant_memories (vector + keyword hybrid) ----
        try:
            mem_rows = await db.execute(
                text("""
                    WITH vec_match AS (
                        SELECT memory_id, normalized_text, memory_type, subject_id,
                               importance_score, sensitivity, stability_score,
                               1 - (embedding <=> CAST(:vec AS vector)) as similarity,
                               0.0 as lex_score
                        FROM assistant_memories
                        WHERE embedding IS NOT NULL
                        AND owner_tenant_id = ANY(:tids)
                        AND status IN ('active', 'confirmed', 'provisional')
                        ORDER BY embedding <=> CAST(:vec AS vector)
                        LIMIT 15
                    ),
                    lex_match AS (
                        SELECT memory_id, normalized_text, memory_type, subject_id,
                               importance_score, sensitivity, stability_score,
                               0.0 as similarity,
                               ts_rank(to_tsvector('english', normalized_text || ' ' || subject_id),
                                       websearch_to_tsquery('english', :query)) as lex_score
                        FROM assistant_memories
                        WHERE owner_tenant_id = ANY(:tids)
                        AND status IN ('active', 'confirmed', 'provisional')
                        AND to_tsvector('english', normalized_text || ' ' || subject_id)
                            @@ websearch_to_tsquery('english', :query)
                        LIMIT 15
                    )
                    SELECT memory_id, normalized_text, memory_type, subject_id,
                           importance_score, sensitivity, stability_score,
                           MAX(similarity) as similarity, MAX(lex_score) as lex_score
                    FROM (SELECT * FROM vec_match UNION ALL SELECT * FROM lex_match) combined
                    GROUP BY memory_id, normalized_text, memory_type, subject_id,
                             importance_score, sensitivity, stability_score
                    ORDER BY GREATEST(MAX(similarity), MAX(lex_score)) DESC
                    LIMIT 20
                """),
                {"vec": vec_str, "tids": tids, "query": query},
            )
            for row in mem_rows:
                sim = float(row.similarity)
                lex = float(row.lex_score) if row.lex_score else 0.0
                # Combine vector and lexical: use weighted max
                # ts_rank returns small values (0.01-0.1), normalize to 0-1 range
                lex_normalized = min(1.0, lex * 10.0) if lex > 0 else 0.0
                combined = max(sim, lex_normalized * 0.7) if lex_normalized > 0 else sim
                if combined < 0.2:
                    continue
                # Memories get a 1.5x boost over raw segments
                boosted_score = combined * 1.5
                tc = count_tokens(row.normalized_text)
                evidence_items.append(EvidenceItem(
                    item_id=row.memory_id,
                    item_type="memory",
                    text=row.normalized_text,
                    score=boosted_score,
                    importance=float(row.importance_score) if row.importance_score else 0.8,
                    recency_score=0.9,
                    token_count=tc,
                    metadata={
                        "memory_type": row.memory_type,
                        "subject": row.subject_id,
                        "sensitivity": row.sensitivity,
                        "source_type": "assistant_memory",
                        "knowledge_tier": _tenant_tiers.get(row.owner_tenant_id, "operational") if hasattr(row, 'owner_tenant_id') else "operational",
                    },
                ))
        except Exception as exc:
            logger.warning("Memory search failed: %s", exc)

        # ---- 4b. Search entities (vector + keyword hybrid) ----
        try:
            ent_rows = await db.execute(
                text("""
                    WITH vec_match AS (
                        SELECT entity_id, canonical_name, entity_type, description,
                               aliases_json, importance_score,
                               1 - (embedding <=> CAST(:vec AS vector)) as similarity,
                               0.0 as lex_score
                        FROM entities
                        WHERE embedding IS NOT NULL
                        AND owner_tenant_id = ANY(:tids)
                        AND status IN ('active', 'provisional')
                        ORDER BY embedding <=> CAST(:vec AS vector)
                        LIMIT 10
                    ),
                    lex_match AS (
                        SELECT entity_id, canonical_name, entity_type, description,
                               aliases_json, importance_score,
                               0.0 as similarity,
                               ts_rank(to_tsvector('english',
                                   canonical_name || ' ' || COALESCE(description, '') || ' ' ||
                                   COALESCE(aliases_json::text, '')),
                                   websearch_to_tsquery('english', :query)) as lex_score
                        FROM entities
                        WHERE owner_tenant_id = ANY(:tids)
                        AND status IN ('active', 'provisional')
                        AND to_tsvector('english',
                            canonical_name || ' ' || COALESCE(description, '') || ' ' ||
                            COALESCE(aliases_json::text, ''))
                            @@ websearch_to_tsquery('english', :query)
                        LIMIT 10
                    )
                    SELECT entity_id, canonical_name, entity_type, description,
                           aliases_json, importance_score,
                           MAX(similarity) as similarity, MAX(lex_score) as lex_score
                    FROM (SELECT * FROM vec_match UNION ALL SELECT * FROM lex_match) combined
                    GROUP BY entity_id, canonical_name, entity_type, description,
                             aliases_json, importance_score
                    ORDER BY GREATEST(MAX(similarity), MAX(lex_score)) DESC
                    LIMIT 15
                """),
                {"vec": vec_str, "tids": tids, "query": query},
            )
            for row in ent_rows:
                sim = float(row.similarity)
                lex = float(row.lex_score) if row.lex_score else 0.0
                lex_normalized = min(1.0, lex * 10.0) if lex > 0 else 0.0
                combined = max(sim, lex_normalized * 0.7) if lex_normalized > 0 else sim
                if combined < 0.2:
                    continue
                entity_text = f"{row.canonical_name} ({row.entity_type}): {row.description or ''}"
                # Entities get a 1.3x boost
                boosted_score = combined * 1.3
                tc = count_tokens(entity_text)
                evidence_items.append(EvidenceItem(
                    item_id=row.entity_id,
                    item_type="entity",
                    text=entity_text,
                    score=boosted_score,
                    importance=float(row.importance_score) if row.importance_score else 0.7,
                    recency_score=0.8,
                    token_count=tc,
                    metadata={
                        "entity_type": row.entity_type,
                        "canonical_name": row.canonical_name,
                        "source_type": "entity",
                        "knowledge_tier": _tenant_tiers.get(row.owner_tenant_id, "operational") if hasattr(row, 'owner_tenant_id') else "operational",
                    },
                ))
        except Exception as exc:
            logger.warning("Entity search failed: %s", exc)

        # ---- 4c. Search claims (vector + keyword hybrid) ----
        try:
            clm_rows = await db.execute(
                text("""
                    WITH vec_match AS (
                        SELECT claim_id, summary_text, predicate, confidence,
                               object_literal_json, subject_entity_id,
                               1 - (embedding <=> CAST(:vec AS vector)) as similarity,
                               0.0 as lex_score
                        FROM claims
                        WHERE embedding IS NOT NULL
                        AND owner_tenant_id = ANY(:tids)
                        AND status = 'active'
                        ORDER BY embedding <=> CAST(:vec AS vector)
                        LIMIT 10
                    ),
                    lex_match AS (
                        SELECT claim_id, summary_text, predicate, confidence,
                               object_literal_json, subject_entity_id,
                               0.0 as similarity,
                               ts_rank(to_tsvector('english',
                                   COALESCE(summary_text, '') || ' ' || predicate || ' ' ||
                                   COALESCE(object_literal_json::text, '')),
                                   websearch_to_tsquery('english', :query)) as lex_score
                        FROM claims
                        WHERE owner_tenant_id = ANY(:tids)
                        AND status = 'active'
                        AND to_tsvector('english',
                            COALESCE(summary_text, '') || ' ' || predicate || ' ' ||
                            COALESCE(object_literal_json::text, ''))
                            @@ websearch_to_tsquery('english', :query)
                        LIMIT 10
                    )
                    SELECT claim_id, summary_text, predicate, confidence,
                           object_literal_json, subject_entity_id,
                           MAX(similarity) as similarity, MAX(lex_score) as lex_score
                    FROM (SELECT * FROM vec_match UNION ALL SELECT * FROM lex_match) combined
                    GROUP BY claim_id, summary_text, predicate, confidence,
                             object_literal_json, subject_entity_id
                    ORDER BY GREATEST(MAX(similarity), MAX(lex_score)) DESC
                    LIMIT 15
                """),
                {"vec": vec_str, "tids": tids, "query": query},
            )
            for row in clm_rows:
                sim = float(row.similarity)
                lex = float(row.lex_score) if row.lex_score else 0.0
                lex_normalized = min(1.0, lex * 10.0) if lex > 0 else 0.0
                combined = max(sim, lex_normalized * 0.7) if lex_normalized > 0 else sim
                if combined < 0.2:
                    continue
                claim_text = row.summary_text or f"{row.predicate}: {row.object_literal_json}"
                # Claims get a 1.4x boost
                boosted_score = combined * 1.4
                tc = count_tokens(claim_text)
                evidence_items.append(EvidenceItem(
                    item_id=row.claim_id,
                    item_type="claim",
                    text=claim_text,
                    score=boosted_score,
                    importance=float(row.confidence) if row.confidence else 0.8,
                    recency_score=0.8,
                    token_count=tc,
                    metadata={
                        "predicate": row.predicate,
                        "confidence": float(row.confidence) if row.confidence else 0,
                        "source_type": "claim",
                        "knowledge_tier": _tenant_tiers.get(row.owner_tenant_id, "operational") if hasattr(row, 'owner_tenant_id') else "operational",
                    },
                ))
        except Exception as exc:
            logger.warning("Claim search failed: %s", exc)

        # ---- 4c2. Graph expansion: follow relations from found entities ----
        found_entity_ids = [item.item_id for item in evidence_items if item.item_type == "entity"]
        if found_entity_ids:
            try:
                rel_rows = await db.execute(
                    text("""
                        SELECT DISTINCT e.entity_id, e.canonical_name, e.entity_type,
                               e.description, r.relation_type,
                               1 - (e.embedding <=> CAST(:vec AS vector)) as similarity
                        FROM relations r
                        JOIN entities e ON (
                            (r.to_node_id = e.entity_id AND r.from_node_id = ANY(:eids))
                            OR (r.from_node_id = e.entity_id AND r.to_node_id = ANY(:eids))
                        )
                        WHERE e.embedding IS NOT NULL
                        AND e.status = 'active'
                        AND e.entity_id != ALL(:eids)
                        AND r.status = 'active'
                        ORDER BY similarity DESC
                        LIMIT 5
                    """),
                    {"vec": vec_str, "eids": found_entity_ids},
                )
                for row in rel_rows:
                    sim = float(row.similarity) if row.similarity else 0.0
                    if sim < 0.2:
                        continue
                    entity_text = f"{row.canonical_name} ({row.entity_type}): {row.description or ''} [related via {row.relation_type}]"
                    evidence_items.append(EvidenceItem(
                        item_id=row.entity_id,
                        item_type="entity",
                        text=entity_text,
                        score=sim * 1.1,  # Slight boost for graph-connected
                        importance=0.6,
                        recency_score=0.7,
                        token_count=count_tokens(entity_text),
                        metadata={
                            "entity_type": row.entity_type,
                            "canonical_name": row.canonical_name,
                            "source_type": "graph_expansion",
                            "relation_type": row.relation_type,
                            "knowledge_tier": _tenant_tiers.get("claude-opus", "operational"),
                        },
                    ))
            except Exception as exc:
                logger.debug("Graph expansion failed (non-critical): %s", exc)

        # ---- 4d. Search segments (standard hybrid search) ----
        results = await hybrid_search(
            db, query, query_embedding, tids,
            limit=search_limit,
            vector_weight=0.7,
            lexical_weight=0.3,
        )
        search_ms = (time.monotonic() - t_search) * 1000

        # Filter noise from segments
        results = [r for r in results if len(r.get("text", "")) >= 50]

        # Score segment results
        for r in results:
            relevance, importance, recency = _compute_evidence_score(
                r, classification.mode
            )
            hot = _hot_score(
                r.get("retrieval_count", 0),
                r.get("last_retrieved_at"),
            )
            relevance = relevance * (1.0 + 0.15 * hot)

            # Penalize tool_call/tool_result segment types (often noise)
            seg_type = r.get("segment_type", "")
            if seg_type in ("tool_call", "tool_result"):
                relevance *= 0.4
            # Boost markdown/doc segments
            source_type = r.get("source_type", "")
            if source_type == "markdown":
                relevance *= 1.2

            token_count = r.get("token_count") or count_tokens(r.get("text", ""))

            seg_tenant = r.get("owner_tenant_id", "")
            seg_tier = _tenant_tiers.get(seg_tenant, "operational")

            # Reference tier: lower score so operational data beats book data
            if seg_tier in ("reference", "reference-technical"):
                relevance *= 0.7

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
                    "owner_tenant_id": seg_tenant,
                    "knowledge_tier": seg_tier,
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

    # 7. Build context text (with tier-aware attribution)
    context_parts = []
    for i, ev in enumerate(evidence_results, 1):
        tier = ev.metadata.get("knowledge_tier", "operational")
        header_parts = []
        if ev.metadata.get("title_or_heading"):
            header_parts.append(ev.metadata["title_or_heading"])
        if ev.metadata.get("speaker_role"):
            header_parts.append(f"[{ev.metadata['speaker_role']}]")
        if ev.metadata.get("segment_type"):
            header_parts.append(f"({ev.metadata['segment_type']})")
        header = " ".join(header_parts) if header_parts else f"Evidence {i}"

        # Reference tier: always attribute to source, never present as fact
        if tier in ("reference", "reference-technical"):
            source_name = ev.metadata.get("title_or_heading", "unknown source")
            prefix = f"**[Reference — from \"{source_name}\"]** This is a historical/reference claim, not verified operational fact.\n"
            context_parts.append(f"### [{i}] {header} [REFERENCE]\n{prefix}{ev.text}")
        else:
            context_parts.append(f"### [{i}] {header}\n{ev.text}")

    context_text = "\n\n---\n\n".join(context_parts)

    total_ms = (time.monotonic() - t_start) * 1000

    # Fire-and-forget access tracking for returned segments
    returned_ids = [ev.item_id for ev in evidence_results]
    if returned_ids:
        asyncio.get_event_loop().create_task(_track_access(returned_ids))

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
    if "shared" not in tids:
        tids = list(tids) + ["shared"]
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
