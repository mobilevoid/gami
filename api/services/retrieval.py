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
from api.search.hybrid import (
    hybrid_search,
    vector_search,
    lexical_search,
    hybrid_manifold_search,
)
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
from api.services.learning_service import RetrievalLogger
from api.services.event_publisher import publish_event
from manifold.retrieval.query_router import (
    route_query,
    ManifoldType,
    MANIFOLD_WEIGHT_PROFILES,
    QueryIntent,
)
from manifold.retrieval.query_routing import (
    QueryRouter as GraphQueryRouter,
    route_query as route_query_graph,
    IndexType,
)
from manifold.retrieval.multi_index_retriever import MultiIndexRetriever

logger = logging.getLogger("gami.services.retrieval")

# Global retrieval logger instance
_retrieval_logger: Optional[RetrievalLogger] = None


def _get_retrieval_logger() -> RetrievalLogger:
    """Get or create the retrieval logger singleton."""
    global _retrieval_logger
    if _retrieval_logger is None:
        _retrieval_logger = RetrievalLogger()
    return _retrieval_logger


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


class ContradictionResult(BaseModel):
    """Information about detected contradictions."""
    group_id: str
    predicate: str
    claim_ids: list[str]
    values: list[str]
    confidences: list[float]
    status: str  # unresolved, proposed, resolved
    proposal_id: Optional[str] = None


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
    # Phase 7: Contradiction awareness
    has_contradictions: bool = False
    contradictions: list[ContradictionResult] = []
    needs_resolution: bool = False


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
    tenant_id: str = "default",
    tenant_ids: Optional[list[str]] = None,
    max_tokens: int = 4000,
    mode: Optional[str] = None,
    include_citations: bool = True,
    citation_level: CitationLevel = CitationLevel.BRIEF,
    session_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    # Phase 5: Bi-temporal query support
    event_after: Optional[str] = None,
    event_before: Optional[str] = None,
    ingested_after: Optional[str] = None,
    ingested_before: Optional[str] = None,
    # Phase 3: Compression detail level
    detail_level: str = "normal",
    # Phase 9: True product manifold search
    use_product_manifold: bool = False,
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
        session_id: Optional session ID for tracking.
        agent_id: Optional agent ID for tracking.
        event_after: Filter to events that happened after this ISO timestamp.
        event_before: Filter to events that happened before this ISO timestamp.
        ingested_after: Filter to content ingested after this ISO timestamp.
        ingested_before: Filter to content ingested before this ISO timestamp.
        detail_level: Level of detail - "summary" (abstractions only),
                      "normal" (+ important deltas), "full" (original text).

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
                            "knowledge_tier": _tenant_tiers.get("default", "operational"),
                        },
                    ))
            except Exception as exc:
                logger.debug("Graph expansion failed (non-critical): %s", exc)

        # ---- 4d. Search segments (multi-manifold + hybrid search) ----
        # Get manifold weights based on query intent
        manifold_weights_dict, detected_intent, intent_confidence = route_query(query)
        # Convert enum keys to string keys for the search function
        manifold_weights = {
            mt.value if hasattr(mt, 'value') else str(mt): w
            for mt, w in manifold_weights_dict.items()
        }

        logger.debug(
            f"Query intent: {detected_intent} (conf={intent_confidence:.2f}), "
            f"manifold_weights: {manifold_weights}"
        )

        # Phase 9: Use true product manifold search (H^32 × S^16 × E^64) if enabled
        results = []  # Initialize to avoid possibly unbound
        if use_product_manifold:
            try:
                from api.search.hybrid import product_manifold_hybrid_search
                results = await product_manifold_hybrid_search(
                    db, query, query_embedding, tids,
                    limit=search_limit,
                    use_product_manifold=True,
                    manifold_weight=0.4,
                )
                logger.info(f"Product manifold search returned {len(results)} results")
            except Exception as pm_err:
                logger.warning(f"Product manifold search failed: {pm_err}, falling back")
                use_product_manifold = False  # Fall through to standard search

        # Try multi-manifold search first, fall back to traditional hybrid
        if not use_product_manifold:
            try:
                results = await hybrid_manifold_search(
                    db, query, query_embedding, tids,
                    manifold_weights=manifold_weights,
                    limit=search_limit,
                    vector_weight=0.4,  # Traditional search gets 40%
                    manifold_weight=0.6,  # Manifold search gets 60%
                )
            except Exception as manifold_err:
                logger.warning(f"Manifold search failed, falling back to hybrid: {manifold_err}")
                results = await hybrid_search(
                    db, query, query_embedding, tids,
                    limit=search_limit,
                    vector_weight=0.7,
                    lexical_weight=0.3,
                )

        search_ms = (time.monotonic() - t_search) * 1000

        # Filter noise from segments
        results = [r for r in results if len(r.get("text", "")) >= 50]

        # Phase 5: Apply bi-temporal filtering
        if event_after or event_before or ingested_after or ingested_before:
            def _passes_temporal(r):
                # Event time filtering (when the event actually happened)
                event_time = r.get("event_time") or r.get("message_timestamp")
                if event_time:
                    if isinstance(event_time, str):
                        try:
                            event_time = datetime.fromisoformat(event_time.replace("Z", "+00:00"))
                        except ValueError:
                            event_time = None
                    if event_time:
                        if event_after:
                            try:
                                after_dt = datetime.fromisoformat(event_after.replace("Z", "+00:00"))
                                if event_time < after_dt:
                                    return False
                            except ValueError:
                                pass
                        if event_before:
                            try:
                                before_dt = datetime.fromisoformat(event_before.replace("Z", "+00:00"))
                                if event_time > before_dt:
                                    return False
                            except ValueError:
                                pass
                elif event_after or event_before:
                    # If we need event filtering but have no event time, exclude
                    return False

                # Ingestion time filtering (when we learned about it)
                created_at = r.get("created_at")
                if created_at:
                    if isinstance(created_at, str):
                        try:
                            created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                        except ValueError:
                            created_at = None
                    if created_at:
                        if ingested_after:
                            try:
                                after_dt = datetime.fromisoformat(ingested_after.replace("Z", "+00:00"))
                                if created_at < after_dt:
                                    return False
                            except ValueError:
                                pass
                        if ingested_before:
                            try:
                                before_dt = datetime.fromisoformat(ingested_before.replace("Z", "+00:00"))
                                if created_at > before_dt:
                                    return False
                            except ValueError:
                                pass
                elif ingested_after or ingested_before:
                    return False

                return True

            results = [r for r in results if _passes_temporal(r)]
            logger.debug(f"After temporal filtering: {len(results)} results")

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

        # 4e. Phase 6: Multi-index retrieval based on query routing
        try:
            graph_routing = route_query_graph(query)
            if graph_routing.routing_confidence >= 0.6:
                multi_retriever = MultiIndexRetriever()
                index_results = await multi_retriever.retrieve(
                    db, query, query_embedding, graph_routing, tids,
                    limit_per_index=10,
                )
                # Convert IndexResult to EvidenceItem
                for ir in index_results:
                    evidence_items.append(EvidenceItem(
                        item_id=ir.item_id,
                        item_type=ir.item_type,
                        text=ir.text,
                        score=ir.score,
                        importance=0.7,
                        recency_score=0.5,
                        token_count=count_tokens(ir.text),
                        metadata={
                            "index_source": ir.index_source.value,
                            **ir.metadata,
                        },
                    ))
                logger.debug(
                    f"Multi-index retrieval added {len(index_results)} items "
                    f"(routing: {graph_routing.primary_index.value}, "
                    f"conf={graph_routing.routing_confidence:.2f})"
                )
        except Exception as mir_err:
            logger.warning(f"Multi-index retrieval failed (non-critical): {mir_err}")

        # 4f. Cross-encoder reranking (if enabled)
        from api.config import settings
        if settings.RERANKER_ENABLED and len(evidence_items) > settings.RERANKER_FINAL_K:
            try:
                from api.search.reranker import CrossEncoderReranker
                reranker = CrossEncoderReranker(blend_ratio=settings.RERANKER_BLEND_RATIO)
                if reranker.is_available():
                    # Take top candidates for reranking
                    evidence_items.sort(key=lambda x: -x.score)
                    to_rerank = evidence_items[:settings.RERANKER_TOP_K]
                    evidence_items = reranker.rerank_evidence(
                        query, to_rerank, top_n=settings.RERANKER_FINAL_K
                    )
                    logger.debug(f"Reranked {len(to_rerank)} items to {len(evidence_items)}")
            except Exception as e:
                logger.warning(f"Reranking failed, using original order: {e}")

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
    returned_scores = [ev.score for ev in evidence_results]
    if returned_ids:
        asyncio.get_event_loop().create_task(_track_access(returned_ids))

    # Log retrieval for learning (fire-and-forget)
    if session_id:
        async def _log_retrieval_task():
            try:
                async with AsyncSessionLocal() as log_db:
                    retrieval_logger = _get_retrieval_logger()
                    await retrieval_logger.log_retrieval(
                        db=log_db,
                        session_id=session_id,
                        query_text=query,
                        query_mode=classification.mode.value,
                        segments_returned=returned_ids,
                        scores_returned=returned_scores,
                        tenant_id=tenant_id,
                        agent_id=agent_id,
                    )
            except Exception as log_err:
                logger.warning(f"Failed to log retrieval for learning: {log_err}")

        asyncio.get_event_loop().create_task(_log_retrieval_task())

    # Publish event for subconscious daemon (fire-and-forget)
    try:
        publish_event(
            event_type="query",
            session_id=session_id,
            tenant_id=tenant_id,
            agent_id=agent_id,
            query=query,
            mode=classification.mode.value,
            results=[ev.item_id for ev in evidence_results[:10]],
            result_count=len(evidence_results),
            latency_ms=round(total_ms, 1),
        )
    except Exception as pub_err:
        logger.debug(f"Event publish failed (non-critical): {pub_err}")

    # Phase 7: Check for contradictions in returned claims/segments
    contradiction_results: list[ContradictionResult] = []
    has_contradictions = False
    needs_resolution = False

    async with AsyncSessionLocal() as db:
        try:
            from api.services.contradiction_service import ContradictionChecker

            checker = ContradictionChecker()

            # Collect claim and segment IDs from evidence
            claim_ids = [e.item_id for e in evidence_results if e.item_type == "claim"]
            segment_ids = [e.item_id for e in evidence_results if e.item_type == "segment"]

            contradictions = await checker.check_for_contradictions(
                db, claim_ids, segment_ids, tids
            )

            if contradictions:
                has_contradictions = True
                needs_resolution = any(c.status == "unresolved" for c in contradictions)

                # Convert to result format
                for c in contradictions:
                    contradiction_results.append(ContradictionResult(
                        group_id=c.contradiction_group_id,
                        predicate=c.predicate,
                        claim_ids=c.claim_ids,
                        values=c.conflicting_values,
                        confidences=c.confidence_scores,
                        status=c.status,
                        proposal_id=c.proposal_id,
                    ))

                # Append warning to context text
                warning = checker.format_contradiction_warning(contradictions)
                if warning:
                    context_text = context_text + warning

        except Exception as contra_err:
            logger.warning(f"Contradiction check failed (non-critical): {contra_err}")

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
        has_contradictions=has_contradictions,
        contradictions=contradiction_results,
        needs_resolution=needs_resolution,
    )


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

async def verify_claim(
    claim_text: str,
    tenant_id: str = "default",
    tenant_ids: Optional[list[str]] = None,
    max_evidence: int = 10,
) -> VerifyResult:
    """Check whether a claim is supported or contradicted by stored evidence.

    Searches for the claim using manifold-aware search, prioritizing
    CLAIM and EVIDENCE manifolds. Verdict is based on score distribution.
    """
    tids = tenant_ids or [tenant_id]
    if "shared" not in tids:
        tids = list(tids) + ["shared"]
    query_embedding = await embed_text(claim_text)

    # Use verification-optimized manifold weights
    # Prioritize CLAIM and EVIDENCE manifolds for fact-checking
    verification_weights = {
        "TOPIC": 0.20,
        "CLAIM": 0.35,
        "PROCEDURE": 0.05,
        "RELATION": 0.10,
        "TIME": 0.05,
        "EVIDENCE": 0.25,
    }

    async with AsyncSessionLocal() as db:
        try:
            results = await hybrid_manifold_search(
                db, claim_text, query_embedding, tids,
                manifold_weights=verification_weights,
                limit=max_evidence * 2,
                vector_weight=0.4,
                manifold_weight=0.6,
            )
        except Exception as e:
            # Fall back to standard search if manifold search fails
            logger.warning(f"Manifold search failed in verify_claim, falling back: {e}")
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
