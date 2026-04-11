"""Memory search, recall, and assistant memory API for GAMI.

Provides hybrid vector + lexical search, orchestrated recall with budget
management and citations, claim verification, and assistant memory CRUD.
"""
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.config import settings
from api.llm.embeddings import embed_text
from api.search.hybrid import hybrid_search, lexical_search, vector_search
from api.services.db import AsyncSessionLocal
from api.services.retrieval import recall, verify_claim, RecallResult, VerifyResult
from api.services.citation_service import CitationLevel, get_provenance_chain
from api.services.memory_service import (
    remember,
    forget,
    update_memory,
    confirm,
    get_context,
)
from api.services.learning_service import get_retrieval_logger, OUTCOME_SIGNALS
from api.services.session_service import (
    ensure_session,
    record_retrieval,
    record_learning_signal,
    update_session_state,
)
from api.services.event_publisher import publish_query_event, publish_feedback_event

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response models — Search (existing)
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    tenant_id: str = Field(default="claude-opus")
    tenant_ids: Optional[list[str]] = Field(default=None, description="Search multiple tenants")
    limit: int = Field(default=20, ge=1, le=100)
    vector_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    lexical_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    search_mode: str = Field(default="hybrid", pattern="^(hybrid|vector|lexical)$")


class SearchResult(BaseModel):
    segment_id: str
    text: str
    source_id: str
    owner_tenant_id: str
    segment_type: str
    title_or_heading: Optional[str] = None
    speaker_role: Optional[str] = None
    token_count: Optional[int] = None
    combined_score: Optional[float] = None
    vector_score: Optional[float] = None
    lexical_score: Optional[float] = None
    search_type: str = "hybrid"
    message_timestamp: Optional[str] = None


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    total: int


# ---------------------------------------------------------------------------
# Request / Response models — Recall (Phase 6)
# ---------------------------------------------------------------------------

class RecallRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    tenant_id: str = Field(default="claude-opus")
    tenant_ids: Optional[list[str]] = Field(default=None)
    max_tokens: int = Field(default=4000, ge=100, le=16000)
    mode: Optional[str] = Field(default=None, description="Override query mode")
    include_citations: bool = Field(default=True)
    citation_level: str = Field(default="brief", pattern="^(brief|full|drill_down)$")
    session_id: Optional[str] = Field(default=None, description="Session ID for learning")
    agent_id: Optional[str] = Field(default=None, description="Agent ID for attribution")


class RecallResponse(BaseModel):
    query: str
    mode: str
    granularity: str
    evidence: list[dict]
    context_text: str
    total_tokens_used: int
    total_candidates: int
    classification_ms: float = 0.0
    search_ms: float = 0.0
    total_ms: float = 0.0


# ---------------------------------------------------------------------------
# Request / Response models — Verify
# ---------------------------------------------------------------------------

class VerifyRequest(BaseModel):
    claim: str = Field(..., min_length=1, max_length=2000)
    tenant_id: str = Field(default="claude-opus")
    tenant_ids: Optional[list[str]] = Field(default=None)
    max_evidence: int = Field(default=10, ge=1, le=50)


class VerifyResponse(BaseModel):
    claim: str
    verdict: str
    confidence: float
    supporting_evidence: list[dict]
    contradicting_evidence: list[dict]


# ---------------------------------------------------------------------------
# Request / Response models — Citation
# ---------------------------------------------------------------------------

class CiteRequest(BaseModel):
    item_id: str
    item_type: str = Field(default="segment", pattern="^(segment|claim|summary|entity|memory)$")
    level: str = Field(default="full", pattern="^(brief|full|drill_down)$")


# ---------------------------------------------------------------------------
# Request / Response models — Assistant Memory (Phase 8)
# ---------------------------------------------------------------------------

class RememberRequest(BaseModel):
    tenant_id: str = Field(default="claude-opus")
    text: str = Field(..., min_length=1, max_length=5000)
    memory_type: str = Field(default="fact")
    subject: str = Field(default="general")
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    source_info: Optional[dict] = None
    session_id: Optional[str] = None


class ForgetRequest(BaseModel):
    memory_id: str
    reason: str = Field(default="user_requested")


class UpdateMemoryRequest(BaseModel):
    memory_id: str
    new_text: str = Field(..., min_length=1, max_length=5000)
    reason: str = Field(default="correction")


class ConfirmRequest(BaseModel):
    memory_id: str


class FeedbackRequest(BaseModel):
    """Request to record retrieval outcome for learning."""
    session_id: str = Field(..., description="Session ID from recall request")
    feedback_type: str = Field(
        ...,
        description="Outcome type: confirmed, used, continued, ignored, corrected, wrong",
        pattern="^(confirmed|used|continued|ignored|corrected|wrong)$"
    )
    correction_text: Optional[str] = Field(
        default=None,
        description="Corrected information if feedback_type is 'corrected'"
    )


# ---------------------------------------------------------------------------
# Search endpoint (existing, unchanged)
# ---------------------------------------------------------------------------

@router.post("/search", response_model=SearchResponse)
async def search_memory(req: SearchRequest):
    """Search memory using hybrid vector + lexical search."""
    tenant_ids = req.tenant_ids or [req.tenant_id]

    try:
        query_embedding = None
        if req.search_mode != "lexical":
            query_embedding = await embed_text(req.query)

        async with AsyncSessionLocal() as db:
            if req.search_mode == "lexical":
                results = await lexical_search(db, req.query, tenant_ids, limit=req.limit)
                for r in results:
                    r["combined_score"] = r.get("rank", 0.0)
                    r["vector_score"] = 0.0
                    r["lexical_score"] = r.get("rank", 0.0)
            elif req.search_mode == "vector":
                results = await vector_search(db, query_embedding, tenant_ids, limit=req.limit)
                for r in results:
                    r["combined_score"] = r.get("similarity", 0.0)
                    r["vector_score"] = r.get("similarity", 0.0)
                    r["lexical_score"] = 0.0
            else:
                results = await hybrid_search(
                    db,
                    req.query,
                    query_embedding,
                    tenant_ids,
                    limit=req.limit,
                    vector_weight=req.vector_weight,
                    lexical_weight=req.lexical_weight,
                )
    except Exception as exc:
        logger.error("Search failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}")

    return SearchResponse(
        query=req.query,
        results=[
            SearchResult(
                segment_id=r["segment_id"],
                text=r["text"],
                source_id=r["source_id"],
                owner_tenant_id=r["owner_tenant_id"],
                segment_type=r["segment_type"],
                title_or_heading=r.get("title_or_heading"),
                speaker_role=r.get("speaker_role"),
                token_count=r.get("token_count"),
                combined_score=r.get("combined_score"),
                vector_score=r.get("vector_score"),
                lexical_score=r.get("lexical_score"),
                search_type=r.get("search_type", "hybrid"),
                message_timestamp=r.get("message_timestamp"),
            )
            for r in results
        ],
        total=len(results),
    )


# ---------------------------------------------------------------------------
# Phase 6: Enhanced recall endpoint
# ---------------------------------------------------------------------------

@router.post("/recall", response_model=RecallResponse)
async def recall_memory(req: RecallRequest):
    """Full retrieval with query classification, budget management, and citations.

    The orchestrator classifies the query, runs hybrid search, scores results
    with semantic + lexical + confidence + importance + recency + provenance,
    applies token budget, and attaches citations.
    """
    try:
        cit_level = CitationLevel(req.citation_level)
    except ValueError:
        cit_level = CitationLevel.BRIEF

    try:
        # Track session if session_id provided
        if req.session_id:
            async with AsyncSessionLocal() as db:
                await ensure_session(
                    db=db,
                    session_id=req.session_id,
                    tenant_id=req.tenant_id,
                    agent_id=req.agent_id,
                )

        result = await recall(
            query=req.query,
            tenant_id=req.tenant_id,
            tenant_ids=req.tenant_ids,
            max_tokens=req.max_tokens,
            mode=req.mode,
            include_citations=req.include_citations,
            citation_level=cit_level,
            session_id=req.session_id,
            agent_id=req.agent_id,
        )

        # Record retrieval in session (fire-and-forget)
        if req.session_id:
            async def _record_session_retrieval():
                try:
                    async with AsyncSessionLocal() as db:
                        await record_retrieval(db, req.session_id)
                        # Update session state based on query mode
                        await update_session_state(
                            db=db,
                            session_id=req.session_id,
                            state=result.mode,
                            confidence=0.7,
                        )
                except Exception as e:
                    logger.warning("Session retrieval tracking failed: %s", e)

            import asyncio
            asyncio.create_task(_record_session_retrieval())

            # Publish event for subconscious daemon
            publish_query_event(
                session_id=req.session_id,
                query=req.query,
                tenant_id=req.tenant_id,
                agent_id=req.agent_id,
                results=[{"segment_id": e.item_id, "score": e.score} for e in result.evidence[:10]],
            )

        return RecallResponse(
            query=result.query,
            mode=result.mode,
            granularity=result.granularity,
            evidence=[e.model_dump() for e in result.evidence],
            context_text=result.context_text,
            total_tokens_used=result.total_tokens_used,
            total_candidates=result.total_candidates,
            classification_ms=result.classification_ms,
            search_ms=result.search_ms,
            total_ms=result.total_ms,
        )
    except Exception as exc:
        logger.error("Recall failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Recall failed: {exc}")


# ---------------------------------------------------------------------------
# Phase 6: Verify endpoint
# ---------------------------------------------------------------------------

@router.post("/verify", response_model=VerifyResponse)
async def verify(req: VerifyRequest):
    """Check if a claim is supported or contradicted by stored evidence."""
    try:
        result = await verify_claim(
            claim_text=req.claim,
            tenant_id=req.tenant_id,
            tenant_ids=req.tenant_ids,
            max_evidence=req.max_evidence,
        )
        return VerifyResponse(
            claim=result.claim,
            verdict=result.verdict,
            confidence=result.confidence,
            supporting_evidence=[e.model_dump() for e in result.supporting_evidence],
            contradicting_evidence=[e.model_dump() for e in result.contradicting_evidence],
        )
    except Exception as exc:
        logger.error("Verify failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Verify failed: {exc}")


# ---------------------------------------------------------------------------
# Phase 6: Citation endpoint
# ---------------------------------------------------------------------------

@router.post("/cite")
async def cite_source(req: CiteRequest):
    """Get source location/provenance for a claim or segment."""
    try:
        async with AsyncSessionLocal() as db:
            chain = await get_provenance_chain(db, req.item_id, req.item_type)

            from api.services.citation_service import build_citation
            cit = await build_citation(
                db,
                item_id=req.item_id,
                item_type=req.item_type,
                level=CitationLevel(req.level),
            )

            return {
                "item_id": req.item_id,
                "item_type": req.item_type,
                "citation": cit.model_dump(),
                "provenance_chain": chain,
            }
    except Exception as exc:
        logger.error("Citation failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Citation failed: {exc}")


# ---------------------------------------------------------------------------
# Phase 8: Assistant Memory endpoints
# ---------------------------------------------------------------------------

@router.post("/remember")
async def remember_endpoint(req: RememberRequest):
    """Store a new assistant memory.

    Performs duplicate check (embedding similarity > 0.92), detects credential
    sensitivity, and starts as provisional status.
    """
    try:
        # Track session if session_id provided
        if req.session_id:
            async with AsyncSessionLocal() as db:
                await ensure_session(
                    db=db,
                    session_id=req.session_id,
                    tenant_id=req.tenant_id,
                )

        result = await remember(
            tenant_id=req.tenant_id,
            text_content=req.text,
            memory_type=req.memory_type,
            subject=req.subject,
            importance=req.importance,
            source_info=req.source_info,
            session_id=req.session_id,
        )
        return result
    except Exception as exc:
        logger.error("Remember failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Remember failed: {exc}")


@router.post("/forget")
async def forget_endpoint(req: ForgetRequest):
    """Archive a memory (never delete). Provide memory_id and reason."""
    try:
        result = await forget(memory_id=req.memory_id, reason=req.reason)
        return result
    except Exception as exc:
        logger.error("Forget failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Forget failed: {exc}")


@router.post("/update")
async def update_memory_endpoint(req: UpdateMemoryRequest):
    """Revise a memory: creates a new version and supersedes the old one."""
    try:
        result = await update_memory(
            memory_id=req.memory_id,
            new_text=req.new_text,
            reason=req.reason,
        )
        return result
    except Exception as exc:
        logger.error("Update memory failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Update failed: {exc}")


@router.post("/confirm")
async def confirm_endpoint(req: ConfirmRequest):
    """Confirm a memory, incrementing confirmation_count and updating stability.

    Memories with 2+ confirmations are promoted from provisional to active.
    """
    try:
        result = await confirm(memory_id=req.memory_id)
        return result
    except Exception as exc:
        logger.error("Confirm failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Confirm failed: {exc}")


@router.get("/context/{session_id}")
async def get_session_context(session_id: str, tenant_id: str = "claude-opus"):
    """Get active session working memory: recently retrieved memories + relevant active ones."""
    try:
        result = await get_context(
            session_id=session_id,
            tenant_id=tenant_id,
        )
        return result
    except Exception as exc:
        logger.error("Get context failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Context failed: {exc}")


# ---------------------------------------------------------------------------
# Feedback endpoint for learning
# ---------------------------------------------------------------------------

@router.post("/feedback")
async def record_feedback(req: FeedbackRequest):
    """Record user feedback on retrieval for learning.

    Feedback types and their signals:
    - confirmed: User explicitly confirmed information was correct (+1.0)
    - used: User used the retrieved information (+0.8)
    - continued: User continued conversation without issue (+0.3)
    - ignored: User ignored the retrieved results (-0.2)
    - corrected: User provided correction (-0.8)
    - wrong: User indicated information was wrong (-1.0)
    """
    try:
        retrieval_logger = get_retrieval_logger()

        # Get the signal value for this feedback type
        signal = OUTCOME_SIGNALS.get(req.feedback_type, 0.0)

        # Record the outcome in the database
        async with AsyncSessionLocal() as db:
            success = await retrieval_logger.record_outcome(
                db=db,
                session_id=req.session_id,
                outcome_type=req.feedback_type,
                correction_text=req.correction_text,
            )

            # Also record learning signal in session tracking
            if success:
                await record_learning_signal(
                    db=db,
                    session_id=req.session_id,
                    positive=(signal > 0),
                )

        return {
            "status": "recorded" if success else "no_matching_log",
            "session_id": req.session_id,
            "feedback_type": req.feedback_type,
            "signal_value": signal,
        }
    except Exception as exc:
        logger.error("Feedback recording failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Feedback failed: {exc}")
