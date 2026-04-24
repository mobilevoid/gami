"""MCP tool implementations for multi-manifold memory system.

These tools are exposed via the MCP server and can be called
by Claude Code and other AI agents.
"""
import logging
from typing import Optional, List, Dict, Any
from dataclasses import asdict

from ..retrieval.orchestrator import RetrievalOrchestrator, RetrievalResult
from ..retrieval.query_classifier_v2 import classify_query_v2, explain_classification
from ..models.schemas import QueryModeV2, ManifoldWeights
from ..config import get_config
from ..exceptions import QueryError, query_empty, query_too_long
from ..metrics import track_classification

logger = logging.getLogger("manifold.mcp.tools")

# Maximum query length
MAX_QUERY_LENGTH = 2000


class ManifoldTools:
    """MCP tool implementations for manifold memory system."""

    def __init__(
        self,
        orchestrator: Optional[RetrievalOrchestrator] = None,
        config=None,
    ):
        self.orchestrator = orchestrator or RetrievalOrchestrator()
        self.config = config or get_config()

    async def memory_recall(
        self,
        query: str,
        top_k: int = 20,
        tenant_id: str = "shared",
        mode: Optional[str] = None,
        include_citations: bool = True,
        explain: bool = False,
    ) -> Dict[str, Any]:
        """Recall relevant memories for a query.

        This is the primary retrieval tool. It:
        1. Classifies the query to determine intent
        2. Retrieves from all relevant manifolds
        3. Fuses scores and ranks results
        4. Attaches source citations

        Args:
            query: The user's query.
            top_k: Number of results to return (default 20, max 100).
            tenant_id: Tenant context for access control.
            mode: Optional mode override (fact_lookup, timeline, etc.)
            include_citations: Whether to include source citations.
            explain: Whether to include classification explanation.

        Returns:
            Dict with 'results', 'mode', 'confidence', and optional 'explanation'.
        """
        # Validate query
        if not query or not query.strip():
            raise query_empty()
        if len(query) > MAX_QUERY_LENGTH:
            raise query_too_long(len(query), MAX_QUERY_LENGTH)

        # Clamp top_k
        top_k = max(1, min(top_k, self.config.max_top_k))

        # Parse mode override
        mode_override = None
        if mode:
            try:
                mode_override = QueryModeV2(mode.lower())
            except ValueError:
                logger.warning(f"Invalid mode override: {mode}")

        # Execute recall
        result = await self.orchestrator.recall(
            query=query.strip(),
            top_k=top_k,
            tenant_id=tenant_id,
            mode_override=mode_override,
            include_citations=include_citations,
        )

        # Track metrics
        track_classification(result.mode.value, result.confidence)

        # Format response
        response = {
            "query": query,
            "mode": result.mode.value,
            "confidence": result.confidence,
            "latency_ms": result.latency_ms,
            "cached": result.from_cache,
            "results": [
                {
                    "id": c.item_id,
                    "type": c.item_type,
                    "text": c.text,
                    "score": c.fused_score,
                    "citations": c.citations if include_citations else [],
                }
                for c in result.candidates
            ],
        }

        if explain:
            explanation = explain_classification(query)
            response["explanation"] = explanation

        return response

    async def memory_search(
        self,
        query: str,
        manifold: str = "topic",
        top_k: int = 20,
        tenant_id: str = "shared",
        threshold: float = 0.3,
    ) -> Dict[str, Any]:
        """Search a specific manifold directly.

        Lower-level search that bypasses classification and fusion.
        Useful for targeted searches when you know what you want.

        Args:
            query: The search query.
            manifold: Which manifold to search (topic, claim, procedure).
            top_k: Number of results.
            tenant_id: Tenant context.
            threshold: Minimum similarity score.

        Returns:
            Dict with 'results' list.
        """
        import time
        from ..embedding import embed_text

        if not query or not query.strip():
            raise query_empty()

        valid_manifolds = ["topic", "claim", "procedure", "relation", "time", "evidence"]
        if manifold not in valid_manifolds:
            manifold = "topic"

        start_time = time.time()

        # Get query embedding
        query_embedding = await embed_text(
            query.strip(),
            model=self.config.embedding_model,
            base_url=self.config.ollama_url,
        )

        # Search the specified manifold via repository
        if self.orchestrator.topic_index:
            results = await self.orchestrator.topic_index.search_by_embedding(
                embedding=query_embedding,
                manifold_type=manifold,
                limit=top_k,
                tenant_id=tenant_id if tenant_id != "shared" else None,
                min_score=threshold,
            )
        else:
            results = []

        latency_ms = (time.time() - start_time) * 1000

        return {
            "query": query,
            "manifold": manifold,
            "threshold": threshold,
            "latency_ms": round(latency_ms, 2),
            "results": [
                {
                    "id": r.object_id,
                    "type": r.object_type,
                    "text": r.text_preview[:500] if r.text_preview else "",
                    "score": r.similarity,
                }
                for r in results
                if r.similarity >= threshold
            ],
        }

    async def memory_classify(
        self,
        query: str,
        explain: bool = True,
    ) -> Dict[str, Any]:
        """Classify a query without retrieving results.

        Useful for understanding how queries are interpreted.

        Args:
            query: The query to classify.
            explain: Whether to include explanation.

        Returns:
            Classification result with mode, confidence, weights.
        """
        if not query or not query.strip():
            raise query_empty()

        result = classify_query_v2(query.strip())

        response = {
            "query": query,
            "mode": result.mode.value,
            "confidence": result.confidence,
            "manifold_weights": {
                "topic": result.manifold_weights.topic,
                "claim": result.manifold_weights.claim,
                "procedure": result.manifold_weights.procedure,
                "relation": result.manifold_weights.relation,
                "time": result.manifold_weights.time,
                "evidence": result.manifold_weights.evidence,
            },
        }

        if explain:
            explanation = explain_classification(query)
            response["explanation"] = explanation

        return response

    async def memory_cite(
        self,
        item_id: str,
        tenant_id: str = "shared",
    ) -> Dict[str, Any]:
        """Get full citation information for an item.

        Args:
            item_id: The item ID to cite.
            tenant_id: Tenant context.

        Returns:
            Full citation with source, location, confidence.
        """
        if not item_id or not item_id.strip():
            raise QueryError("ITEM_ID_EMPTY", "Item ID cannot be empty")

        # Query provenance and source information
        if not self.orchestrator.topic_index or not self.orchestrator.topic_index.pool:
            return {
                "item_id": item_id,
                "citations": [],
                "error": "Database not connected",
            }

        pool = self.orchestrator.topic_index.pool
        conn = await pool.acquire()

        try:
            # Get provenance records for this item
            provenance_rows = await conn.fetch(
                """
                SELECT
                    p.id as provenance_id,
                    p.source_segment_id,
                    p.extraction_method,
                    p.confidence,
                    p.created_at,
                    s.source_id,
                    s.text as segment_text,
                    s.start_offset,
                    s.end_offset,
                    src.title as source_title,
                    src.source_type,
                    src.uri as source_uri,
                    src.created_at as source_created
                FROM provenance p
                JOIN segments s ON s.id = p.source_segment_id
                JOIN sources src ON src.id = s.source_id
                WHERE p.derived_id = $1
                ORDER BY p.confidence DESC, p.created_at DESC
                LIMIT 10
                """,
                item_id,
            )

            citations = []
            for row in provenance_rows:
                citation = {
                    "provenance_id": row["provenance_id"],
                    "source": {
                        "id": row["source_id"],
                        "title": row["source_title"],
                        "type": row["source_type"],
                        "uri": row["source_uri"],
                        "created": row["source_created"].isoformat() if row["source_created"] else None,
                    },
                    "location": {
                        "segment_id": row["source_segment_id"],
                        "start_offset": row["start_offset"],
                        "end_offset": row["end_offset"],
                    },
                    "extraction_method": row["extraction_method"],
                    "confidence": row["confidence"],
                    "excerpt": row["segment_text"][:300] if row["segment_text"] else None,
                }
                citations.append(citation)

            # If no provenance, try to get direct source info
            if not citations:
                # Check if item is a segment itself
                segment_row = await conn.fetchrow(
                    """
                    SELECT
                        s.id, s.text, s.source_id, s.start_offset, s.end_offset,
                        src.title, src.source_type, src.uri
                    FROM segments s
                    JOIN sources src ON src.id = s.source_id
                    WHERE s.id = $1
                    """,
                    item_id,
                )

                if segment_row:
                    citations.append({
                        "source": {
                            "id": segment_row["source_id"],
                            "title": segment_row["title"],
                            "type": segment_row["source_type"],
                            "uri": segment_row["uri"],
                        },
                        "location": {
                            "segment_id": item_id,
                            "start_offset": segment_row["start_offset"],
                            "end_offset": segment_row["end_offset"],
                        },
                        "confidence": 1.0,
                        "excerpt": segment_row["text"][:300] if segment_row["text"] else None,
                        "is_primary": True,
                    })

            return {
                "item_id": item_id,
                "citations": citations,
                "citation_count": len(citations),
            }

        finally:
            await pool.release(conn)

    async def memory_verify(
        self,
        claim: str,
        tenant_id: str = "shared",
    ) -> Dict[str, Any]:
        """Verify a claim against stored knowledge.

        Searches for supporting and contradicting evidence.

        Args:
            claim: The claim to verify.
            tenant_id: Tenant context.

        Returns:
            Verification result with supporting/contradicting evidence.
        """
        from ..scoring.evidence import compute_evidence_score, EvidenceFactors
        from ..canonical.claim_normalizer import ClaimNormalizer

        if not claim or not claim.strip():
            raise query_empty()

        # Normalize the claim to canonical SPO form for comparison
        normalizer = ClaimNormalizer()
        canonical_claim = normalizer.normalize(claim.strip())

        # Retrieve evidence using verification mode
        result = await self.orchestrator.recall(
            query=claim.strip(),
            top_k=20,
            tenant_id=tenant_id,
            mode_override=QueryModeV2.VERIFICATION,
        )

        supporting_evidence = []
        contradicting_evidence = []
        neutral_evidence = []

        for candidate in result.candidates:
            # Get evidence scores
            evidence_score = candidate.manifold_scores.evidence if candidate.manifold_scores else 0.5

            # Determine if this evidence supports or contradicts
            # Check for negation patterns and semantic opposition
            text_lower = (candidate.text or "").lower()
            claim_lower = claim.lower()

            # Simple contradiction detection
            negation_words = ["not", "never", "no ", "isn't", "wasn't", "doesn't", "didn't", "won't", "can't", "couldn't"]
            has_negation = any(neg in text_lower for neg in negation_words)
            claim_has_negation = any(neg in claim_lower for neg in negation_words)

            # Check for subject/predicate overlap
            if canonical_claim:
                subject_match = canonical_claim.subject and canonical_claim.subject.lower() in text_lower
                predicate_match = canonical_claim.predicate and canonical_claim.predicate.lower() in text_lower
            else:
                subject_match = False
                predicate_match = False

            # Classify evidence
            if evidence_score > 0.7 and (has_negation != claim_has_negation) and (subject_match or predicate_match):
                contradicting_evidence.append({
                    "id": candidate.object_id,
                    "type": candidate.object_type,
                    "text": candidate.text[:500],
                    "evidence_score": evidence_score,
                    "fused_score": candidate.fused_score,
                    "reason": "Contains opposing assertion with high evidence score",
                })
            elif evidence_score > 0.5 and (subject_match or predicate_match):
                supporting_evidence.append({
                    "id": candidate.object_id,
                    "type": candidate.object_type,
                    "text": candidate.text[:500],
                    "evidence_score": evidence_score,
                    "fused_score": candidate.fused_score,
                })
            else:
                neutral_evidence.append({
                    "id": candidate.object_id,
                    "type": candidate.object_type,
                    "text": candidate.text[:300],
                    "evidence_score": evidence_score,
                })

        # Determine verification status
        supporting_count = len(supporting_evidence)
        contradicting_count = len(contradicting_evidence)
        total_high_evidence = supporting_count + contradicting_count

        if total_high_evidence == 0:
            verification_status = "insufficient_evidence"
            verification_confidence = 0.2
        elif contradicting_count > supporting_count * 2:
            verification_status = "contradicted"
            verification_confidence = min(0.9, 0.5 + (contradicting_count / 10))
        elif supporting_count > contradicting_count * 2:
            verification_status = "supported"
            verification_confidence = min(0.9, 0.5 + (supporting_count / 10))
        elif supporting_count > 0 and contradicting_count == 0:
            verification_status = "likely_supported"
            verification_confidence = min(0.7, 0.4 + (supporting_count / 10))
        elif contradicting_count > 0 and supporting_count == 0:
            verification_status = "likely_contradicted"
            verification_confidence = min(0.7, 0.4 + (contradicting_count / 10))
        else:
            verification_status = "uncertain"
            verification_confidence = 0.3

        return {
            "claim": claim,
            "canonical_form": {
                "subject": canonical_claim.subject if canonical_claim else None,
                "predicate": canonical_claim.predicate if canonical_claim else None,
                "object": canonical_claim.object if canonical_claim else None,
            } if canonical_claim else None,
            "verification_status": verification_status,
            "verification_confidence": round(verification_confidence, 3),
            "supporting_evidence": supporting_evidence[:5],
            "contradicting_evidence": contradicting_evidence[:5],
            "neutral_evidence": neutral_evidence[:3],
            "evidence_summary": {
                "supporting_count": supporting_count,
                "contradicting_count": contradicting_count,
                "total_candidates": len(result.candidates),
            },
            "retrieval_latency_ms": result.latency_ms,
        }

    async def manifold_stats(
        self,
        tenant_id: str = "shared",
    ) -> Dict[str, Any]:
        """Get statistics about manifold system.

        Args:
            tenant_id: Tenant to get stats for.

        Returns:
            Stats about embeddings, promotions, queries, etc.
        """
        import redis.asyncio as redis
        from datetime import datetime, timezone, timedelta

        stats = {
            "tenant_id": tenant_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "embeddings": {},
            "promoted_objects": 0,
            "canonical_claims": 0,
            "canonical_procedures": 0,
        }

        # Database stats
        if self.orchestrator.topic_index and self.orchestrator.topic_index.pool:
            pool = self.orchestrator.topic_index.pool
            conn = await pool.acquire()

            try:
                # Tenant filter for queries
                tenant_filter = "AND tenant_id = $1" if tenant_id != "shared" else ""
                params = [tenant_id] if tenant_id != "shared" else []

                # Embedding counts by manifold
                embed_query = f"""
                    SELECT manifold_type, COUNT(*) as cnt
                    FROM manifold_embeddings
                    WHERE 1=1 {tenant_filter}
                    GROUP BY manifold_type
                """
                embed_rows = await conn.fetch(embed_query, *params) if params else await conn.fetch(embed_query)
                stats["embeddings"] = {row["manifold_type"]: row["cnt"] for row in embed_rows}

                # Promotion tier breakdown
                promo_query = f"""
                    SELECT tier, COUNT(*) as cnt
                    FROM promotion_scores
                    WHERE 1=1 {tenant_filter.replace('tenant_id', 'object_type')}
                    GROUP BY tier
                """
                promo_rows = await conn.fetch(promo_query.replace("$1", f"'{tenant_id}'") if tenant_id != "shared" else promo_query)
                stats["promotion_tiers"] = {row["tier"]: row["cnt"] for row in promo_rows}

                # Promoted object count
                promoted_query = """
                    SELECT COUNT(*) as cnt FROM promotion_scores WHERE tier = 'promoted'
                """
                promoted_row = await conn.fetchrow(promoted_query)
                stats["promoted_objects"] = promoted_row["cnt"] if promoted_row else 0

                # Canonical claims count
                claims_query = f"""
                    SELECT COUNT(*) as cnt FROM canonical_claims WHERE 1=1 {tenant_filter}
                """
                claims_row = await conn.fetchrow(claims_query, *params) if params else await conn.fetchrow(claims_query)
                stats["canonical_claims"] = claims_row["cnt"] if claims_row else 0

                # Canonical procedures count
                procs_query = f"""
                    SELECT COUNT(*) as cnt FROM canonical_procedures WHERE 1=1 {tenant_filter}
                """
                procs_row = await conn.fetchrow(procs_query, *params) if params else await conn.fetchrow(procs_query)
                stats["canonical_procedures"] = procs_row["cnt"] if procs_row else 0

                # Query stats (last 24 hours)
                yesterday = datetime.now(timezone.utc) - timedelta(hours=24)
                query_stats = await conn.fetchrow(
                    f"""
                    SELECT
                        COUNT(*) as total_queries,
                        AVG(latency_ms) as avg_latency,
                        SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful,
                        SUM(CASE WHEN from_cache THEN 1 ELSE 0 END) as cache_hits
                    FROM query_logs
                    WHERE executed_at > $1 {tenant_filter.replace('$1', '$2')}
                    """,
                    yesterday, *params
                ) if params else await conn.fetchrow(
                    """
                    SELECT
                        COUNT(*) as total_queries,
                        AVG(latency_ms) as avg_latency,
                        SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful,
                        SUM(CASE WHEN from_cache THEN 1 ELSE 0 END) as cache_hits
                    FROM query_logs
                    WHERE executed_at > $1
                    """,
                    yesterday,
                )

                if query_stats:
                    total = query_stats["total_queries"] or 0
                    cache_hits = query_stats["cache_hits"] or 0
                    stats["queries_24h"] = {
                        "total": total,
                        "successful": query_stats["successful"] or 0,
                        "avg_latency_ms": round(query_stats["avg_latency"] or 0, 2),
                        "cache_hits": cache_hits,
                        "cache_hit_rate": round(cache_hits / max(1, total), 3),
                    }

                # Shadow mode stats
                shadow_stats = await conn.fetchrow(
                    """
                    SELECT
                        COUNT(*) as total,
                        AVG(overlap_ratio) as avg_overlap,
                        SUM(CASE WHEN manifold_better THEN 1 ELSE 0 END) as manifold_wins
                    FROM shadow_comparisons
                    WHERE created_at > NOW() - INTERVAL '24 hours'
                    """
                )
                if shadow_stats and shadow_stats["total"]:
                    stats["shadow_mode"] = {
                        "comparisons_24h": shadow_stats["total"],
                        "avg_overlap": round(shadow_stats["avg_overlap"] or 0, 3),
                        "manifold_win_rate": round((shadow_stats["manifold_wins"] or 0) / max(1, shadow_stats["total"]), 3),
                    }

            finally:
                await pool.release(conn)
        else:
            stats["database"] = "not connected"

        # Redis cache stats
        try:
            redis_client = redis.from_url(self.config.redis_url, decode_responses=True)
            try:
                info = await redis_client.info("memory")
                stats["cache"] = {
                    "used_memory": info.get("used_memory_human", "N/A"),
                    "total_keys": await redis_client.dbsize(),
                }
            finally:
                await redis_client.close()
        except Exception as e:
            stats["cache"] = {"status": f"error: {e}"}

        return stats


# Tool definitions for MCP server registration
TOOL_DEFINITIONS = [
    {
        "name": "memory_recall",
        "description": "Recall relevant memories for a query. Uses multi-manifold retrieval with automatic query classification.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to search for",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results (default 20, max 100)",
                    "default": 20,
                },
                "tenant_id": {
                    "type": "string",
                    "description": "Tenant context",
                    "default": "shared",
                },
                "mode": {
                    "type": "string",
                    "description": "Optional mode override: fact_lookup, timeline, procedure, verification, comparison, synthesis, report, assistant_memory",
                },
                "include_citations": {
                    "type": "boolean",
                    "description": "Include source citations",
                    "default": True,
                },
                "explain": {
                    "type": "boolean",
                    "description": "Include classification explanation",
                    "default": False,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "memory_search",
        "description": "Search a specific manifold directly, bypassing automatic classification.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "manifold": {
                    "type": "string",
                    "enum": ["topic", "claim", "procedure"],
                    "default": "topic",
                },
                "top_k": {"type": "integer", "default": 20},
                "tenant_id": {"type": "string", "default": "shared"},
                "threshold": {"type": "number", "default": 0.3},
            },
            "required": ["query"],
        },
    },
    {
        "name": "memory_classify",
        "description": "Classify a query to see how it would be interpreted.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "explain": {"type": "boolean", "default": True},
            },
            "required": ["query"],
        },
    },
    {
        "name": "memory_verify",
        "description": "Verify a claim against stored knowledge.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "claim": {"type": "string"},
                "tenant_id": {"type": "string", "default": "shared"},
            },
            "required": ["claim"],
        },
    },
    {
        "name": "manifold_stats",
        "description": "Get statistics about the manifold system.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "tenant_id": {"type": "string", "default": "shared"},
            },
        },
    },
]
