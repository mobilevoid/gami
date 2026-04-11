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

logger = logging.getLogger("gami.manifold.mcp.tools")

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
        if not query or not query.strip():
            raise query_empty()

        valid_manifolds = ["topic", "claim", "procedure"]
        if manifold not in valid_manifolds:
            manifold = "topic"

        # Direct manifold search would be implemented here
        # For now, return empty results (stub)
        return {
            "query": query,
            "manifold": manifold,
            "threshold": threshold,
            "results": [],
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
        # Would query provenance table
        return {
            "item_id": item_id,
            "citations": [],
            "note": "Citation lookup not implemented in isolated module",
        }

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
        if not claim or not claim.strip():
            raise query_empty()

        # Would use evidence scoring and contradiction detection
        result = await self.orchestrator.recall(
            query=claim.strip(),
            top_k=10,
            tenant_id=tenant_id,
            mode_override=QueryModeV2.VERIFICATION,
        )

        return {
            "claim": claim,
            "verification_status": "unknown",  # Would be: confirmed/contradicted/uncertain
            "supporting_evidence": [
                {
                    "id": c.item_id,
                    "text": c.text,
                    "evidence_score": c.manifold_scores.evidence,
                }
                for c in result.candidates[:5]
            ],
            "contradicting_evidence": [],
            "confidence": result.confidence,
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
        # Would query database for counts
        return {
            "tenant_id": tenant_id,
            "embeddings": {
                "topic": 0,
                "claim": 0,
                "procedure": 0,
            },
            "promoted_objects": 0,
            "canonical_claims": 0,
            "queries_today": 0,
            "cache_hit_rate": 0.0,
            "note": "Stats not available in isolated module",
        }


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
