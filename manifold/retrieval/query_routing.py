"""Graph-based query routing (MAGMA-style).

Routes queries to optimal indexes based on query patterns and intent.
Different query types benefit from different indexes:
- "what caused X" → causal_relations
- "how to X" → procedures
- "who is X" → entities
- general queries → segments
"""
import re
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional

logger = logging.getLogger("gami.retrieval.routing")


class IndexType(str, Enum):
    """Available indexes for query routing."""
    SEGMENTS = "segments"      # Raw text segments
    ENTITIES = "entities"      # Entity knowledge base
    RELATIONS = "relations"    # Graph relationships
    CAUSAL = "causal"          # Cause-effect relations
    PROCEDURES = "procedures"  # How-to/step content
    CLAIMS = "claims"          # Factual assertions
    MEMORIES = "memories"      # Assistant memories
    CLUSTERS = "clusters"      # Memory cluster abstractions


@dataclass
class QueryRouting:
    """Routing decision for a query."""
    primary_index: IndexType
    secondary_indexes: List[IndexType] = field(default_factory=list)
    index_weights: Dict[IndexType, float] = field(default_factory=dict)
    routing_reason: str = ""
    routing_confidence: float = 0.5
    temporal_hint: Optional[str] = None  # e.g., "recent", "historical"
    entity_expansion: bool = False

    def should_query_index(self, index: IndexType, threshold: float = 0.1) -> bool:
        """Check if an index should be queried based on weight."""
        return self.index_weights.get(index, 0.0) >= threshold

    def get_active_indexes(self, threshold: float = 0.1) -> List[IndexType]:
        """Get all indexes above the weight threshold."""
        return [idx for idx, w in self.index_weights.items() if w >= threshold]


class RoutingPatterns:
    """Regex patterns for index routing."""

    CAUSAL = re.compile(
        r"\b(caused|because|due to|leads to|results? in|effect of|"
        r"why did|why does|why is|reason for|consequence|trigger|"
        r"what caused|what made|what led to|root cause)\b",
        re.IGNORECASE,
    )

    PROCEDURAL = re.compile(
        r"\b(how to|how do|how can|steps to|instructions|guide|"
        r"setup|configure|install|deploy|run|execute|create|build|"
        r"procedure|workflow|process for|way to)\b",
        re.IGNORECASE,
    )

    RELATIONAL = re.compile(
        r"\b(related to|connected to|linked to|associated with|"
        r"between .+ and|relationship|connection|dependencies|"
        r"what .+ uses|what uses|who works with)\b",
        re.IGNORECASE,
    )

    ENTITY_LOOKUP = re.compile(
        r"\b(what is|who is|tell me about|describe|details on|"
        r"info about|information on|explain|definition of|"
        r"what are the .+ of)\b",
        re.IGNORECASE,
    )

    CLAIM_VERIFICATION = re.compile(
        r"\b(is it true|verify|confirm|check if|fact.?check|"
        r"did .+ really|was .+ actually|prove|evidence for|"
        r"according to|source for)\b",
        re.IGNORECASE,
    )

    TEMPORAL_RECENT = re.compile(
        r"\b(recent|latest|last|today|yesterday|this week|"
        r"just|newly|current|now)\b",
        re.IGNORECASE,
    )

    TEMPORAL_HISTORICAL = re.compile(
        r"\b(when did|history|historical|originally|"
        r"in the past|previously|before|ago|back when)\b",
        re.IGNORECASE,
    )

    MEMORY_PREFERENCE = re.compile(
        r"\b(prefer|preference|like|dislike|want|don't want|"
        r"always|never|usually|style|habit)\b",
        re.IGNORECASE,
    )


class QueryRouter:
    """Route queries to optimal indexes based on patterns."""

    def __init__(self):
        self.patterns = RoutingPatterns()

    def route(self, query: str) -> QueryRouting:
        """Determine routing for a query.

        Args:
            query: The search query

        Returns:
            QueryRouting with index weights and metadata
        """
        # Initialize weights
        weights: Dict[IndexType, float] = {idx: 0.0 for idx in IndexType}

        # Base weights - always search some indexes
        weights[IndexType.SEGMENTS] = 0.3
        weights[IndexType.MEMORIES] = 0.2

        # Pattern-based routing
        primary = IndexType.SEGMENTS
        reason = "Default routing"
        confidence = 0.5

        if self.patterns.CAUSAL.search(query):
            weights[IndexType.CAUSAL] = 0.5
            weights[IndexType.RELATIONS] = 0.3
            weights[IndexType.CLAIMS] = 0.2
            primary = IndexType.CAUSAL
            reason = "Causal query pattern detected"
            confidence = 0.8

        elif self.patterns.PROCEDURAL.search(query):
            weights[IndexType.PROCEDURES] = 0.5
            weights[IndexType.SEGMENTS] = 0.4
            primary = IndexType.PROCEDURES
            reason = "Procedural/how-to query"
            confidence = 0.85

        elif self.patterns.RELATIONAL.search(query):
            weights[IndexType.RELATIONS] = 0.4
            weights[IndexType.ENTITIES] = 0.4
            weights[IndexType.SEGMENTS] = 0.2
            primary = IndexType.RELATIONS
            reason = "Relational query pattern"
            confidence = 0.75

        elif self.patterns.ENTITY_LOOKUP.search(query):
            weights[IndexType.ENTITIES] = 0.5
            weights[IndexType.CLAIMS] = 0.3
            weights[IndexType.SEGMENTS] = 0.2
            primary = IndexType.ENTITIES
            reason = "Entity lookup query"
            confidence = 0.8

        elif self.patterns.CLAIM_VERIFICATION.search(query):
            weights[IndexType.CLAIMS] = 0.5
            weights[IndexType.SEGMENTS] = 0.3
            primary = IndexType.CLAIMS
            reason = "Claim verification query"
            confidence = 0.75

        elif self.patterns.MEMORY_PREFERENCE.search(query):
            weights[IndexType.MEMORIES] = 0.5
            weights[IndexType.SEGMENTS] = 0.3
            primary = IndexType.MEMORIES
            reason = "Memory/preference query"
            confidence = 0.7

        # Check temporal hints
        temporal_hint = None
        if self.patterns.TEMPORAL_RECENT.search(query):
            temporal_hint = "recent"
        elif self.patterns.TEMPORAL_HISTORICAL.search(query):
            temporal_hint = "historical"

        # Entity expansion for entity-focused queries
        entity_expansion = primary in (IndexType.ENTITIES, IndexType.RELATIONS)

        # Normalize weights
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        # Determine secondary indexes
        secondary = [
            idx for idx, w in weights.items()
            if idx != primary and w >= 0.15
        ]

        return QueryRouting(
            primary_index=primary,
            secondary_indexes=secondary,
            index_weights=weights,
            routing_reason=reason,
            routing_confidence=confidence,
            temporal_hint=temporal_hint,
            entity_expansion=entity_expansion,
        )


# Module-level instance for convenience
_router: Optional[QueryRouter] = None


def get_router() -> QueryRouter:
    """Get or create the query router singleton."""
    global _router
    if _router is None:
        _router = QueryRouter()
    return _router


def route_query(query: str) -> QueryRouting:
    """Route a query to optimal indexes.

    Convenience function that uses the singleton router.
    """
    return get_router().route(query)
