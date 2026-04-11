"""Query Router — classifies query intent and assigns manifold weights.

Maps query patterns to manifold α-weights:
- debugging: PROCEDURE (35%), TIME (15%), TOPIC (15%), RELATION (15%)
- recalling: TOPIC (30%), TIME (25%), CLAIM (25%)
- exploring: TOPIC (40%), CLAIM (20%), RELATION (20%)
- planning: PROCEDURE (30%), TOPIC (20%), RELATION (20%)
- verifying: CLAIM (35%), EVIDENCE (25%), TOPIC (20%)
- default: TOPIC (40%), CLAIM (20%), PROCEDURE (15%), others split

The router uses pattern matching for fast classification with optional
LLM fallback for ambiguous queries.
"""
import logging
import re
from enum import Enum
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger("gami.manifold.query_router")


class QueryIntent(str, Enum):
    """Query intent categories."""
    DEBUGGING = "debugging"
    RECALLING = "recalling"
    EXPLORING = "exploring"
    PLANNING = "planning"
    VERIFYING = "verifying"
    PROCEDURAL = "procedural"
    DEFAULT = "default"


class ManifoldType(str, Enum):
    """The 6 semantic manifold types."""
    TOPIC = "TOPIC"
    CLAIM = "CLAIM"
    PROCEDURE = "PROCEDURE"
    RELATION = "RELATION"
    TIME = "TIME"
    EVIDENCE = "EVIDENCE"


@dataclass
class ManifoldWeights:
    """Weights for each manifold type."""
    topic: float = 0.0
    claim: float = 0.0
    procedure: float = 0.0
    relation: float = 0.0
    time: float = 0.0
    evidence: float = 0.0

    def to_dict(self) -> Dict[ManifoldType, float]:
        return {
            ManifoldType.TOPIC: self.topic,
            ManifoldType.CLAIM: self.claim,
            ManifoldType.PROCEDURE: self.procedure,
            ManifoldType.RELATION: self.relation,
            ManifoldType.TIME: self.time,
            ManifoldType.EVIDENCE: self.evidence,
        }

    @classmethod
    def from_dict(cls, d: Dict[ManifoldType, float]) -> "ManifoldWeights":
        return cls(
            topic=d.get(ManifoldType.TOPIC, 0.0),
            claim=d.get(ManifoldType.CLAIM, 0.0),
            procedure=d.get(ManifoldType.PROCEDURE, 0.0),
            relation=d.get(ManifoldType.RELATION, 0.0),
            time=d.get(ManifoldType.TIME, 0.0),
            evidence=d.get(ManifoldType.EVIDENCE, 0.0),
        )

    def normalize(self) -> "ManifoldWeights":
        """Normalize weights to sum to 1.0."""
        total = self.topic + self.claim + self.procedure + self.relation + self.time + self.evidence
        if total <= 0:
            return ManifoldWeights(topic=1.0)  # Fallback to topic-only
        return ManifoldWeights(
            topic=self.topic / total,
            claim=self.claim / total,
            procedure=self.procedure / total,
            relation=self.relation / total,
            time=self.time / total,
            evidence=self.evidence / total,
        )


# ---------------------------------------------------------------------------
# Intent → Manifold Weight Profiles
# ---------------------------------------------------------------------------

MANIFOLD_WEIGHT_PROFILES: Dict[QueryIntent, ManifoldWeights] = {
    QueryIntent.DEBUGGING: ManifoldWeights(
        topic=0.15,
        claim=0.10,
        procedure=0.35,  # How to fix
        relation=0.15,
        time=0.15,       # Recent errors
        evidence=0.10,
    ),
    QueryIntent.RECALLING: ManifoldWeights(
        topic=0.30,
        claim=0.25,
        procedure=0.05,
        relation=0.10,
        time=0.25,       # When did we discuss
        evidence=0.05,
    ),
    QueryIntent.EXPLORING: ManifoldWeights(
        topic=0.40,
        claim=0.20,
        procedure=0.10,
        relation=0.20,
        time=0.05,
        evidence=0.05,
    ),
    QueryIntent.PLANNING: ManifoldWeights(
        topic=0.20,
        claim=0.15,
        procedure=0.30,
        relation=0.20,
        time=0.05,
        evidence=0.10,
    ),
    QueryIntent.VERIFYING: ManifoldWeights(
        topic=0.20,
        claim=0.35,
        procedure=0.05,
        relation=0.10,
        time=0.05,
        evidence=0.25,
    ),
    QueryIntent.PROCEDURAL: ManifoldWeights(
        topic=0.15,
        claim=0.10,
        procedure=0.45,
        relation=0.15,
        time=0.10,
        evidence=0.05,
    ),
    QueryIntent.DEFAULT: ManifoldWeights(
        topic=0.40,
        claim=0.20,
        procedure=0.15,
        relation=0.10,
        time=0.10,
        evidence=0.05,
    ),
}


# ---------------------------------------------------------------------------
# Pattern-Based Intent Classification
# ---------------------------------------------------------------------------

INTENT_PATTERNS: Dict[QueryIntent, List[str]] = {
    QueryIntent.DEBUGGING: [
        r"\b(?:error|fail(?:ed|ure)?|bug|crash|exception|traceback)\b",
        r"\b(?:not working|broken|wrong|issue|problem)\b",
        r"\b(?:fix|debug|resolve|troubleshoot)\b",
        r"(?:why (?:is|does|did).+(?:fail|error|crash|break))",
        r"(?:getting (?:an?|this) error)",
    ],
    QueryIntent.RECALLING: [
        r"\b(?:remember|recall|previously|earlier|before|last time)\b",
        r"(?:you (?:told|said|mentioned|showed) me)",
        r"(?:we (?:discussed|talked about|covered))",
        r"(?:what (?:was|were|did we))",
        r"\b(?:when did (?:we|i|you))\b",
        r"\b(?:history|past|ago)\b",
    ],
    QueryIntent.EXPLORING: [
        r"(?:what is|what are|what does)\b",
        r"\b(?:explain|describe|tell me about|overview)\b",
        r"\b(?:how does|how do).+(?:work|function)\b",
        r"(?:can you (?:explain|describe|tell me))",
        r"\b(?:definition|meaning|concept)\b",
    ],
    QueryIntent.PLANNING: [
        r"(?:how (?:should|do|can) (?:we|i))\b",
        r"\b(?:plan|strategy|approach|design|architect)\b",
        r"\b(?:steps to|best way to|implement)\b",
        r"(?:what's the (?:best|right) way)",
        r"\b(?:roadmap|milestones|timeline)\b",
    ],
    QueryIntent.VERIFYING: [
        r"\b(?:is it true|is that correct|verify|confirm|check)\b",
        r"(?:(?:did|does|is|are).+(?:true|correct|right|accurate))",
        r"\b(?:fact-check|validate|prove)\b",
        r"(?:according to|based on|evidence)",
    ],
    QueryIntent.PROCEDURAL: [
        r"(?:how (?:to|do i|can i))\b",
        r"\b(?:steps|instructions|guide|tutorial|walkthrough)\b",
        r"\b(?:run|execute|install|create|configure|set up)\b",
        r"(?:show me how)",
        r"\b(?:command|script|code to)\b",
    ],
}


class QueryRouter:
    """Routes queries to appropriate manifold weights based on intent."""

    def __init__(self, use_llm_fallback: bool = False):
        """Initialize the query router.

        Args:
            use_llm_fallback: Whether to use LLM for ambiguous queries
        """
        self.use_llm_fallback = use_llm_fallback
        self._compiled_patterns: Dict[QueryIntent, List[re.Pattern]] = {}

        # Compile patterns
        for intent, patterns in INTENT_PATTERNS.items():
            self._compiled_patterns[intent] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

    def classify_intent(self, query: str) -> Tuple[QueryIntent, float]:
        """Classify query intent using pattern matching.

        Args:
            query: The query text

        Returns:
            Tuple of (intent, confidence)
        """
        if not query or len(query.strip()) < 3:
            return QueryIntent.DEFAULT, 0.3

        scores: Dict[QueryIntent, float] = {intent: 0.0 for intent in QueryIntent}

        # Score each intent based on pattern matches
        for intent, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(query):
                    scores[intent] += 1.0

        # Find best match
        best_intent = max(scores, key=scores.get)
        best_score = scores[best_intent]

        if best_score == 0:
            return QueryIntent.DEFAULT, 0.5

        # Normalize confidence
        total_score = sum(scores.values())
        confidence = best_score / total_score if total_score > 0 else 0.5

        # Require minimum confidence
        if confidence < 0.4:
            return QueryIntent.DEFAULT, confidence

        return best_intent, confidence

    def get_manifold_weights(
        self,
        query: str,
        intent_override: Optional[QueryIntent] = None,
    ) -> Tuple[ManifoldWeights, QueryIntent, float]:
        """Get manifold weights for a query.

        Args:
            query: The query text
            intent_override: Optional explicit intent

        Returns:
            Tuple of (weights, detected_intent, confidence)
        """
        if intent_override:
            intent = intent_override
            confidence = 1.0
        else:
            intent, confidence = self.classify_intent(query)

        weights = MANIFOLD_WEIGHT_PROFILES.get(intent, MANIFOLD_WEIGHT_PROFILES[QueryIntent.DEFAULT])

        logger.debug(f"Query intent: {intent.value} (conf={confidence:.2f})")

        return weights, intent, confidence

    def blend_weights(
        self,
        base_weights: ManifoldWeights,
        override_weights: Dict[ManifoldType, float],
        blend_factor: float = 0.5,
    ) -> ManifoldWeights:
        """Blend base weights with custom overrides.

        Args:
            base_weights: Base weights from intent classification
            override_weights: Custom weight overrides
            blend_factor: How much to weight overrides (0=base, 1=override)

        Returns:
            Blended weights
        """
        result = ManifoldWeights()
        base_dict = base_weights.to_dict()

        for manifold in ManifoldType:
            base_val = base_dict.get(manifold, 0.0)
            override_val = override_weights.get(manifold, base_val)
            blended = (1 - blend_factor) * base_val + blend_factor * override_val
            setattr(result, manifold.value.lower(), blended)

        return result.normalize()


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------

_router_instance: Optional[QueryRouter] = None


def get_query_router() -> QueryRouter:
    """Get singleton QueryRouter instance."""
    global _router_instance
    if _router_instance is None:
        _router_instance = QueryRouter()
    return _router_instance


def route_query(query: str) -> Tuple[Dict[ManifoldType, float], str, float]:
    """Route a query to manifold weights.

    Args:
        query: The query text

    Returns:
        Tuple of (weights_dict, intent_name, confidence)
    """
    router = get_query_router()
    weights, intent, confidence = router.get_manifold_weights(query)
    return weights.to_dict(), intent.value, confidence
