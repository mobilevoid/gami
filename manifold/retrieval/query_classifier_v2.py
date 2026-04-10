"""Enhanced query classifier for manifold-aware retrieval.

Extends the original query classifier to:
1. Support additional query modes (comparison, procedure)
2. Compute manifold weight vector α(q) for each query
3. Return confidence scores for classification
4. Support manual mode override

The classifier uses pattern-based rules for fast classification and
optionally falls back to LLM for ambiguous queries.
"""
import logging
import re
from typing import Optional, Dict, Any

from ..models.schemas import (
    QueryModeV2,
    QueryClassificationV2,
    ManifoldWeights,
    ALPHA_WEIGHTS,
    get_alpha_weights,
)

logger = logging.getLogger("gami.manifold.retrieval.classifier")


# ---------------------------------------------------------------------------
# Pattern Definitions
# ---------------------------------------------------------------------------

class ClassificationPatterns:
    """Regex patterns for query classification."""

    TIMELINE = re.compile(
        r"\b(when|timeline|history|chronolog|before|after|sequence|"
        r"first|last|recent|march|april|january|february|may|june|july|august|"
        r"september|october|november|december|date|year \d{4}|in \d{4}|"
        r"yesterday|today|tomorrow|last (?:week|month|year)|"
        r"what happened|occurred)\b",
        re.IGNORECASE,
    )

    ENTITY = re.compile(
        r"\b(who is|what is|tell me about|describe|details on|info on|"
        r"explain|define|meaning of|what does .+ mean)\b",
        re.IGNORECASE,
    )

    VERIFICATION = re.compile(
        r"\b(is it true|verify|confirm|check if|did .+ really|"
        r"was .+ actually|correct that|support|contradict|"
        r"is this accurate|fact.?check|prove|evidence for|evidence against)\b",
        re.IGNORECASE,
    )

    SYNTHESIS = re.compile(
        r"\b(summarize|overview|summary|combine|integrate|"
        r"big picture|main points|key takeaways|essence of)\b",
        re.IGNORECASE,
    )

    COMPARISON = re.compile(
        r"\b(compare|contrast|difference|differences|similar|similarities|"
        r"versus|vs\.?|better|worse|pros and cons|advantages|disadvantages|"
        r"how does .+ differ|what.+s the difference|which is)\b",
        re.IGNORECASE,
    )

    PROCEDURE = re.compile(
        r"\b(how to|how do i|how can i|steps to|instructions|"
        r"guide|tutorial|procedure|process for|way to|method for|"
        r"walk me through|show me how|help me)\b",
        re.IGNORECASE,
    )

    REPORT = re.compile(
        r"\b(report|all .+ about|comprehensive|everything|full list|"
        r"enumerate|inventory|catalog|complete|exhaustive|detailed|"
        r"all instances|every|each)\b",
        re.IGNORECASE,
    )

    MEMORY = re.compile(
        r"\b(remember|recall|you told me|we discussed|last time|"
        r"previous conversation|session|forgot|memory|you said|"
        r"you mentioned|earlier you|we talked about)\b",
        re.IGNORECASE,
    )

    FACT = re.compile(
        r"\b(what|where|which|whose|how many|how much|"
        r"password|ip address|port|version|name of|value of|"
        r"setting|config|credential|key|secret|token)\b",
        re.IGNORECASE,
    )


# ---------------------------------------------------------------------------
# Pattern-based Classifier
# ---------------------------------------------------------------------------

def _classify_by_patterns(query: str) -> QueryClassificationV2:
    """Classify query using regex patterns.

    Returns classification with mode, granularity, and manifold weights.
    """
    q = query.strip()
    patterns = ClassificationPatterns

    # Check patterns in order of specificity
    # (More specific patterns should be checked first)

    # Memory queries (high priority - reference to prior conversation)
    if patterns.MEMORY.search(q):
        return QueryClassificationV2(
            mode=QueryModeV2.ASSISTANT_MEMORY,
            granularity="fine",
            needs_citation=False,
            manifold_weights=get_alpha_weights(QueryModeV2.ASSISTANT_MEMORY),
            confidence=0.85,
            reasoning="Pattern: memory/recall reference",
        )

    # Verification queries
    if patterns.VERIFICATION.search(q):
        return QueryClassificationV2(
            mode=QueryModeV2.VERIFICATION,
            granularity="fine",
            needs_citation=True,
            manifold_weights=get_alpha_weights(QueryModeV2.VERIFICATION),
            confidence=0.85,
            reasoning="Pattern: verification/fact-check",
        )

    # Procedure queries (how-to)
    if patterns.PROCEDURE.search(q):
        return QueryClassificationV2(
            mode=QueryModeV2.PROCEDURE,
            granularity="medium",
            needs_citation=True,
            manifold_weights=get_alpha_weights(QueryModeV2.PROCEDURE),
            confidence=0.85,
            reasoning="Pattern: procedural/how-to",
        )

    # Comparison queries
    if patterns.COMPARISON.search(q):
        return QueryClassificationV2(
            mode=QueryModeV2.COMPARISON,
            granularity="medium",
            needs_citation=True,
            manifold_weights=get_alpha_weights(QueryModeV2.COMPARISON),
            confidence=0.85,
            reasoning="Pattern: comparison",
        )

    # Timeline queries
    if patterns.TIMELINE.search(q):
        return QueryClassificationV2(
            mode=QueryModeV2.TIMELINE,
            granularity="fine",
            needs_citation=True,
            manifold_weights=get_alpha_weights(QueryModeV2.TIMELINE),
            confidence=0.80,
            reasoning="Pattern: temporal/timeline",
        )

    # Synthesis queries
    if patterns.SYNTHESIS.search(q):
        return QueryClassificationV2(
            mode=QueryModeV2.SYNTHESIS,
            granularity="coarse",
            needs_citation=True,
            manifold_weights=get_alpha_weights(QueryModeV2.SYNTHESIS),
            confidence=0.80,
            reasoning="Pattern: synthesis/summary",
        )

    # Report queries
    if patterns.REPORT.search(q):
        return QueryClassificationV2(
            mode=QueryModeV2.REPORT,
            granularity="coarse",
            needs_citation=True,
            manifold_weights=get_alpha_weights(QueryModeV2.REPORT),
            confidence=0.80,
            reasoning="Pattern: report/comprehensive",
        )

    # Fact lookup (credentials, configs, specific values)
    if patterns.FACT.search(q):
        return QueryClassificationV2(
            mode=QueryModeV2.FACT_LOOKUP,
            granularity="fine",
            needs_citation=True,
            manifold_weights=get_alpha_weights(QueryModeV2.FACT_LOOKUP),
            confidence=0.75,
            reasoning="Pattern: fact/credential lookup",
        )

    # Entity-centric queries
    if patterns.ENTITY.search(q):
        return QueryClassificationV2(
            mode=QueryModeV2.FACT_LOOKUP,  # Entity lookup is a type of fact lookup
            granularity="medium",
            needs_citation=True,
            manifold_weights=get_alpha_weights(QueryModeV2.FACT_LOOKUP),
            confidence=0.70,
            reasoning="Pattern: entity description",
        )

    # Default: fact lookup
    return QueryClassificationV2(
        mode=QueryModeV2.FACT_LOOKUP,
        granularity="medium",
        needs_citation=True,
        manifold_weights=get_alpha_weights(QueryModeV2.FACT_LOOKUP),
        confidence=0.50,
        reasoning="Default: no strong pattern match",
    )


# ---------------------------------------------------------------------------
# Query Classifier V2
# ---------------------------------------------------------------------------

class QueryClassifierV2:
    """Enhanced query classifier with manifold weight computation.

    Attributes:
        use_llm: Whether to use LLM for ambiguous queries.
        confidence_threshold: Below this confidence, try LLM if available.
    """

    def __init__(
        self,
        use_llm: bool = False,
        confidence_threshold: float = 0.6,
    ):
        """Initialize the classifier.

        Args:
            use_llm: Whether to use LLM fallback for low-confidence cases.
            confidence_threshold: Confidence below which to try LLM.
        """
        self.use_llm = use_llm
        self.confidence_threshold = confidence_threshold

    def classify(
        self,
        query: str,
        mode_override: Optional[QueryModeV2] = None,
        weight_override: Optional[ManifoldWeights] = None,
    ) -> QueryClassificationV2:
        """Classify a query and compute manifold weights.

        Args:
            query: The query text.
            mode_override: Optional mode override (skip classification).
            weight_override: Optional weight override (use custom weights).

        Returns:
            QueryClassificationV2 with mode and manifold weights.
        """
        # Handle mode override
        if mode_override and mode_override != QueryModeV2.AUTO:
            classification = QueryClassificationV2(
                mode=mode_override,
                granularity="medium",
                needs_citation=True,
                manifold_weights=get_alpha_weights(mode_override),
                confidence=1.0,
                reasoning="Mode override",
            )
        else:
            # Pattern-based classification
            classification = _classify_by_patterns(query)

            # Try LLM for low-confidence cases
            if (
                self.use_llm
                and classification.confidence < self.confidence_threshold
            ):
                llm_classification = self._llm_classify(query)
                if llm_classification:
                    classification = llm_classification

        # Apply weight override if provided
        if weight_override:
            classification.manifold_weights = weight_override.normalize()
            classification.reasoning = (classification.reasoning or "") + " (weights overridden)"

        return classification

    def _llm_classify(self, query: str) -> Optional[QueryClassificationV2]:
        """Use LLM for query classification.

        NOTE: This is a stub in the isolated module.
        Actual implementation will be connected during activation.
        """
        # STUB: LLM classification not connected in isolated module
        logger.debug("LLM classification not available in isolated module")
        return None


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------

def classify_query_v2(
    query: str,
    mode_override: Optional[QueryModeV2] = None,
    weight_override: Optional[ManifoldWeights] = None,
    use_llm: bool = False,
) -> QueryClassificationV2:
    """Convenience function for query classification.

    Args:
        query: The query text.
        mode_override: Optional mode override.
        weight_override: Optional weight override.
        use_llm: Whether to use LLM (disabled by default in isolated module).

    Returns:
        QueryClassificationV2 with mode and manifold weights.
    """
    classifier = QueryClassifierV2(use_llm=use_llm)
    return classifier.classify(query, mode_override, weight_override)


def get_manifold_weights(
    query: str,
    mode: Optional[QueryModeV2] = None,
) -> ManifoldWeights:
    """Get manifold weights for a query.

    Args:
        query: The query text (used if mode not provided).
        mode: Optional explicit mode.

    Returns:
        ManifoldWeights for the query/mode.
    """
    if mode and mode != QueryModeV2.AUTO:
        return get_alpha_weights(mode)

    classification = classify_query_v2(query)
    return classification.manifold_weights


def explain_classification(
    query: str,
) -> Dict[str, Any]:
    """Get detailed explanation of query classification.

    Args:
        query: The query text.

    Returns:
        Dict with classification details and reasoning.
    """
    classification = classify_query_v2(query)

    return {
        "query": query,
        "mode": classification.mode.value,
        "granularity": classification.granularity,
        "needs_citation": classification.needs_citation,
        "confidence": classification.confidence,
        "reasoning": classification.reasoning,
        "manifold_weights": classification.manifold_weights.to_dict(),
        "weight_explanation": {
            "topic": f"General semantic similarity weight: {classification.manifold_weights.topic:.2f}",
            "claim": f"Propositional equivalence weight: {classification.manifold_weights.claim:.2f}",
            "procedure": f"Procedural similarity weight: {classification.manifold_weights.procedure:.2f}",
            "relation": f"Graph relationship weight: {classification.manifold_weights.relation:.2f}",
            "time": f"Temporal compatibility weight: {classification.manifold_weights.time:.2f}",
            "evidence": f"Evidence quality weight: {classification.manifold_weights.evidence:.2f}",
        },
    }
