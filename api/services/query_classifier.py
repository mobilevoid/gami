"""Query classifier for GAMI retrieval orchestrator.

Classifies queries into retrieval modes using Ollama qwen3:8b (fast, CPU).
Falls back to keyword heuristics if Ollama is unavailable.
"""
import json
import logging
import re
from enum import Enum
from typing import Optional

from pydantic import BaseModel

from api.llm.client import call_ollama, parse_json_from_llm

logger = logging.getLogger("gami.services.query_classifier")


class QueryMode(str, Enum):
    FACTUAL = "factual"
    SYNTHESIS = "synthesis"
    TIMELINE = "timeline"
    ENTITY_CENTRIC = "entity_centric"
    ASSISTANT_MEMORY = "assistant_memory"
    VERIFICATION = "verification"
    REPORT = "report"


class QueryClassification(BaseModel):
    mode: QueryMode = QueryMode.FACTUAL
    granularity: str = "medium"  # fine, medium, coarse
    needs_citation: bool = True


# Keyword patterns for fallback classification
_TIMELINE_PATTERNS = re.compile(
    r"\b(when|timeline|history|chronolog|before|after|sequence|"
    r"first|last|recent|march|april|january|february|date)\b",
    re.IGNORECASE,
)
_ENTITY_PATTERNS = re.compile(
    r"\b(who is|what is|tell me about|describe|details on|info on|"
    r"explain)\b",
    re.IGNORECASE,
)
_VERIFICATION_PATTERNS = re.compile(
    r"\b(is it true|verify|confirm|check if|did .+ really|"
    r"was .+ actually|correct that|support|contradict)\b",
    re.IGNORECASE,
)
_SYNTHESIS_PATTERNS = re.compile(
    r"\b(summarize|overview|compare|contrast|relationship between|"
    r"how .+ relate|differences? between|similarities)\b",
    re.IGNORECASE,
)
_REPORT_PATTERNS = re.compile(
    r"\b(report|all .+ about|comprehensive|everything|full list|"
    r"enumerate|inventory|catalog)\b",
    re.IGNORECASE,
)
_MEMORY_PATTERNS = re.compile(
    r"\b(remember|recall|you told me|we discussed|last time|"
    r"previous conversation|session|forgot|memory)\b",
    re.IGNORECASE,
)


def _classify_by_keywords(query: str) -> QueryClassification:
    """Fallback heuristic classification based on keyword patterns."""
    q = query.strip()

    if _MEMORY_PATTERNS.search(q):
        return QueryClassification(
            mode=QueryMode.ASSISTANT_MEMORY,
            granularity="fine",
            needs_citation=False,
        )
    if _VERIFICATION_PATTERNS.search(q):
        return QueryClassification(
            mode=QueryMode.VERIFICATION,
            granularity="fine",
            needs_citation=True,
        )
    if _TIMELINE_PATTERNS.search(q):
        return QueryClassification(
            mode=QueryMode.TIMELINE,
            granularity="fine",
            needs_citation=True,
        )
    if _SYNTHESIS_PATTERNS.search(q):
        return QueryClassification(
            mode=QueryMode.SYNTHESIS,
            granularity="coarse",
            needs_citation=True,
        )
    if _REPORT_PATTERNS.search(q):
        return QueryClassification(
            mode=QueryMode.REPORT,
            granularity="coarse",
            needs_citation=True,
        )
    if _ENTITY_PATTERNS.search(q):
        return QueryClassification(
            mode=QueryMode.ENTITY_CENTRIC,
            granularity="medium",
            needs_citation=True,
        )

    # Default: factual lookup
    return QueryClassification(
        mode=QueryMode.FACTUAL,
        granularity="medium",
        needs_citation=True,
    )


async def classify_query(query: str) -> QueryClassification:
    """Classify a query into a retrieval mode.

    Uses Ollama qwen3:8b for classification. Falls back to keyword
    heuristics if the LLM is unavailable or returns unusable output.
    """
    try:
        prompt = (
            f"/no_think\n"
            f"Classify this search query into exactly one mode.\n\n"
            f"Query: {query}\n\n"
            f"Modes:\n"
            f"- factual: looking up a specific fact or answer\n"
            f"- synthesis: combining information from multiple sources\n"
            f"- timeline: asking about when things happened or ordering of events\n"
            f"- entity_centric: asking about a specific person/place/system/service\n"
            f"- assistant_memory: referencing prior conversations or stored memories\n"
            f"- verification: checking if a claim is true or supported\n"
            f"- report: requesting comprehensive coverage of a topic\n\n"
            f"Return ONLY this JSON (no explanation, no markdown):\n"
            f'{{"mode": "<mode>", "granularity": "fine|medium|coarse", "needs_citation": true|false}}'
        )

        raw = await call_ollama(
            prompt,
            model="qwen3:8b",
            max_tokens=128,
            system_prompt="/no_think You are a query classifier. Return ONLY valid JSON.",
        )

        parsed = parse_json_from_llm(raw)
        if parsed and isinstance(parsed, dict):
            mode_str = parsed.get("mode", "factual")
            try:
                mode = QueryMode(mode_str)
            except ValueError:
                mode = QueryMode.FACTUAL

            granularity = parsed.get("granularity", "medium")
            if granularity not in ("fine", "medium", "coarse"):
                granularity = "medium"

            needs_citation = parsed.get("needs_citation", True)
            if not isinstance(needs_citation, bool):
                needs_citation = True

            return QueryClassification(
                mode=mode,
                granularity=granularity,
                needs_citation=needs_citation,
            )

    except Exception as exc:
        logger.warning("Ollama classification failed, using keyword fallback: %s", exc)

    return _classify_by_keywords(query)
