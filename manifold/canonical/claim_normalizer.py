"""Claim normalizer — converts prose claims to structured SPO form.

Takes natural language claims like:
    "The Vietnamese Communist Party exercised significant control over the CPK"

And converts them to canonical form:
    subject: "Vietnamese Communist Party"
    predicate: "exercised control over"
    object: "CPK"
    modality: "factual"
    qualifiers: ["significant"]
    temporal_scope: None

The canonical form is then used to generate the embedding for the claim manifold.
"""
import json
import logging
import re
from typing import Optional, Dict, Any, List

from ..models.schemas import (
    CanonicalClaimSchema,
    Modality,
)

logger = logging.getLogger("gami.manifold.canonical.claim")


# Patterns for modality detection
NEGATION_PATTERNS = re.compile(
    r"\b(not|never|no|didn't|doesn't|won't|can't|cannot|failed to|lacks?|"
    r"absence of|without|unable to|isn't|aren't|wasn't|weren't)\b",
    re.IGNORECASE,
)
POSSIBILITY_PATTERNS = re.compile(
    r"\b(might|may|could|possibly|perhaps|probably|likely|potential|"
    r"uncertain|unclear|appears? to|seems? to|allegedly|reportedly)\b",
    re.IGNORECASE,
)

# Common qualifier words
QUALIFIER_PATTERNS = re.compile(
    r"\b(significant|major|minor|partial|complete|total|substantial|"
    r"considerable|limited|extensive|moderate|strong|weak|full|"
    r"primary|secondary|key|critical|essential|fundamental)\b",
    re.IGNORECASE,
)

# Temporal patterns
TEMPORAL_PATTERNS = re.compile(
    r"\b(in \d{4}|during \d{4}s?|since \d{4}|from \d{4}|until \d{4}|"
    r"before \d{4}|after \d{4}|through \d{4}|by \d{4}|"
    r"in the \d{4}s|during the \w+ century|"
    r"in (?:january|february|march|april|may|june|july|august|september|"
    r"october|november|december)(?: \d{4})?|"
    r"during the (?:first|second|third) (?:world war|war)|"
    r"in (?:early|mid|late) \d{4}s?|"
    r"beginning in \d{4}|ending in \d{4})\b",
    re.IGNORECASE,
)


class ClaimNormalizer:
    """Normalizes prose claims to structured SPO form.

    Uses pattern-based extraction for simple cases and LLM for complex cases.
    """

    def __init__(self, use_llm: bool = True, vllm_url: Optional[str] = None):
        """Initialize the claim normalizer.

        Args:
            use_llm: Whether to use LLM for complex extractions.
            vllm_url: URL of the vLLM server (default: from settings).
        """
        self.use_llm = use_llm
        self.vllm_url = vllm_url

    def normalize(self, claim_text: str, claim_id: Optional[str] = None) -> Optional[CanonicalClaimSchema]:
        """Normalize a prose claim to SPO form.

        Args:
            claim_text: The prose claim text.
            claim_id: Optional claim ID for reference.

        Returns:
            CanonicalClaimSchema if successful, None if extraction failed.
        """
        if not claim_text or len(claim_text.strip()) < 10:
            return None

        # Detect modality
        modality = self._detect_modality(claim_text)

        # Extract qualifiers
        qualifiers = self._extract_qualifiers(claim_text)

        # Extract temporal scope
        temporal_scope = self._extract_temporal(claim_text)

        # Try pattern-based extraction first
        spo = self._pattern_extract(claim_text)

        if spo:
            return CanonicalClaimSchema(
                claim_id=claim_id,
                subject=spo["subject"],
                predicate=spo["predicate"],
                object=spo.get("object"),
                modality=modality,
                qualifiers=qualifiers,
                temporal_scope=temporal_scope,
                confidence=0.7,  # Pattern-based confidence
            )

        # Use LLM for complex cases
        if self.use_llm:
            spo = self._llm_extract(claim_text)
            if spo:
                return CanonicalClaimSchema(
                    claim_id=claim_id,
                    subject=spo["subject"],
                    predicate=spo["predicate"],
                    object=spo.get("object"),
                    modality=modality,
                    qualifiers=qualifiers,
                    temporal_scope=temporal_scope,
                    confidence=0.8,  # LLM confidence
                )

        # Fallback: treat entire claim as subject with generic predicate
        return CanonicalClaimSchema(
            claim_id=claim_id,
            subject=claim_text[:200],
            predicate="states",
            object=None,
            modality=modality,
            qualifiers=qualifiers,
            temporal_scope=temporal_scope,
            confidence=0.3,  # Low confidence fallback
        )

    def _detect_modality(self, text: str) -> Modality:
        """Detect claim modality (factual, possible, negated)."""
        if NEGATION_PATTERNS.search(text):
            return Modality.NEGATED
        if POSSIBILITY_PATTERNS.search(text):
            return Modality.POSSIBLE
        return Modality.FACTUAL

    def _extract_qualifiers(self, text: str) -> List[str]:
        """Extract qualifier words from claim."""
        matches = QUALIFIER_PATTERNS.findall(text.lower())
        # Deduplicate and limit
        return list(set(matches))[:5]

    def _extract_temporal(self, text: str) -> Optional[str]:
        """Extract temporal scope from claim."""
        match = TEMPORAL_PATTERNS.search(text)
        if match:
            return match.group(0)
        return None

    def _pattern_extract(self, text: str) -> Optional[Dict[str, str]]:
        """Try pattern-based SPO extraction.

        Handles common sentence structures like:
        - "X is Y"
        - "X has Y"
        - "X caused Y"
        - "X controls Y"
        """
        # Clean up text
        text = text.strip().rstrip(".")

        # Pattern: "X is/was/are/were Y"
        match = re.match(
            r"^(.+?)\s+(is|was|are|were)\s+(.+)$",
            text,
            re.IGNORECASE,
        )
        if match:
            return {
                "subject": match.group(1).strip(),
                "predicate": match.group(2).lower(),
                "object": match.group(3).strip(),
            }

        # Pattern: "X has/have/had Y"
        match = re.match(
            r"^(.+?)\s+(has|have|had)\s+(.+)$",
            text,
            re.IGNORECASE,
        )
        if match:
            return {
                "subject": match.group(1).strip(),
                "predicate": match.group(2).lower(),
                "object": match.group(3).strip(),
            }

        # Pattern: "X [verb]ed Y"
        match = re.match(
            r"^(.+?)\s+(\w+ed)\s+(.+)$",
            text,
            re.IGNORECASE,
        )
        if match:
            return {
                "subject": match.group(1).strip(),
                "predicate": match.group(2).lower(),
                "object": match.group(3).strip(),
            }

        # Pattern: "X [verb]s Y"
        match = re.match(
            r"^(.+?)\s+(\w+s)\s+(.+)$",
            text,
            re.IGNORECASE,
        )
        if match and len(match.group(2)) > 2:  # Avoid matching plurals
            return {
                "subject": match.group(1).strip(),
                "predicate": match.group(2).lower(),
                "object": match.group(3).strip(),
            }

        return None

    def _llm_extract(self, text: str) -> Optional[Dict[str, str]]:
        """Use LLM to extract SPO structure.

        NOTE: This method requires vLLM to be running.
        In the isolated manifold module, this is a stub that returns None.
        Actual implementation will be connected during activation.
        """
        # STUB: LLM extraction not connected in isolated module
        # This will be implemented when manifold system is activated
        logger.debug("LLM extraction not available in isolated module")
        return None


def normalize_claim(
    claim_text: str,
    claim_id: Optional[str] = None,
    use_llm: bool = False,
) -> Optional[CanonicalClaimSchema]:
    """Convenience function for claim normalization.

    Args:
        claim_text: The prose claim to normalize.
        claim_id: Optional claim ID.
        use_llm: Whether to use LLM (disabled by default in isolated module).

    Returns:
        CanonicalClaimSchema if successful, None otherwise.
    """
    normalizer = ClaimNormalizer(use_llm=use_llm)
    return normalizer.normalize(claim_text, claim_id)


def batch_normalize_claims(
    claims: List[Dict[str, str]],
    use_llm: bool = False,
) -> List[Optional[CanonicalClaimSchema]]:
    """Normalize multiple claims.

    Args:
        claims: List of dicts with 'text' and optional 'claim_id'.
        use_llm: Whether to use LLM for complex cases.

    Returns:
        List of CanonicalClaimSchema or None for each claim.
    """
    normalizer = ClaimNormalizer(use_llm=use_llm)
    results = []
    for claim in claims:
        result = normalizer.normalize(
            claim.get("text", ""),
            claim.get("claim_id"),
        )
        results.append(result)
    return results
