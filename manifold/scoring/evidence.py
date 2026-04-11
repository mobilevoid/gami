"""Evidence score computation for verification queries.

The evidence manifold is a 5-dimensional composite score:
1. source_authority - trustworthiness of the source
2. corroboration_count - number of independent confirmations
3. recency - how recent the evidence is
4. specificity - how specific vs. vague the claim is
5. contradiction_ratio - proportion of contradicting evidence

These combine into a single evidence score used for verification
and fact-checking queries.
"""
from dataclasses import dataclass
from typing import List, Optional
import math
from datetime import datetime, timedelta


@dataclass
class EvidenceFactors:
    """Input factors for evidence scoring."""
    source_authority: float = 0.5  # [0, 1]
    corroboration_count: int = 0
    days_since_observed: float = 0.0
    specificity: float = 0.5  # [0, 1]
    contradiction_count: int = 0
    total_mentions: int = 1


def compute_evidence_score(factors: EvidenceFactors) -> float:
    """Compute composite evidence score.

    Args:
        factors: The 5 evidence factors.

    Returns:
        Evidence score in [0, 1].
    """
    # 1. Source authority (direct)
    authority = _clamp(factors.source_authority)

    # 2. Corroboration (log-scaled, saturates around 5 confirmations)
    corroboration = min(1.0, math.log1p(factors.corroboration_count) / 1.8)

    # 3. Recency (exponential decay, half-life 60 days)
    recency = math.exp(-factors.days_since_observed / 86.6)  # ln(2)/60 ≈ 0.0116

    # 4. Specificity (direct)
    specificity = _clamp(factors.specificity)

    # 5. Contradiction ratio (inverted - more contradictions = lower score)
    if factors.total_mentions > 0:
        contradiction_ratio = factors.contradiction_count / factors.total_mentions
    else:
        contradiction_ratio = 0.0
    non_contradiction = 1.0 - min(1.0, contradiction_ratio)

    # Weighted combination
    # Authority and corroboration are most important for verification
    score = (
        0.25 * authority
        + 0.30 * corroboration
        + 0.15 * recency
        + 0.10 * specificity
        + 0.20 * non_contradiction
    )

    return _clamp(score)


def _clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamp value to range."""
    return max(min_val, min(max_val, value))


def compute_source_authority(
    source_type: str,
    author_confidence: float = 0.5,
    citation_count: int = 0,
    is_primary: bool = False,
) -> float:
    """Compute source authority score.

    Args:
        source_type: Type of source (documentation, conversation, etc.)
        author_confidence: Confidence in the author/speaker
        citation_count: How often this source is cited
        is_primary: Whether this is a primary source vs. secondary

    Returns:
        Authority score in [0, 1].
    """
    # Base authority by source type
    type_authority = {
        "documentation": 0.9,
        "configuration": 0.85,
        "code_comment": 0.7,
        "assistant_response": 0.6,
        "conversation": 0.5,
        "user_message": 0.4,
        "log": 0.3,
        "unknown": 0.3,
    }

    base = type_authority.get(source_type, 0.5)

    # Adjust for author confidence
    author_factor = 0.8 + 0.2 * _clamp(author_confidence)

    # Adjust for citation count (log-scaled)
    citation_factor = 1.0 + 0.1 * min(1.0, math.log1p(citation_count) / 3.0)

    # Primary sources get a boost
    primary_factor = 1.1 if is_primary else 1.0

    authority = base * author_factor * citation_factor * primary_factor

    return _clamp(authority)


def compute_specificity(text: str) -> float:
    """Compute specificity score for a claim or statement.

    More specific claims (with numbers, names, concrete details)
    score higher than vague statements.

    Args:
        text: The claim or statement text.

    Returns:
        Specificity score in [0, 1].
    """
    score = 0.3  # Base score

    # Check for specific indicators
    import re

    # Numbers and quantities
    if re.search(r'\d+', text):
        score += 0.15

    # Specific names (capitalized words that aren't sentence starts)
    if re.search(r'(?<!\. )[A-Z][a-z]+', text):
        score += 0.1

    # Technical terms (mixed case, underscores, dots)
    if re.search(r'[a-z]+_[a-z]+|[a-z]+\.[a-z]+', text):
        score += 0.1

    # Dates and times
    if re.search(r'\d{1,2}[:/]\d{2}|\d{4}-\d{2}-\d{2}|January|February|March|April|May|June|July|August|September|October|November|December', text, re.I):
        score += 0.15

    # IP addresses, ports, paths
    if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|:\d+|/\w+/', text):
        score += 0.1

    # Penalize vague language
    vague_patterns = [
        r'\bsome\b', r'\bmany\b', r'\bfew\b', r'\boften\b',
        r'\busually\b', r'\bsometimes\b', r'\bmaybe\b', r'\bperhaps\b',
        r'\bmight\b', r'\bcould\b', r'\bprobably\b',
    ]
    for pattern in vague_patterns:
        if re.search(pattern, text, re.I):
            score -= 0.05

    return _clamp(score)


def find_corroborating_evidence(
    claim_text: str,
    candidate_segments: List[dict],
    similarity_threshold: float = 0.7,
) -> List[dict]:
    """Find segments that corroborate a claim.

    Args:
        claim_text: The claim to verify.
        candidate_segments: Potential corroborating segments.
        similarity_threshold: Minimum similarity to count as corroboration.

    Returns:
        List of corroborating segments with scores.
    """
    # This would use embedding similarity in production
    # Placeholder implementation
    corroborating = []

    for segment in candidate_segments:
        # In production: compute semantic similarity
        # Here: simple keyword overlap as placeholder
        claim_words = set(claim_text.lower().split())
        segment_words = set(segment.get("text", "").lower().split())

        if claim_words and segment_words:
            overlap = len(claim_words & segment_words) / len(claim_words | segment_words)
            if overlap >= similarity_threshold:
                corroborating.append({
                    **segment,
                    "corroboration_score": overlap,
                })

    return corroborating


def find_contradicting_evidence(
    claim_text: str,
    candidate_segments: List[dict],
) -> List[dict]:
    """Find segments that contradict a claim.

    Args:
        claim_text: The claim to check.
        candidate_segments: Potential contradicting segments.

    Returns:
        List of contradicting segments.
    """
    # This would use LLM-based contradiction detection in production
    # Placeholder: look for negation patterns
    contradicting = []

    negation_patterns = [
        "not", "never", "no longer", "doesn't", "don't",
        "isn't", "aren't", "wasn't", "weren't", "won't",
        "incorrect", "false", "wrong", "outdated",
    ]

    claim_lower = claim_text.lower()

    for segment in candidate_segments:
        segment_text = segment.get("text", "").lower()

        # Check if segment discusses same topic but with negation
        # This is a simplified heuristic
        for neg in negation_patterns:
            if neg in segment_text:
                # Check for topic overlap
                claim_words = set(claim_lower.split())
                segment_words = set(segment_text.split())
                overlap = len(claim_words & segment_words) / max(len(claim_words), 1)

                if overlap > 0.3:
                    contradicting.append({
                        **segment,
                        "contradiction_type": "negation",
                    })
                    break

    return contradicting


@dataclass
class EvidenceVector:
    """5-dimensional evidence vector for the evidence manifold."""
    source_authority: float
    corroboration: float
    recency: float
    specificity: float
    non_contradiction: float

    def to_list(self) -> List[float]:
        """Convert to list for storage/computation."""
        return [
            self.source_authority,
            self.corroboration,
            self.recency,
            self.specificity,
            self.non_contradiction,
        ]

    @classmethod
    def from_factors(cls, factors: EvidenceFactors) -> "EvidenceVector":
        """Create evidence vector from factors."""
        corroboration = min(1.0, math.log1p(factors.corroboration_count) / 1.8)
        recency = math.exp(-factors.days_since_observed / 86.6)

        if factors.total_mentions > 0:
            contradiction_ratio = factors.contradiction_count / factors.total_mentions
        else:
            contradiction_ratio = 0.0

        return cls(
            source_authority=_clamp(factors.source_authority),
            corroboration=corroboration,
            recency=recency,
            specificity=_clamp(factors.specificity),
            non_contradiction=1.0 - min(1.0, contradiction_ratio),
        )
