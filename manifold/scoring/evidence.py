"""Evidence score computation for verification queries.

The evidence manifold is a 5-dimensional composite score:
1. source_authority - trustworthiness of the source
2. corroboration_count - number of independent confirmations
3. recency - how recent the evidence is
4. specificity - how specific vs. vague the claim is
5. contradiction_ratio - proportion of contradicting evidence

These combine into a single evidence score used for verification
and fact-checking queries.

All weights are configurable via ManifoldConfigV2.scoring.
"""
from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING
import math
import re
from datetime import datetime, timedelta

if TYPE_CHECKING:
    from ..config_v2 import ScoringWeights


@dataclass
class EvidenceFactors:
    """Input factors for evidence scoring."""
    source_authority: float = 0.5  # [0, 1]
    corroboration_count: int = 0
    days_since_observed: float = 0.0
    specificity: float = 0.5  # [0, 1]
    contradiction_count: int = 0
    total_mentions: int = 1


def compute_evidence_score(
    factors: EvidenceFactors,
    weights: Optional["ScoringWeights"] = None,
) -> float:
    """Compute composite evidence score.

    Args:
        factors: The 5 evidence factors.
        weights: Optional scoring weights from config. If None, uses defaults.

    Returns:
        Evidence score in [0, 1].
    """
    # Get weights from config or use defaults
    if weights is None:
        from ..config import get_scoring_weights
        weights = get_scoring_weights()

    # 1. Source authority (direct)
    authority = _clamp(factors.source_authority)

    # 2. Corroboration (log-scaled, saturates around 5 confirmations)
    corroboration = min(1.0, math.log1p(factors.corroboration_count) / 1.8)

    # 3. Recency (exponential decay)
    # halflife_days determines decay rate: ln(2) / halflife
    halflife = weights.recency_halflife_days
    decay_constant = halflife / math.log(2)  # Convert to exponential decay constant
    recency = math.exp(-factors.days_since_observed / decay_constant)

    # 4. Specificity (direct)
    specificity = _clamp(factors.specificity)

    # 5. Contradiction ratio (inverted - more contradictions = lower score)
    if factors.total_mentions > 0:
        contradiction_ratio = factors.contradiction_count / factors.total_mentions
    else:
        contradiction_ratio = 0.0
    non_contradiction = 1.0 - min(1.0, contradiction_ratio)

    # Weighted combination using configurable weights
    base_score = (
        weights.evidence_authority * authority
        + weights.evidence_corroboration * corroboration
        + weights.evidence_recency * recency
        + weights.evidence_specificity * specificity
        + weights.evidence_non_contradiction * non_contradiction
    )

    # Apply contradiction penalty as multiplier (harsh for high contradiction rates)
    contradiction_multiplier = non_contradiction ** weights.evidence_contradiction_exponent
    score = base_score * contradiction_multiplier

    return _clamp(score)


def _clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamp value to range."""
    return max(min_val, min(max_val, value))


def compute_source_authority(
    source_type: str,
    author_confidence: float = 0.5,
    citation_count: int = 0,
    is_primary: bool = False,
    weights: Optional["ScoringWeights"] = None,
) -> float:
    """Compute source authority score.

    Args:
        source_type: Type of source (documentation, conversation, etc.)
        author_confidence: Confidence in the author/speaker
        citation_count: How often this source is cited
        is_primary: Whether this is a primary source vs. secondary
        weights: Optional scoring weights from config.

    Returns:
        Authority score in [0, 1].
    """
    # Get weights from config or use defaults
    if weights is None:
        from ..config import get_scoring_weights
        weights = get_scoring_weights()

    # Get base authority from config weights
    base = weights.get_authority(source_type)

    # Adjust for author confidence (0.8 to 1.0 multiplier)
    author_factor = 0.8 + 0.2 * _clamp(author_confidence)

    # Adjust for citation count (log-scaled, up to 10% boost)
    citation_factor = 1.0 + 0.1 * min(1.0, math.log1p(citation_count) / 3.0)

    # Primary sources get a boost
    primary_factor = 1.1 if is_primary else 1.0

    authority = base * author_factor * citation_factor * primary_factor

    return _clamp(authority)


def compute_specificity(
    text: str,
    base_score: float = 0.3,
    number_boost: float = 0.15,
    name_boost: float = 0.1,
    technical_boost: float = 0.1,
    date_boost: float = 0.15,
    ip_boost: float = 0.1,
    vague_penalty: float = 0.05,
) -> float:
    """Compute specificity score for a claim or statement.

    More specific claims (with numbers, names, concrete details)
    score higher than vague statements.

    Args:
        text: The claim or statement text.
        base_score: Starting score before adjustments.
        number_boost: Boost for presence of numbers.
        name_boost: Boost for presence of proper names.
        technical_boost: Boost for technical terms.
        date_boost: Boost for dates/times.
        ip_boost: Boost for IPs/ports/paths.
        vague_penalty: Penalty per vague word.

    Returns:
        Specificity score in [0, 1].
    """
    score = base_score

    # Numbers and quantities
    if re.search(r'\d+', text):
        score += number_boost

    # Specific names (capitalized words that aren't sentence starts)
    if re.search(r'(?<!\. )[A-Z][a-z]+', text):
        score += name_boost

    # Technical terms (mixed case, underscores, dots)
    if re.search(r'[a-z]+_[a-z]+|[a-z]+\.[a-z]+', text):
        score += technical_boost

    # Dates and times
    if re.search(
        r'\d{1,2}[:/]\d{2}|\d{4}-\d{2}-\d{2}|'
        r'January|February|March|April|May|June|'
        r'July|August|September|October|November|December',
        text,
        re.I
    ):
        score += date_boost

    # IP addresses, ports, paths
    if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|:\d+|/\w+/', text):
        score += ip_boost

    # Penalize vague language
    vague_patterns = [
        r'\bsome\b', r'\bmany\b', r'\bfew\b', r'\boften\b',
        r'\busually\b', r'\bsometimes\b', r'\bmaybe\b', r'\bperhaps\b',
        r'\bmight\b', r'\bcould\b', r'\bprobably\b',
    ]
    for pattern in vague_patterns:
        if re.search(pattern, text, re.I):
            score -= vague_penalty

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
    corroborating = []

    claim_words = set(claim_text.lower().split())
    if not claim_words:
        return corroborating

    for segment in candidate_segments:
        segment_text = segment.get("text", "")
        segment_words = set(segment_text.lower().split())

        if segment_words:
            # Jaccard similarity
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
    topic_overlap_threshold: float = 0.3,
) -> List[dict]:
    """Find segments that contradict a claim.

    Args:
        claim_text: The claim to check.
        candidate_segments: Potential contradicting segments.
        topic_overlap_threshold: Minimum topic overlap to consider.

    Returns:
        List of contradicting segments.
    """
    contradicting = []

    negation_patterns = [
        "not", "never", "no longer", "doesn't", "don't",
        "isn't", "aren't", "wasn't", "weren't", "won't",
        "incorrect", "false", "wrong", "outdated",
    ]

    claim_words = set(claim_text.lower().split())
    if not claim_words:
        return contradicting

    for segment in candidate_segments:
        segment_text = segment.get("text", "").lower()
        segment_words = set(segment_text.split())

        # Check if segment discusses same topic but with negation
        for neg in negation_patterns:
            if neg in segment_text:
                # Check for topic overlap
                overlap = len(claim_words & segment_words) / len(claim_words)

                if overlap > topic_overlap_threshold:
                    contradicting.append({
                        **segment,
                        "contradiction_type": "negation",
                        "negation_word": neg,
                        "topic_overlap": overlap,
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
    def from_factors(
        cls,
        factors: EvidenceFactors,
        weights: Optional["ScoringWeights"] = None,
    ) -> "EvidenceVector":
        """Create evidence vector from factors.

        Args:
            factors: Input evidence factors.
            weights: Optional scoring weights from config.
        """
        if weights is None:
            from ..config import get_scoring_weights
            weights = get_scoring_weights()

        corroboration = min(1.0, math.log1p(factors.corroboration_count) / 1.8)

        halflife = weights.recency_halflife_days
        decay_constant = halflife / math.log(2)
        recency = math.exp(-factors.days_since_observed / decay_constant)

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

    def compute_score(self, weights: Optional["ScoringWeights"] = None) -> float:
        """Compute weighted score from vector components.

        Args:
            weights: Optional scoring weights from config.

        Returns:
            Evidence score in [0, 1].
        """
        if weights is None:
            from ..config import get_scoring_weights
            weights = get_scoring_weights()

        base_score = (
            weights.evidence_authority * self.source_authority
            + weights.evidence_corroboration * self.corroboration
            + weights.evidence_recency * self.recency
            + weights.evidence_specificity * self.specificity
            + weights.evidence_non_contradiction * self.non_contradiction
        )

        # Apply contradiction penalty
        contradiction_multiplier = self.non_contradiction ** weights.evidence_contradiction_exponent
        score = base_score * contradiction_multiplier

        return _clamp(score)
