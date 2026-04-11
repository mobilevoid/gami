"""Multi-Manifold Embedder — generates embeddings for all 6 semantic manifolds.

Produces TRUE multi-dimensional embeddings:
- TOPIC: Direct text embedding (nomic-embed-text)
- CLAIM: Extracted claims → embedded claim text
- PROCEDURE: Extracted steps → embedded procedure
- RELATION: Graph fingerprint → projected to 768d
- TIME: 12 temporal features → projected to 768d
- EVIDENCE: 5 evidence factors → projected to 768d

This module connects to the production embedding service and generates
real embeddings for each manifold type.
"""
import logging
import re
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger("gami.manifold.multi_embedder")


class ManifoldType(str, Enum):
    """The 6 semantic manifold types."""
    TOPIC = "TOPIC"
    CLAIM = "CLAIM"
    PROCEDURE = "PROCEDURE"
    RELATION = "RELATION"
    TIME = "TIME"
    EVIDENCE = "EVIDENCE"


@dataclass
class ManifoldEmbeddings:
    """Container for all manifold embeddings of an object."""
    target_id: str
    target_type: str
    topic: Optional[np.ndarray] = None
    claim: Optional[np.ndarray] = None
    procedure: Optional[np.ndarray] = None
    relation: Optional[np.ndarray] = None
    time: Optional[np.ndarray] = None
    evidence: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[ManifoldType, Optional[np.ndarray]]:
        return {
            ManifoldType.TOPIC: self.topic,
            ManifoldType.CLAIM: self.claim,
            ManifoldType.PROCEDURE: self.procedure,
            ManifoldType.RELATION: self.relation,
            ManifoldType.TIME: self.time,
            ManifoldType.EVIDENCE: self.evidence,
        }

    def populated_manifolds(self) -> List[ManifoldType]:
        """Return list of manifolds that have embeddings."""
        return [m for m, e in self.to_dict().items() if e is not None]


class MultiManifoldEmbedder:
    """Generates embeddings for all 6 semantic manifolds.

    Uses the production embedding service (sentence-transformers/nomic-embed-text)
    for topic/claim/procedure manifolds, and learned projections for
    relation/time/evidence manifolds.
    """

    EMBEDDING_DIM = 768

    def __init__(
        self,
        relation_projector: Optional["FeatureProjector"] = None,
        time_projector: Optional["FeatureProjector"] = None,
        evidence_projector: Optional["FeatureProjector"] = None,
    ):
        """Initialize the multi-manifold embedder.

        Args:
            relation_projector: Projector for graph fingerprints (64d → 768d)
            time_projector: Projector for temporal features (12d → 768d)
            evidence_projector: Projector for evidence factors (5d → 768d)
        """
        self.relation_projector = relation_projector
        self.time_projector = time_projector
        self.evidence_projector = evidence_projector

        # Import projectors lazily
        if self.relation_projector is None:
            from .projectors import get_relation_projector
            self.relation_projector = get_relation_projector()
        if self.time_projector is None:
            from .projectors import get_time_projector
            self.time_projector = get_time_projector()
        if self.evidence_projector is None:
            from .projectors import get_evidence_projector
            self.evidence_projector = get_evidence_projector()

    def embed_all_manifolds(
        self,
        text: str,
        target_id: str,
        target_type: str = "segment",
        graph_fingerprint: Optional[np.ndarray] = None,
        temporal_features: Optional[np.ndarray] = None,
        evidence_factors: Optional[np.ndarray] = None,
    ) -> ManifoldEmbeddings:
        """Generate embeddings for all 6 manifolds.

        Args:
            text: The text content to embed
            target_id: ID of the target object
            target_type: Type of target (segment, entity, claim, memory)
            graph_fingerprint: 64d graph fingerprint (optional)
            temporal_features: 12d temporal features (optional)
            evidence_factors: 5d evidence factors (optional)

        Returns:
            ManifoldEmbeddings containing all populated embeddings
        """
        result = ManifoldEmbeddings(target_id=target_id, target_type=target_type)

        # TOPIC: Direct text embedding
        result.topic = self._embed_text(text)

        # CLAIM: Extract claims and embed
        claims = self._extract_claims(text)
        if claims:
            claim_text = " | ".join(claims)
            result.claim = self._embed_text(claim_text)

        # PROCEDURE: Extract steps and embed
        steps = self._extract_procedure_steps(text)
        if steps:
            procedure_text = " → ".join(steps)
            result.procedure = self._embed_text(procedure_text)

        # RELATION: Project graph fingerprint
        if graph_fingerprint is not None and self.relation_projector:
            result.relation = self.relation_projector.project(graph_fingerprint)

        # TIME: Project temporal features
        if temporal_features is not None and self.time_projector:
            result.time = self.time_projector.project(temporal_features)
        else:
            # Extract temporal features from text and project
            features = self._extract_temporal_features(text)
            if features is not None:
                result.time = self.time_projector.project(features)

        # EVIDENCE: Project evidence factors
        if evidence_factors is not None and self.evidence_projector:
            result.evidence = self.evidence_projector.project(evidence_factors)

        return result

    def embed_topic(self, text: str) -> Optional[np.ndarray]:
        """Generate topic manifold embedding (direct text embedding)."""
        return self._embed_text(text)

    def embed_claim(self, text: str) -> Optional[np.ndarray]:
        """Generate claim manifold embedding (extracted claims)."""
        claims = self._extract_claims(text)
        if not claims:
            return None
        claim_text = " | ".join(claims)
        return self._embed_text(claim_text)

    def embed_procedure(self, text: str) -> Optional[np.ndarray]:
        """Generate procedure manifold embedding (extracted steps)."""
        steps = self._extract_procedure_steps(text)
        if not steps:
            return None
        procedure_text = " → ".join(steps)
        return self._embed_text(procedure_text)

    def _embed_text(self, text: str) -> Optional[np.ndarray]:
        """Embed text using production embedding service."""
        if not text or len(text.strip()) < 3:
            return None

        try:
            from api.llm.embeddings import embed_text_sync
            embedding = embed_text_sync(text)
            return np.array(embedding, dtype=np.float32)
        except Exception as e:
            logger.warning(f"Embedding failed: {e}")
            return None

    def _extract_claims(self, text: str) -> List[str]:
        """Extract claim-like statements from text.

        Uses pattern matching for common claim structures.
        Could be upgraded to use LLM extraction.
        """
        claims = []

        # Pattern-based claim extraction
        # Look for "X is Y", "X has Y", "X does Y" patterns
        patterns = [
            r"(?:^|\. )([A-Z][^.!?]+(?:is|are|was|were|has|have|does|do)\s+[^.!?]+)[.!?]",
            r"(?:^|\. )([A-Z][^.!?]+(?:can|will|should|must)\s+[^.!?]+)[.!?]",
            r"(?:^|\. )(The\s+[^.!?]+(?:is|are|was|were)\s+[^.!?]+)[.!?]",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            claims.extend([m.strip() for m in matches if len(m) > 20])

        # Deduplicate and limit
        seen = set()
        unique_claims = []
        for claim in claims:
            normalized = claim.lower()
            if normalized not in seen:
                seen.add(normalized)
                unique_claims.append(claim)

        return unique_claims[:10]  # Limit to top 10 claims

    def _extract_procedure_steps(self, text: str) -> List[str]:
        """Extract procedure/how-to steps from text.

        Looks for numbered lists, bullet points, and imperative sentences.
        """
        steps = []

        # Numbered steps
        numbered = re.findall(r"(?:^|\n)\s*(\d+)[.):]\s*([^\n]+)", text)
        if numbered:
            steps.extend([step.strip() for _, step in numbered])

        # Bullet points
        bullets = re.findall(r"(?:^|\n)\s*[-•*]\s*([^\n]+)", text)
        if bullets:
            steps.extend([step.strip() for step in bullets])

        # Imperative sentences (command verbs)
        imperatives = re.findall(
            r"(?:^|\. )((?:Run|Execute|Install|Create|Add|Remove|Set|Configure|Enable|Disable|Click|Open|Close|Enter|Type|Press|Go to)\s+[^.!?]+)[.!?]",
            text, re.IGNORECASE
        )
        if imperatives:
            steps.extend([step.strip() for step in imperatives])

        # Deduplicate
        seen = set()
        unique_steps = []
        for step in steps:
            normalized = step.lower()
            if normalized not in seen and len(step) > 5:
                seen.add(normalized)
                unique_steps.append(step)

        return unique_steps[:20]  # Limit to 20 steps

    def _extract_temporal_features(self, text: str) -> Optional[np.ndarray]:
        """Extract 12-dimensional temporal features from text.

        Features:
        0: has_explicit_timestamp (0/1)
        1: recency_score (0-1, based on temporal keywords)
        2: future_tense_ratio
        3: past_tense_ratio
        4: present_tense_ratio
        5: temporal_keyword_count
        6: has_duration (0/1)
        7: has_frequency (0/1)
        8: sequence_indicator_count
        9: deadline_indicator (0/1)
        10: historical_reference (0/1)
        11: normalized_word_count
        """
        features = np.zeros(12, dtype=np.float32)

        if not text:
            return features

        text_lower = text.lower()
        words = text_lower.split()
        word_count = len(words)

        # Feature 0: Explicit timestamp
        timestamp_patterns = [
            r"\d{4}-\d{2}-\d{2}",  # ISO date
            r"\d{1,2}/\d{1,2}/\d{2,4}",  # US date
            r"\d{1,2}:\d{2}(?::\d{2})?(?:\s*[ap]m)?",  # Time
        ]
        has_timestamp = any(re.search(p, text_lower) for p in timestamp_patterns)
        features[0] = 1.0 if has_timestamp else 0.0

        # Feature 1: Recency score
        recency_keywords = ["now", "today", "currently", "recently", "just", "latest"]
        features[1] = min(1.0, sum(1 for w in words if w in recency_keywords) / 3)

        # Features 2-4: Tense ratios
        past_indicators = ["was", "were", "had", "did", "went", "said", "been"]
        future_indicators = ["will", "shall", "going to", "about to", "plan to"]
        present_indicators = ["is", "are", "has", "does", "do"]

        past_count = sum(1 for w in words if w in past_indicators)
        future_count = sum(1 for w in words if w in future_indicators)
        present_count = sum(1 for w in words if w in present_indicators)
        total_tense = past_count + future_count + present_count + 1

        features[2] = future_count / total_tense
        features[3] = past_count / total_tense
        features[4] = present_count / total_tense

        # Feature 5: Temporal keyword count
        temporal_keywords = [
            "yesterday", "tomorrow", "ago", "later", "before", "after",
            "during", "when", "while", "until", "since", "always", "never",
            "sometimes", "often", "rarely", "weekly", "monthly", "yearly"
        ]
        features[5] = min(1.0, sum(1 for w in words if w in temporal_keywords) / 5)

        # Feature 6: Duration
        duration_patterns = [r"\d+\s*(second|minute|hour|day|week|month|year)s?"]
        features[6] = 1.0 if any(re.search(p, text_lower) for p in duration_patterns) else 0.0

        # Feature 7: Frequency
        frequency_keywords = ["daily", "weekly", "monthly", "yearly", "hourly", "every"]
        features[7] = 1.0 if any(w in text_lower for w in frequency_keywords) else 0.0

        # Feature 8: Sequence indicators
        sequence_keywords = ["first", "then", "next", "finally", "lastly", "after that"]
        features[8] = min(1.0, sum(1 for w in words if w in sequence_keywords) / 3)

        # Feature 9: Deadline indicator
        deadline_keywords = ["deadline", "due", "by", "before", "until"]
        features[9] = 1.0 if any(w in text_lower for w in deadline_keywords) else 0.0

        # Feature 10: Historical reference
        historical_keywords = ["history", "historical", "originally", "formerly", "in the past"]
        features[10] = 1.0 if any(w in text_lower for w in historical_keywords) else 0.0

        # Feature 11: Normalized word count
        features[11] = min(1.0, word_count / 500)

        return features


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------

_embedder_instance = None


def get_multi_embedder() -> MultiManifoldEmbedder:
    """Get singleton instance of MultiManifoldEmbedder."""
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = MultiManifoldEmbedder()
    return _embedder_instance


def embed_segment_all_manifolds(
    text: str,
    segment_id: str,
    graph_fingerprint: Optional[np.ndarray] = None,
    temporal_features: Optional[np.ndarray] = None,
    evidence_factors: Optional[np.ndarray] = None,
) -> ManifoldEmbeddings:
    """Convenience function to embed a segment into all manifolds."""
    embedder = get_multi_embedder()
    return embedder.embed_all_manifolds(
        text=text,
        target_id=segment_id,
        target_type="segment",
        graph_fingerprint=graph_fingerprint,
        temporal_features=temporal_features,
        evidence_factors=evidence_factors,
    )
