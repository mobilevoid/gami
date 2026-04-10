"""Manifold embedder — generates embeddings for each semantic manifold.

The embedder takes objects (segments, claims, procedures, etc.) and generates
embeddings in the appropriate manifold spaces:

- Topic manifold: Direct text embedding
- Claim manifold: Embedding of canonical SPO form
- Procedure manifold: Embedding of canonical procedure form
- Relation manifold: Derived from graph neighborhood (not dense embedding)
- Time manifold: Structured features, not dense embedding
- Evidence manifold: Composite score, not embedding

This module is isolated and does not connect to the production embedding
service. It provides the interface that will be connected during activation.
"""
import hashlib
import logging
from typing import Optional, List, Dict, Any, Tuple

from ..models.schemas import (
    ManifoldType,
    TargetType,
    ManifoldScore,
    CanonicalClaimSchema,
    CanonicalProcedureSchema,
)
from ..canonical.forms import (
    CanonicalClaimForm,
    CanonicalProcedureForm,
    CanonicalEntityForm,
)

logger = logging.getLogger("gami.manifold.embeddings.embedder")


# Embedding dimensions
EMBEDDING_DIM = 768  # nomic-embed-text dimension
TIME_FEATURE_DIM = 12
RELATION_FEATURE_DIM = 64


class ManifoldEmbedder:
    """Generates embeddings for different semantic manifolds.

    Uses a shared embedding model (nomic-embed-text) with manifold-specific
    canonical forms as input. The canonical form determines what aspects
    of the object are emphasized in each manifold.
    """

    def __init__(
        self,
        embedding_model: str = "nomic-embed-text",
        ollama_url: Optional[str] = None,
    ):
        """Initialize the manifold embedder.

        Args:
            embedding_model: Name of the embedding model.
            ollama_url: URL of the Ollama server (default: from settings).
        """
        self.embedding_model = embedding_model
        self.ollama_url = ollama_url
        self._embedding_cache: Dict[str, List[float]] = {}

    def embed_for_manifold(
        self,
        text: str,
        manifold_type: ManifoldType,
        canonical_form: Optional[str] = None,
    ) -> Optional[List[float]]:
        """Generate embedding for a specific manifold.

        Args:
            text: Original text to embed.
            manifold_type: Which manifold to embed for.
            canonical_form: Optional pre-computed canonical form.

        Returns:
            Embedding vector (768d) or None if embedding failed.
        """
        # Determine what text to embed based on manifold
        embed_text = canonical_form or text

        if manifold_type == ManifoldType.TOPIC:
            # Topic manifold: embed the raw text directly
            embed_text = text
        elif manifold_type == ManifoldType.CLAIM:
            # Claim manifold: needs canonical SPO form
            if not canonical_form:
                logger.debug("Claim manifold requires canonical form")
                return None
            embed_text = canonical_form
        elif manifold_type == ManifoldType.PROCEDURE:
            # Procedure manifold: needs canonical procedure form
            if not canonical_form:
                logger.debug("Procedure manifold requires canonical form")
                return None
            embed_text = canonical_form
        elif manifold_type in (ManifoldType.RELATION, ManifoldType.TIME, ManifoldType.EVIDENCE):
            # These are not dense embeddings
            logger.debug(f"{manifold_type} is not a dense embedding manifold")
            return None

        # Generate embedding
        return self._embed(embed_text)

    def embed_segment(
        self,
        segment_text: str,
        manifolds: List[ManifoldType] = None,
    ) -> Dict[ManifoldType, Optional[List[float]]]:
        """Embed a segment into specified manifolds.

        Args:
            segment_text: The segment text.
            manifolds: Which manifolds to embed into (default: topic only).

        Returns:
            Dict mapping manifold type to embedding (or None).
        """
        if manifolds is None:
            manifolds = [ManifoldType.TOPIC]

        results = {}
        for manifold in manifolds:
            if manifold == ManifoldType.TOPIC:
                results[manifold] = self._embed(segment_text)
            else:
                # Other manifolds require canonical forms
                results[manifold] = None

        return results

    def embed_claim(
        self,
        claim: CanonicalClaimSchema,
    ) -> Dict[ManifoldType, Optional[List[float]]]:
        """Embed a canonical claim into relevant manifolds.

        Args:
            claim: The canonical claim.

        Returns:
            Dict with topic and claim manifold embeddings.
        """
        canonical_text = claim.to_canonical_text()

        return {
            ManifoldType.TOPIC: self._embed(canonical_text),
            ManifoldType.CLAIM: self._embed(canonical_text),
        }

    def embed_procedure(
        self,
        procedure: CanonicalProcedureSchema,
    ) -> Dict[ManifoldType, Optional[List[float]]]:
        """Embed a canonical procedure into relevant manifolds.

        Args:
            procedure: The canonical procedure.

        Returns:
            Dict with topic and procedure manifold embeddings.
        """
        canonical_text = procedure.to_canonical_text()

        return {
            ManifoldType.TOPIC: self._embed(canonical_text),
            ManifoldType.PROCEDURE: self._embed(canonical_text),
        }

    def embed_entity(
        self,
        entity: CanonicalEntityForm,
    ) -> Dict[ManifoldType, Optional[List[float]]]:
        """Embed an entity into relevant manifolds.

        Args:
            entity: The canonical entity form.

        Returns:
            Dict with topic manifold embedding.
            (Relation manifold is derived, not embedded)
        """
        canonical_text = entity.to_text()

        return {
            ManifoldType.TOPIC: self._embed(canonical_text),
            # Relation manifold computed separately from graph
        }

    def _embed(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text.

        NOTE: This is a stub in the isolated module.
        Returns a deterministic pseudo-embedding based on text hash.
        Actual implementation will connect to Ollama during activation.
        """
        if not text or len(text.strip()) < 3:
            return None

        # Check cache
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        # STUB: Generate deterministic pseudo-embedding
        # This is NOT a real embedding - just for testing structure
        # Real implementation will call Ollama
        logger.debug("Using stub embedding (not connected to Ollama)")

        # Generate reproducible pseudo-embedding from hash
        import struct
        h = hashlib.sha256(text.encode()).digest()
        # Convert hash bytes to floats
        embedding = []
        for i in range(EMBEDDING_DIM):
            # Use rolling bytes from hash
            byte_idx = i % 32
            val = h[byte_idx] / 255.0  # Normalize to 0-1
            val = (val - 0.5) * 2  # Scale to -1 to 1
            embedding.append(val)

        # Normalize to unit length
        norm = sum(x * x for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x / norm for x in embedding]

        self._embedding_cache[cache_key] = embedding
        return embedding

    def compute_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float],
    ) -> float:
        """Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector.
            embedding2: Second embedding vector.

        Returns:
            Cosine similarity (-1 to 1).
        """
        if len(embedding1) != len(embedding2):
            return 0.0

        dot = sum(a * b for a, b in zip(embedding1, embedding2))
        norm1 = sum(x * x for x in embedding1) ** 0.5
        norm2 = sum(x * x for x in embedding2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot / (norm1 * norm2)


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------

def embed_for_manifold(
    text: str,
    manifold_type: ManifoldType,
    canonical_form: Optional[str] = None,
) -> Optional[List[float]]:
    """Convenience function for manifold embedding.

    Args:
        text: Text to embed.
        manifold_type: Target manifold.
        canonical_form: Optional canonical form.

    Returns:
        Embedding vector or None.
    """
    embedder = ManifoldEmbedder()
    return embedder.embed_for_manifold(text, manifold_type, canonical_form)


def embed_batch(
    texts: List[str],
    manifold_type: ManifoldType = ManifoldType.TOPIC,
) -> List[Optional[List[float]]]:
    """Embed multiple texts into a manifold.

    Args:
        texts: List of texts.
        manifold_type: Target manifold.

    Returns:
        List of embeddings (or None for each).
    """
    embedder = ManifoldEmbedder()
    return [embedder.embed_for_manifold(text, manifold_type) for text in texts]


def compute_manifold_scores(
    query_embeddings: Dict[ManifoldType, List[float]],
    candidate_embeddings: Dict[ManifoldType, List[float]],
) -> ManifoldScore:
    """Compute similarity scores across all manifolds.

    Args:
        query_embeddings: Query embeddings per manifold.
        candidate_embeddings: Candidate embeddings per manifold.

    Returns:
        ManifoldScore with similarity for each manifold.
    """
    embedder = ManifoldEmbedder()

    scores = ManifoldScore()

    for manifold in [ManifoldType.TOPIC, ManifoldType.CLAIM, ManifoldType.PROCEDURE]:
        q_emb = query_embeddings.get(manifold)
        c_emb = candidate_embeddings.get(manifold)

        if q_emb and c_emb:
            sim = embedder.compute_similarity(q_emb, c_emb)
            setattr(scores, manifold.value, sim)

    return scores
