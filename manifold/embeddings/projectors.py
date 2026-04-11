"""Feature Projectors — project sparse features to dense 768d embedding space.

Used for manifolds where we have structured features instead of text:
- RELATION: 64d graph fingerprint → 768d
- TIME: 12d temporal features → 768d
- EVIDENCE: 5d evidence factors → 768d

The projectors use learned linear transformations. They can be upgraded to
MLPs or more sophisticated architectures if needed.
"""
import logging
import os
from typing import Optional, Tuple
from pathlib import Path

import numpy as np

logger = logging.getLogger("gami.manifold.projectors")

# Embedding dimension
EMBEDDING_DIM = 768

# Feature dimensions for each manifold
RELATION_DIM = 64
TIME_DIM = 12
EVIDENCE_DIM = 5

# Directory for stored projection matrices
PROJECTOR_DIR = Path("/opt/gami/storage/projectors")


class FeatureProjector:
    """Projects sparse features to dense embedding space.

    Uses a learned linear transformation W: R^d → R^768
    where d is the input feature dimension.

    The projection is: output = normalize(W @ input)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = EMBEDDING_DIM,
        name: str = "generic",
    ):
        """Initialize the projector.

        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output embedding (default: 768)
            name: Name of this projector (for loading/saving)
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name

        # Initialize projection matrix
        self.W: Optional[np.ndarray] = None
        self.bias: Optional[np.ndarray] = None

        # Try to load pre-trained weights
        self._load_or_initialize()

    def _load_or_initialize(self):
        """Load pre-trained weights or initialize randomly."""
        weight_path = PROJECTOR_DIR / f"{self.name}_weights.npy"
        bias_path = PROJECTOR_DIR / f"{self.name}_bias.npy"

        if weight_path.exists():
            try:
                self.W = np.load(weight_path)
                if bias_path.exists():
                    self.bias = np.load(bias_path)
                logger.info(f"Loaded projector weights for {self.name}")
                return
            except Exception as e:
                logger.warning(f"Failed to load weights for {self.name}: {e}")

        # Initialize with random orthogonal projection
        # This gives a reasonable starting point before training
        self._initialize_random()

    def _initialize_random(self):
        """Initialize with random orthogonal projection."""
        # Use Xavier initialization
        scale = np.sqrt(2.0 / (self.input_dim + self.output_dim))
        self.W = np.random.randn(self.output_dim, self.input_dim).astype(np.float32) * scale
        self.bias = np.zeros(self.output_dim, dtype=np.float32)
        logger.debug(f"Initialized random projector for {self.name}: {self.input_dim}→{self.output_dim}")

    def project(self, features: np.ndarray) -> np.ndarray:
        """Project features to embedding space.

        Args:
            features: Input features of shape (input_dim,) or (batch, input_dim)

        Returns:
            Projected embedding of shape (output_dim,) or (batch, output_dim)
        """
        if self.W is None:
            self._initialize_random()

        features = np.asarray(features, dtype=np.float32)

        # Handle both single vectors and batches
        is_batch = features.ndim == 2

        if not is_batch:
            features = features.reshape(1, -1)

        # Ensure correct input dimension
        if features.shape[1] != self.input_dim:
            # Pad or truncate
            if features.shape[1] < self.input_dim:
                padded = np.zeros((features.shape[0], self.input_dim), dtype=np.float32)
                padded[:, :features.shape[1]] = features
                features = padded
            else:
                features = features[:, :self.input_dim]

        # Project: output = W @ input + bias
        output = features @ self.W.T
        if self.bias is not None:
            output = output + self.bias

        # L2 normalize
        norms = np.linalg.norm(output, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)  # Avoid division by zero
        output = output / norms

        if not is_batch:
            output = output.squeeze(0)

        return output

    def save(self, directory: Optional[Path] = None):
        """Save projection weights."""
        if directory is None:
            directory = PROJECTOR_DIR

        directory.mkdir(parents=True, exist_ok=True)

        np.save(directory / f"{self.name}_weights.npy", self.W)
        if self.bias is not None:
            np.save(directory / f"{self.name}_bias.npy", self.bias)

        logger.info(f"Saved projector weights for {self.name}")


class RelationProjector(FeatureProjector):
    """Projector for graph fingerprint features (64d → 768d)."""

    def __init__(self):
        super().__init__(input_dim=RELATION_DIM, name="relation")


class TimeProjector(FeatureProjector):
    """Projector for temporal features (12d → 768d)."""

    def __init__(self):
        super().__init__(input_dim=TIME_DIM, name="time")


class EvidenceProjector(FeatureProjector):
    """Projector for evidence factors (5d → 768d)."""

    def __init__(self):
        super().__init__(input_dim=EVIDENCE_DIM, name="evidence")


# ---------------------------------------------------------------------------
# Singleton Instances
# ---------------------------------------------------------------------------

_relation_projector: Optional[RelationProjector] = None
_time_projector: Optional[TimeProjector] = None
_evidence_projector: Optional[EvidenceProjector] = None


def get_relation_projector() -> RelationProjector:
    """Get singleton RelationProjector."""
    global _relation_projector
    if _relation_projector is None:
        _relation_projector = RelationProjector()
    return _relation_projector


def get_time_projector() -> TimeProjector:
    """Get singleton TimeProjector."""
    global _time_projector
    if _time_projector is None:
        _time_projector = TimeProjector()
    return _time_projector


def get_evidence_projector() -> EvidenceProjector:
    """Get singleton EvidenceProjector."""
    global _evidence_projector
    if _evidence_projector is None:
        _evidence_projector = EvidenceProjector()
    return _evidence_projector


# ---------------------------------------------------------------------------
# Graph Fingerprint Computation
# ---------------------------------------------------------------------------

def compute_graph_fingerprint(
    in_edges: dict,
    out_edges: dict,
    connected_types: list,
    centrality: float = 0.0,
) -> np.ndarray:
    """Compute 64d graph fingerprint from graph structure.

    Args:
        in_edges: Dict of edge_type → count for incoming edges
        out_edges: Dict of edge_type → count for outgoing edges
        connected_types: List of entity types this node connects to
        centrality: Graph centrality score (0-1)

    Returns:
        64-dimensional fingerprint vector
    """
    fingerprint = np.zeros(64, dtype=np.float32)

    # Edge types (16 slots for in, 16 for out)
    edge_types = [
        "relates_to", "mentions", "defines", "explains", "references",
        "contradicts", "supports", "causes", "precedes", "follows",
        "contains", "part_of", "example_of", "same_as", "derived_from", "other"
    ]

    # In-edge counts (slots 0-15)
    for i, edge_type in enumerate(edge_types):
        count = in_edges.get(edge_type, 0)
        fingerprint[i] = min(1.0, np.log1p(count) / 3.0)

    # Out-edge counts (slots 16-31)
    for i, edge_type in enumerate(edge_types):
        count = out_edges.get(edge_type, 0)
        fingerprint[16 + i] = min(1.0, np.log1p(count) / 3.0)

    # Entity type connections (slots 32-47)
    entity_types = [
        "person", "organization", "location", "concept", "technology",
        "event", "document", "code", "configuration", "error",
        "feature", "api", "database", "service", "other", "unknown"
    ]
    for i, etype in enumerate(entity_types):
        fingerprint[32 + i] = 1.0 if etype in connected_types else 0.0

    # Summary statistics (slots 48-63)
    total_in = sum(in_edges.values())
    total_out = sum(out_edges.values())
    in_diversity = len(in_edges)
    out_diversity = len(out_edges)

    fingerprint[48] = min(1.0, np.log1p(total_in) / 5.0)
    fingerprint[49] = min(1.0, np.log1p(total_out) / 5.0)
    fingerprint[50] = min(1.0, in_diversity / 10.0)
    fingerprint[51] = min(1.0, out_diversity / 10.0)
    fingerprint[52] = centrality
    fingerprint[53] = 1.0 if total_in > total_out else 0.0  # sink vs source
    fingerprint[54] = len(connected_types) / 16.0
    # Remaining slots (55-63) reserved for future features

    return fingerprint


# ---------------------------------------------------------------------------
# Evidence Factor Computation
# ---------------------------------------------------------------------------

def compute_evidence_factors(
    authority_score: float,
    corroboration_count: int,
    recency_days: float,
    specificity_score: float,
    contradiction_ratio: float,
) -> np.ndarray:
    """Compute 5d evidence factors from evidence scores.

    Args:
        authority_score: Source authority (0-1)
        corroboration_count: Number of corroborating sources
        recency_days: Days since creation
        specificity_score: Specificity of claims (0-1)
        contradiction_ratio: Ratio of contradicting evidence (0-1)

    Returns:
        5-dimensional evidence factor vector
    """
    factors = np.zeros(5, dtype=np.float32)

    factors[0] = authority_score
    factors[1] = min(1.0, np.log1p(corroboration_count) / 2.0)
    factors[2] = np.exp(-recency_days / 60.0)  # Exponential decay, half-life ~60 days
    factors[3] = specificity_score
    factors[4] = 1.0 - contradiction_ratio  # Higher is better (less contradiction)

    return factors
