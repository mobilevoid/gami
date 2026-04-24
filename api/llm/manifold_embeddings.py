"""True manifold embeddings using hyperbolic (Poincaré ball) and product spaces.

This module provides REAL manifold embeddings, not flat Euclidean vectors:
- Hyperbolic space for hierarchical structure (entities, clusters, taxonomies)
- Spherical space for categorical/type information
- Euclidean space for general semantics

Distance in these spaces is geodesic distance, not cosine similarity.
"""
import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Optional, Tuple, List, Dict, Union
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Note: We implement our own Poincaré ball operations below rather than
# depending on geoopt. This keeps dependencies minimal while providing
# the core hyperbolic geometry operations we need.


@dataclass
class ManifoldCoordinates:
    """Coordinates in product manifold space H^n × S^m × E^k."""
    hyperbolic: np.ndarray    # Poincaré ball coordinates (hierarchy)
    spherical: np.ndarray     # Sphere coordinates (type/category)
    euclidean: np.ndarray     # Euclidean coordinates (semantics)

    def to_dict(self) -> Dict[str, List[float]]:
        return {
            "hyperbolic": self.hyperbolic.tolist(),
            "spherical": self.spherical.tolist(),
            "euclidean": self.euclidean.tolist(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, List[float]]) -> "ManifoldCoordinates":
        return cls(
            hyperbolic=np.array(d["hyperbolic"]),
            spherical=np.array(d["spherical"]),
            euclidean=np.array(d["euclidean"]),
        )

    def to_flat(self) -> List[float]:
        """Concatenate all coordinates for storage."""
        return self.hyperbolic.tolist() + self.spherical.tolist() + self.euclidean.tolist()

    @classmethod
    def from_flat(cls, flat: List[float], dims: Tuple[int, int, int] = (32, 16, 64)) -> "ManifoldCoordinates":
        """Reconstruct from flattened coordinates."""
        h_dim, s_dim, e_dim = dims
        return cls(
            hyperbolic=np.array(flat[:h_dim]),
            spherical=np.array(flat[h_dim:h_dim+s_dim]),
            euclidean=np.array(flat[h_dim+s_dim:h_dim+s_dim+e_dim]),
        )


class PoincareOperations:
    """Operations in the Poincaré ball model of hyperbolic space.

    The Poincaré ball is the open unit ball {x : ||x|| < 1} with the metric:
    ds² = 4 * ||dx||² / (1 - ||x||²)²

    Key properties:
    - Center represents root/general concepts
    - Boundary (||x|| → 1) represents leaves/specific instances
    - Distance grows exponentially toward boundary (perfect for trees)
    """

    def __init__(self, curvature: float = 1.0, eps: float = 1e-5):
        self.c = curvature  # Negative curvature parameter
        self.eps = eps

    def mobius_add(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Möbius addition in the Poincaré ball."""
        x_norm_sq = np.sum(x ** 2)
        y_norm_sq = np.sum(y ** 2)
        xy_dot = np.sum(x * y)

        num = (1 + 2 * self.c * xy_dot + self.c * y_norm_sq) * x + (1 - self.c * x_norm_sq) * y
        denom = 1 + 2 * self.c * xy_dot + self.c ** 2 * x_norm_sq * y_norm_sq

        return num / (denom + self.eps)

    def distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """Geodesic distance in the Poincaré ball.

        d(x,y) = (2/√c) * arctanh(√c * ||(-x) ⊕ y||)

        This is the TRUE hyperbolic distance, not cosine similarity.
        """
        diff = self.mobius_add(-x, y)
        diff_norm = np.linalg.norm(diff)
        diff_norm = np.clip(diff_norm, 0, 1 - self.eps)

        return (2.0 / np.sqrt(self.c)) * np.arctanh(np.sqrt(self.c) * diff_norm)

    def expmap0(self, v: np.ndarray) -> np.ndarray:
        """Exponential map from origin (project Euclidean tangent to Poincaré ball)."""
        v_norm = np.linalg.norm(v)
        if v_norm < self.eps:
            return v

        return np.tanh(np.sqrt(self.c) * v_norm / 2) * v / (np.sqrt(self.c) * v_norm)

    def logmap0(self, y: np.ndarray) -> np.ndarray:
        """Logarithmic map to origin (project from Poincaré ball to Euclidean tangent)."""
        y_norm = np.linalg.norm(y)
        if y_norm < self.eps:
            return y

        return (2.0 / np.sqrt(self.c)) * np.arctanh(np.sqrt(self.c) * y_norm) * y / y_norm

    def project(self, x: np.ndarray) -> np.ndarray:
        """Project point to inside the Poincaré ball (||x|| < 1)."""
        norm = np.linalg.norm(x)
        if norm >= 1 - self.eps:
            return x / norm * (1 - self.eps)
        return x


class SphereOperations:
    """Operations on the unit sphere S^n.

    Good for categorical/type embeddings where we want:
    - Discrete clusters
    - Antipodal points for opposites
    - Great circle distances
    """

    def __init__(self, eps: float = 1e-5):
        self.eps = eps

    def distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """Geodesic (great circle) distance on sphere."""
        # Normalize to ensure on unit sphere
        x = x / (np.linalg.norm(x) + self.eps)
        y = y / (np.linalg.norm(y) + self.eps)

        dot = np.clip(np.dot(x, y), -1.0, 1.0)
        return np.arccos(dot)

    def project(self, x: np.ndarray) -> np.ndarray:
        """Project to unit sphere."""
        norm = np.linalg.norm(x)
        if norm < self.eps:
            # Random point on sphere if zero vector
            x = np.random.randn(len(x))
            norm = np.linalg.norm(x)
        return x / norm


class ProductManifold:
    """Product manifold H^n × S^m × E^k.

    Combines:
    - Hyperbolic space (Poincaré ball) for hierarchical structure
    - Spherical space for type/category
    - Euclidean space for general semantics

    Distance is weighted combination of component distances.
    """

    def __init__(
        self,
        hyperbolic_dim: int = 32,
        spherical_dim: int = 16,
        euclidean_dim: int = 64,
        curvature: float = 1.0,
    ):
        self.h_dim = hyperbolic_dim
        self.s_dim = spherical_dim
        self.e_dim = euclidean_dim
        self.total_dim = hyperbolic_dim + spherical_dim + euclidean_dim

        self.poincare = PoincareOperations(curvature=curvature)
        self.sphere = SphereOperations()

    def distance(
        self,
        x: ManifoldCoordinates,
        y: ManifoldCoordinates,
        weights: Tuple[float, float, float] = (0.4, 0.2, 0.4),
    ) -> float:
        """Weighted geodesic distance in product manifold.

        Args:
            x, y: Points in product manifold
            weights: (hyperbolic_weight, spherical_weight, euclidean_weight)

        Returns:
            Combined distance respecting each manifold's geometry
        """
        w_h, w_s, w_e = weights

        # Hyperbolic distance (hierarchy)
        d_h = self.poincare.distance(x.hyperbolic, y.hyperbolic)

        # Spherical distance (type)
        d_s = self.sphere.distance(x.spherical, y.spherical)

        # Euclidean distance (semantics)
        d_e = float(np.linalg.norm(x.euclidean - y.euclidean))

        return float(w_h * d_h + w_s * d_s + w_e * d_e)

    def project(self, coords: ManifoldCoordinates) -> ManifoldCoordinates:
        """Project coordinates to valid manifold points."""
        return ManifoldCoordinates(
            hyperbolic=self.poincare.project(coords.hyperbolic),
            spherical=self.sphere.project(coords.spherical),
            euclidean=coords.euclidean,  # Euclidean needs no projection
        )


class HyperbolicProjectionHead(nn.Module):
    """Neural network to project Euclidean embeddings to Poincaré ball."""

    def __init__(self, input_dim: int = 768, output_dim: int = 32, curvature: float = 1.0):
        super().__init__()
        self.c = curvature
        self.eps = 1e-5

        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project to Poincaré ball via exponential map."""
        # Get tangent vector at origin
        v = self.layers(x)

        # Exponential map from origin to Poincaré ball
        v_norm = torch.norm(v, dim=-1, keepdim=True).clamp(min=self.eps)

        # tanh(√c * ||v|| / 2) * v / (√c * ||v||)
        scale = torch.tanh(np.sqrt(self.c) * v_norm / 2) / (np.sqrt(self.c) * v_norm)

        return v * scale


class SphericalProjectionHead(nn.Module):
    """Neural network to project Euclidean embeddings to unit sphere."""

    def __init__(self, input_dim: int = 768, output_dim: int = 16):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project to unit sphere."""
        v = self.layers(x)
        return nn.functional.normalize(v, p=2, dim=-1)


class EuclideanProjectionHead(nn.Module):
    """Neural network to project to lower-dimensional Euclidean space."""

    def __init__(self, input_dim: int = 768, output_dim: int = 64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ManifoldEncoder(nn.Module):
    """Full encoder from text to product manifold coordinates.

    Architecture:
    text → SentenceTransformer (768-dim) → [HyperbolicHead, SphericalHead, EuclideanHead]
                                                    ↓              ↓            ↓
                                              H^32 (Poincaré)   S^16        E^64

    Total: 112 dimensions in product manifold H^32 × S^16 × E^64
    """

    def __init__(
        self,
        base_model_name: str = "nomic-ai/nomic-embed-text-v1.5",
        hyperbolic_dim: int = 32,
        spherical_dim: int = 16,
        euclidean_dim: int = 64,
        curvature: float = 1.0,
        device: Optional[str] = None,
    ):
        super().__init__()

        self.h_dim = hyperbolic_dim
        self.s_dim = spherical_dim
        self.e_dim = euclidean_dim
        self.dims = (hyperbolic_dim, spherical_dim, euclidean_dim)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Base encoder (frozen)
        from sentence_transformers import SentenceTransformer
        self.base_encoder = SentenceTransformer(base_model_name, trust_remote_code=True)
        self.base_encoder.to(device)
        base_dim = self.base_encoder.get_sentence_embedding_dimension()
        if base_dim is None:
            base_dim = 768  # Default fallback
        self.base_dim = base_dim
        logger.info(f"Loaded base encoder: {base_model_name} ({self.base_dim} dims)")

        # Projection heads to each manifold component
        self.hyperbolic_head = HyperbolicProjectionHead(self.base_dim, hyperbolic_dim, curvature).to(device)
        self.spherical_head = SphericalProjectionHead(self.base_dim, spherical_dim).to(device)
        self.euclidean_head = EuclideanProjectionHead(self.base_dim, euclidean_dim).to(device)

        # Product manifold for distance computation
        self.manifold = ProductManifold(hyperbolic_dim, spherical_dim, euclidean_dim, curvature)

        # Load pretrained projection weights if available
        self._load_projection_weights()

    def _load_projection_weights(self):
        """Load pretrained projection head weights if available."""
        weights_path = Path(__file__).parent / "manifold_weights.pt"
        if weights_path.exists():
            state = torch.load(weights_path, map_location=self.device)
            self.hyperbolic_head.load_state_dict(state["hyperbolic_head"])
            self.spherical_head.load_state_dict(state["spherical_head"])
            self.euclidean_head.load_state_dict(state["euclidean_head"])
            logger.info("Loaded pretrained manifold projection weights")
        else:
            logger.info("No pretrained manifold weights found, using random initialization")

    def save_projection_weights(self, path: Optional[Union[str, Path]] = None):
        """Save projection head weights."""
        if path is None:
            path = Path(__file__).parent / "manifold_weights.pt"
        path = Path(path)
        torch.save({
            "hyperbolic_head": self.hyperbolic_head.state_dict(),
            "spherical_head": self.spherical_head.state_dict(),
            "euclidean_head": self.euclidean_head.state_dict(),
        }, path)
        logger.info(f"Saved manifold projection weights to {path}")

    @torch.no_grad()
    def encode(self, text: str) -> ManifoldCoordinates:
        """Encode text to product manifold coordinates."""
        # Get base embedding
        base_emb = self.base_encoder.encode(text, convert_to_tensor=True)
        if base_emb.dim() == 1:
            base_emb = base_emb.unsqueeze(0)
        base_emb = base_emb.to(self.device)

        # Project to each manifold component
        h = self.hyperbolic_head(base_emb).squeeze(0).cpu().numpy()
        s = self.spherical_head(base_emb).squeeze(0).cpu().numpy()
        e = self.euclidean_head(base_emb).squeeze(0).cpu().numpy()

        return ManifoldCoordinates(hyperbolic=h, spherical=s, euclidean=e)

    @torch.no_grad()
    def encode_batch(self, texts: List[str]) -> List[ManifoldCoordinates]:
        """Encode batch of texts to product manifold coordinates."""
        # Get base embeddings
        base_embs = self.base_encoder.encode(texts, convert_to_tensor=True)
        base_embs = base_embs.to(self.device)

        # Project to each manifold component
        h = self.hyperbolic_head(base_embs).cpu().numpy()
        s = self.spherical_head(base_embs).cpu().numpy()
        e = self.euclidean_head(base_embs).cpu().numpy()

        return [
            ManifoldCoordinates(hyperbolic=h[i], spherical=s[i], euclidean=e[i])
            for i in range(len(texts))
        ]

    @torch.no_grad()
    def encode_batch_from_embeddings(self, embeddings: torch.Tensor) -> List[ManifoldCoordinates]:
        """Project pre-computed base embeddings (768d) to manifold coordinates.

        This is used by the dream cycle when we already have embeddings stored
        in the database and just need to project them to manifold space.

        Args:
            embeddings: Tensor of shape (batch_size, 768) with base embeddings

        Returns:
            List of ManifoldCoordinates for each embedding
        """
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)
        embeddings = embeddings.to(self.device)

        # Project to each manifold component
        h = self.hyperbolic_head(embeddings).cpu().numpy()
        s = self.spherical_head(embeddings).cpu().numpy()
        e = self.euclidean_head(embeddings).cpu().numpy()

        return [
            ManifoldCoordinates(hyperbolic=h[i], spherical=s[i], euclidean=e[i])
            for i in range(embeddings.shape[0])
        ]

    def distance(
        self,
        x: ManifoldCoordinates,
        y: ManifoldCoordinates,
        weights: Tuple[float, float, float] = (0.4, 0.2, 0.4),
    ) -> float:
        """Compute geodesic distance in product manifold."""
        return self.manifold.distance(x, y, weights)


# Global encoder instance (lazy loaded)
_encoder: Optional[ManifoldEncoder] = None


def get_manifold_encoder() -> ManifoldEncoder:
    """Get or create the global manifold encoder."""
    global _encoder
    if _encoder is None:
        _encoder = ManifoldEncoder()
    return _encoder


def embed_to_manifold(text: str) -> ManifoldCoordinates:
    """Embed text to product manifold coordinates."""
    return get_manifold_encoder().encode(text)


def embed_batch_to_manifold(texts: List[str]) -> List[ManifoldCoordinates]:
    """Embed batch of texts to product manifold coordinates."""
    return get_manifold_encoder().encode_batch(texts)


def manifold_distance(
    x: ManifoldCoordinates,
    y: ManifoldCoordinates,
    weights: Tuple[float, float, float] = (0.4, 0.2, 0.4),
) -> float:
    """Compute geodesic distance between points in product manifold."""
    return get_manifold_encoder().distance(x, y, weights)


# For backward compatibility: also provide flat embedding function
def embed_to_flat_manifold(text: str) -> List[float]:
    """Embed text and return flattened manifold coordinates (112 dims)."""
    coords = embed_to_manifold(text)
    return coords.to_flat()


def embed_batch_to_flat_manifold(texts: List[str]) -> List[List[float]]:
    """Embed batch and return flattened manifold coordinates."""
    coords_list = embed_batch_to_manifold(texts)
    return [c.to_flat() for c in coords_list]


def project_embeddings_to_manifold(embeddings: torch.Tensor) -> List[ManifoldCoordinates]:
    """Project pre-computed 768d embeddings to manifold coordinates.

    Use this when you already have base embeddings (e.g., from database)
    and just need to compute manifold coordinates.
    """
    return get_manifold_encoder().encode_batch_from_embeddings(embeddings)
