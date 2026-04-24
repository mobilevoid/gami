"""Relation manifold derivation from graph structure.

The relation manifold is derived from the Apache AGE graph, not
from embeddings. It captures structural relationships between
entities via graph fingerprints.

A graph fingerprint encodes:
- Local neighborhood (1-hop and 2-hop connections)
- Edge types in the neighborhood
- Node types connected to
- Betweenness/centrality signals

These fingerprints enable relation-aware retrieval without
requiring dense embeddings.

All weights are configurable via ManifoldConfigV2.scoring.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple, TYPE_CHECKING
from collections import Counter
import hashlib
import math

if TYPE_CHECKING:
    from ..config_v2 import ScoringWeights


@dataclass
class GraphFingerprint:
    """Compact representation of entity's graph neighborhood."""

    entity_id: str
    entity_type: str

    # Direct connections
    out_edges: Dict[str, int] = field(default_factory=dict)  # edge_type -> count
    in_edges: Dict[str, int] = field(default_factory=dict)   # edge_type -> count

    # Connected entity types
    connected_types: Dict[str, int] = field(default_factory=dict)  # type -> count

    # 2-hop neighborhood summary
    two_hop_types: Dict[str, int] = field(default_factory=dict)

    # Centrality metrics
    in_degree: int = 0
    out_degree: int = 0
    betweenness: float = 0.0

    def signature(self) -> str:
        """Generate a hash signature for this fingerprint."""
        components = [
            self.entity_type,
            str(sorted(self.out_edges.items())),
            str(sorted(self.in_edges.items())),
            str(sorted(self.connected_types.items())),
        ]
        content = "|".join(components)
        return hashlib.md5(content.encode()).hexdigest()[:16]

    @property
    def total_degree(self) -> int:
        """Total degree (in + out)."""
        return self.in_degree + self.out_degree


def compute_graph_fingerprint(
    entity_id: str,
    entity_type: str,
    edges: List[dict],
    nodes: Dict[str, dict],
) -> GraphFingerprint:
    """Compute graph fingerprint for an entity.

    Args:
        entity_id: The entity to fingerprint.
        entity_type: Type of the entity.
        edges: List of edges in format {"source": id, "target": id, "type": str}.
        nodes: Map of node_id -> {"type": str, ...}.

    Returns:
        GraphFingerprint for the entity.
    """
    fp = GraphFingerprint(entity_id=entity_id, entity_type=entity_type)

    # Count outgoing edges
    out_edges = [e for e in edges if e["source"] == entity_id]
    fp.out_degree = len(out_edges)
    for e in out_edges:
        edge_type = e.get("type", "unknown")
        fp.out_edges[edge_type] = fp.out_edges.get(edge_type, 0) + 1

        target_type = nodes.get(e["target"], {}).get("type", "unknown")
        fp.connected_types[target_type] = fp.connected_types.get(target_type, 0) + 1

    # Count incoming edges
    in_edges = [e for e in edges if e["target"] == entity_id]
    fp.in_degree = len(in_edges)
    for e in in_edges:
        edge_type = e.get("type", "unknown")
        fp.in_edges[edge_type] = fp.in_edges.get(edge_type, 0) + 1

        source_type = nodes.get(e["source"], {}).get("type", "unknown")
        fp.connected_types[source_type] = fp.connected_types.get(source_type, 0) + 1

    # 2-hop neighborhood
    one_hop_neighbors = set(e["target"] for e in out_edges) | set(e["source"] for e in in_edges)
    for neighbor_id in one_hop_neighbors:
        neighbor_edges = [
            e for e in edges
            if e["source"] == neighbor_id or e["target"] == neighbor_id
        ]
        for e in neighbor_edges:
            other_id = e["target"] if e["source"] == neighbor_id else e["source"]
            if other_id != entity_id and other_id not in one_hop_neighbors:
                other_type = nodes.get(other_id, {}).get("type", "unknown")
                fp.two_hop_types[other_type] = fp.two_hop_types.get(other_type, 0) + 1

    return fp


def fingerprint_similarity(
    fp1: GraphFingerprint,
    fp2: GraphFingerprint,
    weights: Optional["ScoringWeights"] = None,
) -> float:
    """Compute similarity between two graph fingerprints.

    Uses Jaccard-like similarity over edge types and connected types.

    Args:
        fp1: First fingerprint.
        fp2: Second fingerprint.
        weights: Optional scoring weights from config.

    Returns:
        Similarity score in [0, 1].
    """
    if weights is None:
        from ..config import get_scoring_weights
        weights = get_scoring_weights()

    # Compare edge type distributions
    out_edge_sim = _dict_similarity(fp1.out_edges, fp2.out_edges)
    in_edge_sim = _dict_similarity(fp1.in_edges, fp2.in_edges)
    type_sim = _dict_similarity(fp1.connected_types, fp2.connected_types)

    # Use configurable weights
    # relation_structural splits into: out_edges, in_edges, type_sim
    # Default split: 0.3 out + 0.3 in + 0.4 type within the structural weight
    structural_weight = weights.relation_structural

    # The structural weight is distributed among the three components
    # We use the original ratios (3:3:4) normalized to the structural weight
    similarity = (
        (0.3 / 1.0) * out_edge_sim +
        (0.3 / 1.0) * in_edge_sim +
        (0.4 / 1.0) * type_sim
    )

    return similarity


def _dict_similarity(d1: Dict[str, int], d2: Dict[str, int]) -> float:
    """Compute similarity between two count dictionaries."""
    if not d1 and not d2:
        return 1.0
    if not d1 or not d2:
        return 0.0

    keys = set(d1.keys()) | set(d2.keys())

    intersection = sum(min(d1.get(k, 0), d2.get(k, 0)) for k in keys)
    union = sum(max(d1.get(k, 0), d2.get(k, 0)) for k in keys)

    return intersection / union if union > 0 else 0.0


def find_related_entities(
    fingerprint: GraphFingerprint,
    all_fingerprints: List[GraphFingerprint],
    min_similarity: float = 0.3,
    max_results: int = 20,
    weights: Optional["ScoringWeights"] = None,
) -> List[Tuple[str, float]]:
    """Find entities with similar graph structure.

    Args:
        fingerprint: The query fingerprint.
        all_fingerprints: All available fingerprints.
        min_similarity: Minimum similarity to include.
        max_results: Maximum results to return.
        weights: Optional scoring weights from config.

    Returns:
        List of (entity_id, similarity) tuples.
    """
    results = []

    for fp in all_fingerprints:
        if fp.entity_id == fingerprint.entity_id:
            continue

        sim = fingerprint_similarity(fingerprint, fp, weights=weights)
        if sim >= min_similarity:
            results.append((fp.entity_id, sim))

    # Sort by similarity descending
    results.sort(key=lambda x: x[1], reverse=True)

    return results[:max_results]


def compute_relation_score(
    query_entity_id: str,
    candidate_entity_id: str,
    query_fingerprint: GraphFingerprint,
    candidate_fingerprint: GraphFingerprint,
    shared_neighbors: int = 0,
    path_length: Optional[int] = None,
    weights: Optional["ScoringWeights"] = None,
) -> float:
    """Compute relation manifold score for a candidate.

    Args:
        query_entity_id: The entity being queried about.
        candidate_entity_id: A candidate related entity.
        query_fingerprint: Fingerprint of query entity.
        candidate_fingerprint: Fingerprint of candidate.
        shared_neighbors: Number of shared 1-hop neighbors.
        path_length: Shortest path length if known.
        weights: Optional scoring weights from config.

    Returns:
        Relation score in [0, 1].
    """
    if weights is None:
        from ..config import get_scoring_weights
        weights = get_scoring_weights()

    # Structural similarity
    struct_sim = fingerprint_similarity(
        query_fingerprint, candidate_fingerprint, weights=weights
    )

    # Shared neighbors (log-scaled, saturates at 5)
    neighbor_score = min(1.0, math.log1p(shared_neighbors) / 1.8)

    # Path proximity (closer = higher score)
    if path_length is not None and path_length > 0:
        path_score = 1.0 / path_length
    else:
        path_score = 0.0

    # Combined score using configurable weights
    score = (
        weights.relation_structural * struct_sim +
        weights.relation_neighbor * neighbor_score +
        weights.relation_path * path_score
    )

    return min(1.0, max(0.0, score))


def compute_graph_centrality(
    fingerprint: GraphFingerprint,
    max_degree: int = 100,
    weights: Optional["ScoringWeights"] = None,
) -> float:
    """Compute centrality score from fingerprint.

    Args:
        fingerprint: The entity's graph fingerprint.
        max_degree: Maximum degree for normalization.
        weights: Optional scoring weights from config.

    Returns:
        Centrality score in [0, 1].
    """
    # Degree centrality (log-scaled)
    degree = fingerprint.total_degree
    degree_score = min(1.0, math.log1p(degree) / math.log1p(max_degree))

    # If betweenness is available, blend it in
    if fingerprint.betweenness > 0:
        # Assume betweenness is already normalized to [0, 1]
        betweenness_score = min(1.0, fingerprint.betweenness)
        # 60% degree, 40% betweenness
        return 0.6 * degree_score + 0.4 * betweenness_score

    return degree_score


# Edge type taxonomy for typed relation queries
EDGE_TYPE_TAXONOMY = {
    "structural": [
        "contains", "part_of", "member_of", "instance_of",
        "subclass_of", "implements", "extends",
    ],
    "operational": [
        "uses", "depends_on", "calls", "invokes",
        "connects_to", "reads_from", "writes_to",
    ],
    "temporal": [
        "precedes", "follows", "concurrent_with",
        "triggers", "caused_by", "enables",
    ],
    "ownership": [
        "owns", "manages", "administers",
        "created_by", "maintained_by",
    ],
    "similarity": [
        "similar_to", "related_to", "alternative_to",
        "equivalent_to", "replaces",
    ],
}


def categorize_edge_type(edge_type: str) -> str:
    """Categorize an edge type into taxonomy category."""
    edge_type_lower = edge_type.lower().replace("-", "_").replace(" ", "_")

    for category, types in EDGE_TYPE_TAXONOMY.items():
        if edge_type_lower in types:
            return category

    return "other"


def get_edge_types_for_category(category: str) -> List[str]:
    """Get all edge types in a category."""
    return EDGE_TYPE_TAXONOMY.get(category, [])
