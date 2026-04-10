"""Graph expansion — controlled beam-style expansion from anchors.

The graph expansion process:
1. Start from high-scoring anchor candidates
2. Expand through typed edges with weights
3. Apply drift penalty (distance from query topic)
4. Apply contradiction penalty
5. Limit depth, beam width, and fan-out
6. Return scored paths for reranking

Uses Apache AGE as the graph engine (Neo4j is future option).
"""
import logging
from typing import Optional, List, Dict, Any, Set
from dataclasses import dataclass, field

logger = logging.getLogger("gami.manifold.graph.expansion")


# Relation type priors — higher = more valuable for retrieval
RELATION_PRIORS = {
    "SUPPORTS": 1.00,
    "DERIVED_FROM": 0.95,
    "SUMMARIZES": 0.90,
    "PART_OF": 0.85,
    "INSTANCE_OF": 0.80,
    "UPDATED_BY": 0.80,
    "CONTRADICTS": 0.75,
    "MENTIONS": 0.70,
    "ABOUT": 0.65,
    "RELATED_TO": 0.35,
    "ALIAS_OF": 0.60,
    "PRECEDES": 0.55,
    "FOLLOWS": 0.55,
    "CAUSED_BY": 0.65,
    "LOCATED_IN": 0.50,
    "AUTHORED_BY": 0.55,
    "USES": 0.60,
    "DEFINED_BY": 0.70,
    "ELABORATES": 0.65,
    "SUPERSEDED_BY": 0.75,
    "PREFERENCE_OF": 0.70,
    "CORRECTION_OF": 0.80,
}


# Expansion parameters
DEFAULT_MAX_DEPTH = 3
DEFAULT_BEAM_WIDTH = 20
DEFAULT_FANOUT_LIMIT = 12
DEFAULT_CANDIDATE_THRESHOLD = 0.55
DEFAULT_DEPTH_PENALTY = 0.15
DEFAULT_DRIFT_PENALTY = 0.20
DEFAULT_CONTRADICTION_PENALTY = 0.25


@dataclass
class GraphEdge:
    """Represents an edge in the graph."""
    source_id: str
    target_id: str
    relation_type: str
    weight: float = 1.0
    confidence: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphNode:
    """Represents a node in the graph."""
    node_id: str
    node_type: str
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphPath:
    """Represents a path through the graph."""
    anchor_id: str
    anchor_score: float
    edges: List[GraphEdge] = field(default_factory=list)
    nodes: List[GraphNode] = field(default_factory=list)
    path_score: float = 0.0
    drift_score: float = 0.0
    contradiction_count: int = 0


class GraphExpander:
    """Performs controlled graph expansion from anchor candidates.

    Uses beam-style search with:
    - Type-weighted edges
    - Depth penalty
    - Drift penalty (distance from query topic)
    - Contradiction awareness
    - Fan-out limits
    """

    def __init__(
        self,
        max_depth: int = DEFAULT_MAX_DEPTH,
        beam_width: int = DEFAULT_BEAM_WIDTH,
        fanout_limit: int = DEFAULT_FANOUT_LIMIT,
        candidate_threshold: float = DEFAULT_CANDIDATE_THRESHOLD,
        relation_priors: Optional[Dict[str, float]] = None,
    ):
        """Initialize the graph expander.

        Args:
            max_depth: Maximum expansion depth.
            beam_width: Number of paths to keep at each level.
            fanout_limit: Maximum edges to explore from each node.
            candidate_threshold: Minimum score to include in results.
            relation_priors: Custom relation type weights.
        """
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.fanout_limit = fanout_limit
        self.candidate_threshold = candidate_threshold
        self.relation_priors = relation_priors or RELATION_PRIORS.copy()

    def expand(
        self,
        anchor_ids: List[str],
        anchor_scores: Dict[str, float],
        query_embedding: Optional[List[float]] = None,
    ) -> List[GraphPath]:
        """Expand graph from anchor nodes.

        Args:
            anchor_ids: IDs of anchor nodes to start from.
            anchor_scores: Scores for each anchor.
            query_embedding: Query embedding for drift calculation.

        Returns:
            List of scored GraphPath objects.
        """
        # STUB: Graph expansion not connected in isolated module
        # This will be connected to AGE during activation
        logger.debug("Graph expansion not connected to database")

        # Return placeholder paths
        paths = []
        for anchor_id in anchor_ids[:self.beam_width]:
            path = GraphPath(
                anchor_id=anchor_id,
                anchor_score=anchor_scores.get(anchor_id, 0.0),
                edges=[],
                nodes=[],
                path_score=anchor_scores.get(anchor_id, 0.0),
            )
            paths.append(path)

        return paths

    def compute_path_score(
        self,
        path: GraphPath,
        query_embedding: Optional[List[float]] = None,
    ) -> float:
        """Compute score for a graph path.

        Formula:
            S_path = S_anchor
                   + Σ [λ_rel(type) · weight · confidence]
                   - γ_depth · depth
                   - γ_drift · drift
                   - γ_contra · contradiction_ratio

        Args:
            path: The path to score.
            query_embedding: Query embedding for drift calculation.

        Returns:
            Path score.
        """
        # Start with anchor score
        score = path.anchor_score

        # Add edge contributions
        for edge in path.edges:
            relation_prior = self.relation_priors.get(edge.relation_type, 0.5)
            edge_contribution = relation_prior * edge.weight * edge.confidence
            score += edge_contribution

        # Apply depth penalty
        depth = len(path.edges)
        score -= DEFAULT_DEPTH_PENALTY * depth

        # Apply drift penalty
        if path.drift_score > 0:
            score -= DEFAULT_DRIFT_PENALTY * path.drift_score

        # Apply contradiction penalty
        if path.contradiction_count > 0:
            contradiction_ratio = path.contradiction_count / (len(path.edges) + 1)
            score -= DEFAULT_CONTRADICTION_PENALTY * contradiction_ratio

        return max(0.0, score)

    def compute_drift(
        self,
        path: GraphPath,
        query_embedding: List[float],
    ) -> float:
        """Compute semantic drift from query topic.

        Args:
            path: The path to measure.
            query_embedding: Query embedding.

        Returns:
            Drift score (0-1, higher = more drift).
        """
        # STUB: Would compute centroid of path nodes and compare to query
        # For now, assume moderate drift for longer paths
        return len(path.edges) * 0.1

    def count_contradictions(
        self,
        path: GraphPath,
    ) -> int:
        """Count contradiction edges in path.

        Args:
            path: The path to check.

        Returns:
            Number of CONTRADICTS edges.
        """
        return sum(
            1 for edge in path.edges
            if edge.relation_type == "CONTRADICTS"
        )


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------

def expand_from_anchors(
    anchor_ids: List[str],
    anchor_scores: Dict[str, float],
    max_depth: int = 2,
    beam_width: int = 20,
) -> List[GraphPath]:
    """Convenience function for graph expansion.

    Args:
        anchor_ids: Anchor node IDs.
        anchor_scores: Scores for anchors.
        max_depth: Maximum depth.
        beam_width: Beam width.

    Returns:
        List of GraphPath.
    """
    expander = GraphExpander(max_depth=max_depth, beam_width=beam_width)
    return expander.expand(anchor_ids, anchor_scores)


def score_path(
    path: GraphPath,
    query_embedding: Optional[List[float]] = None,
) -> float:
    """Convenience function for path scoring.

    Args:
        path: The path to score.
        query_embedding: Optional query embedding.

    Returns:
        Path score.
    """
    expander = GraphExpander()
    return expander.compute_path_score(path, query_embedding)
