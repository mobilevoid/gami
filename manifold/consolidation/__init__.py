"""Memory consolidation module for GAMI.

Provides clustering, abstraction, and decay functionality for the dream cycle.
"""

from .clusterer import cluster_memories, ClusterResult
from .abstractor import generate_abstraction, generate_abstractions_batch
from .decay import apply_decay, archive_decayed, calculate_decay_score

__all__ = [
    "cluster_memories",
    "ClusterResult",
    "generate_abstraction",
    "generate_abstractions_batch",
    "apply_decay",
    "archive_decayed",
    "calculate_decay_score",
]
