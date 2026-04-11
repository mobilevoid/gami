"""Memory clustering for GAMI consolidation.

Uses sklearn AgglomerativeClustering with cosine distance to group
similar memories/segments together for abstraction.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger("gami.consolidation.clusterer")


@dataclass
class ClusterResult:
    """Result of clustering operation."""
    cluster_id: int
    member_ids: List[str]
    member_count: int
    centroid_id: str  # ID of most representative member
    centroid_embedding: np.ndarray
    stability_score: float


def cluster_memories(
    embeddings: np.ndarray,
    ids: List[str],
    texts: Optional[List[str]] = None,
    similarity_threshold: float = 0.85,
    min_cluster_size: int = 3,
    max_cluster_size: int = 50,
) -> List[ClusterResult]:
    """Cluster memories by embedding similarity.

    Args:
        embeddings: (N, dim) array of embeddings
        ids: List of memory/segment IDs corresponding to embeddings
        texts: Optional list of texts (for logging)
        similarity_threshold: Cosine similarity threshold for clustering (0.85 = very similar)
        min_cluster_size: Minimum members for a valid cluster
        max_cluster_size: Maximum members per cluster (truncate larger)

    Returns:
        List of ClusterResult objects for valid clusters
    """
    from sklearn.cluster import AgglomerativeClustering

    if len(embeddings) < min_cluster_size:
        logger.debug("Not enough items to cluster (%d < %d)", len(embeddings), min_cluster_size)
        return []

    # Normalize embeddings for cosine distance
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    normalized = embeddings / norms

    # Convert similarity threshold to distance threshold
    # cosine_distance = 1 - cosine_similarity
    distance_threshold = 1 - similarity_threshold

    try:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric="cosine",
            linkage="average",
        )
        labels = clustering.fit_predict(normalized)
    except Exception as e:
        logger.error("Clustering failed: %s", e)
        return []

    # Group by cluster label
    clusters_dict = {}
    for idx, label in enumerate(labels):
        if label not in clusters_dict:
            clusters_dict[label] = []
        clusters_dict[label].append(idx)

    # Build ClusterResult for each valid cluster
    results = []
    for cluster_id, indices in clusters_dict.items():
        if len(indices) < min_cluster_size:
            continue

        # Truncate large clusters
        if len(indices) > max_cluster_size:
            indices = indices[:max_cluster_size]

        member_ids = [ids[i] for i in indices]
        cluster_embeddings = embeddings[indices]

        # Find centroid (mean) and most representative member
        centroid = cluster_embeddings.mean(axis=0)
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        centroid_idx = indices[np.argmin(distances)]
        centroid_id = ids[centroid_idx]

        # Calculate stability score based on cluster tightness
        # Tighter clusters (smaller average distance) = more stable
        avg_distance = distances.mean()
        # Map to 0-1 scale: distance 0 = stability 1.0, distance 0.5 = stability 0.5
        stability = max(0.0, min(1.0, 1.0 - avg_distance))

        # Boost stability for larger clusters
        size_bonus = min(0.2, (len(indices) - min_cluster_size) * 0.02)
        stability = min(1.0, stability + size_bonus)

        results.append(ClusterResult(
            cluster_id=int(cluster_id),
            member_ids=member_ids,
            member_count=len(member_ids),
            centroid_id=centroid_id,
            centroid_embedding=centroid,
            stability_score=round(stability, 3),
        ))

    logger.info(
        "Clustered %d items into %d valid clusters (threshold=%.2f)",
        len(embeddings), len(results), similarity_threshold
    )

    return results


def find_cluster_for_item(
    item_embedding: np.ndarray,
    existing_clusters: List[Tuple[str, np.ndarray]],  # (cluster_id, centroid_embedding)
    similarity_threshold: float = 0.80,
) -> Optional[str]:
    """Find existing cluster for a new item, if similar enough.

    Args:
        item_embedding: Embedding of the new item
        existing_clusters: List of (cluster_id, centroid_embedding) tuples
        similarity_threshold: Minimum similarity to join cluster

    Returns:
        cluster_id if match found, None otherwise
    """
    if not existing_clusters:
        return None

    item_norm = np.linalg.norm(item_embedding)
    if item_norm == 0:
        return None
    item_normalized = item_embedding / item_norm

    best_cluster = None
    best_similarity = similarity_threshold

    for cluster_id, centroid in existing_clusters:
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm == 0:
            continue
        centroid_normalized = centroid / centroid_norm

        similarity = np.dot(item_normalized, centroid_normalized)
        if similarity > best_similarity:
            best_similarity = similarity
            best_cluster = cluster_id

    return best_cluster
