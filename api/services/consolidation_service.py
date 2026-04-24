"""
Memory consolidation service.

Implements sleep-like memory consolidation:
1. Clustering - Group similar memories
2. Abstraction - Generate merged representations
3. Decay - Reduce strength of unused memories
4. Inference - Generate new knowledge from patterns
"""

import logging
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import math

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from manifold.config_v2 import ManifoldConfigV2, ConsolidationConfig, get_config

logger = logging.getLogger("gami.consolidation_service")


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class MemoryCluster:
    """A cluster of related memories."""
    cluster_id: str
    member_ids: List[str]
    centroid_id: str
    abstraction_text: Optional[str]
    stability_score: float
    access_count: int
    current_decay: float


@dataclass
class ConsolidationStats:
    """Statistics from consolidation process."""
    memories_processed: int
    clusters_created: int
    clusters_merged: int
    abstractions_generated: int
    segments_decayed: int
    segments_archived: int
    inferences_generated: int


@dataclass
class GeneratedInference:
    """An inference generated from cluster patterns."""
    inference_text: str
    confidence: float
    inference_type: str  # deductive, inductive, abductive
    source_cluster_ids: List[str]
    premises: List[str]


# =============================================================================
# CLUSTERING
# =============================================================================

def cluster_embeddings(
    embeddings: np.ndarray,
    ids: List[str],
    threshold: float = 0.85,
    min_size: int = 3,
    max_size: int = 50,
) -> List[List[str]]:
    """Cluster embeddings using agglomerative clustering.

    Args:
        embeddings: (N, dim) array of embeddings
        ids: List of memory IDs
        threshold: Cosine similarity threshold
        min_size: Minimum cluster size
        max_size: Maximum cluster size

    Returns:
        List of clusters, each cluster is a list of IDs
    """
    if len(embeddings) < min_size:
        return []

    try:
        from sklearn.cluster import AgglomerativeClustering

        # Convert similarity threshold to distance threshold
        distance_threshold = 1 - threshold

        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric="cosine",
            linkage="average"
        )

        labels = clustering.fit_predict(embeddings)

        # Group by cluster label
        clusters_dict: Dict[int, List[str]] = {}
        for idx, label in enumerate(labels):
            if label not in clusters_dict:
                clusters_dict[label] = []
            clusters_dict[label].append(ids[idx])

        # Filter by size constraints
        clusters = [
            c[:max_size] for c in clusters_dict.values()
            if len(c) >= min_size
        ]

        return clusters

    except ImportError:
        logger.warning("sklearn not available, using simple threshold clustering")
        return _simple_cluster(embeddings, ids, threshold, min_size, max_size)


def _simple_cluster(
    embeddings: np.ndarray,
    ids: List[str],
    threshold: float,
    min_size: int,
    max_size: int,
) -> List[List[str]]:
    """Simple greedy clustering fallback."""
    n = len(embeddings)
    if n < min_size:
        return []

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.maximum(norms, 1e-8)

    # Compute pairwise similarities
    similarities = np.dot(normalized, normalized.T)

    used = set()
    clusters = []

    for i in range(n):
        if i in used:
            continue

        cluster = [i]
        used.add(i)

        for j in range(i + 1, n):
            if j in used:
                continue

            if similarities[i, j] >= threshold:
                cluster.append(j)
                used.add(j)

                if len(cluster) >= max_size:
                    break

        if len(cluster) >= min_size:
            clusters.append([ids[idx] for idx in cluster])

    return clusters


def find_centroid(
    embeddings: np.ndarray,
    ids: List[str],
) -> Tuple[str, np.ndarray]:
    """Find the centroid member of a cluster.

    Returns the ID and embedding of the member closest to the mean.
    """
    centroid = embeddings.mean(axis=0)
    distances = np.linalg.norm(embeddings - centroid, axis=1)
    closest_idx = np.argmin(distances)

    return ids[closest_idx], embeddings[closest_idx]


# =============================================================================
# CONSOLIDATION SERVICE
# =============================================================================

class ConsolidationService:
    """Service for memory consolidation operations."""

    def __init__(self, config: Optional[ManifoldConfigV2] = None):
        """Initialize consolidation service.

        Args:
            config: Manifold configuration
        """
        self._config = config

    @property
    def config(self) -> ConsolidationConfig:
        """Get consolidation config."""
        if self._config is None:
            self._config = get_config()
        return self._config.consolidation

    async def run_consolidation(
        self,
        db: AsyncSession,
        tenant_id: str,
        max_memories: int = 500,
        max_clusters: int = 50,
    ) -> ConsolidationStats:
        """Run full consolidation cycle.

        Args:
            db: Database session
            tenant_id: Tenant to consolidate
            max_memories: Maximum memories to process
            max_clusters: Maximum clusters to create

        Returns:
            ConsolidationStats with results
        """
        if not self.config.enabled:
            return ConsolidationStats(0, 0, 0, 0, 0, 0, 0)

        stats = ConsolidationStats(0, 0, 0, 0, 0, 0, 0)

        # Phase 1: Clustering
        cluster_stats = await self._cluster_memories(db, tenant_id, max_memories, max_clusters)
        stats.memories_processed = cluster_stats[0]
        stats.clusters_created = cluster_stats[1]

        # Phase 2: Abstraction (if enabled)
        if self.config.abstraction_enabled:
            stats.abstractions_generated = await self._generate_abstractions(db, tenant_id)

        # Phase 3: Decay (if enabled)
        if self.config.decay_enabled:
            decay_stats = await self._apply_decay(db, tenant_id)
            stats.segments_decayed = decay_stats[0]
            stats.segments_archived = decay_stats[1]

        # Phase 4: Inference (if enabled)
        if self.config.inference_enabled:
            stats.inferences_generated = await self._generate_inferences(db, tenant_id)

        logger.info(
            f"Consolidation complete for {tenant_id}: "
            f"{stats.clusters_created} clusters, {stats.abstractions_generated} abstractions, "
            f"{stats.segments_decayed} decayed, {stats.inferences_generated} inferences"
        )

        return stats

    async def _cluster_memories(
        self,
        db: AsyncSession,
        tenant_id: str,
        max_memories: int,
        max_clusters: int,
    ) -> Tuple[int, int]:
        """Cluster unclustered memories.

        Returns: (memories_processed, clusters_created)
        """
        # Get unclustered memories with embeddings
        result = await db.execute(text("""
            SELECT memory_id, normalized_text, embedding
            FROM assistant_memories
            WHERE cluster_id IS NULL
            AND embedding IS NOT NULL
            AND owner_tenant_id = :tenant_id
            AND status = 'active'
            ORDER BY created_at DESC
            LIMIT :limit
        """), {"tenant_id": tenant_id, "limit": max_memories})

        rows = result.fetchall()
        if len(rows) < self.config.min_cluster_size:
            return 0, 0

        # Parse embeddings
        ids = []
        texts = []
        embeddings = []

        for row in rows:
            try:
                # Parse vector string to numpy array
                emb = self._parse_vector(row.embedding)
                if emb is not None:
                    ids.append(row.memory_id)
                    texts.append(row.normalized_text)
                    embeddings.append(emb)
            except Exception as e:
                logger.debug(f"Failed to parse embedding for {row.memory_id}: {e}")

        if len(embeddings) < self.config.min_cluster_size:
            return 0, 0

        embeddings_array = np.array(embeddings)

        # Cluster
        clusters = cluster_embeddings(
            embeddings_array,
            ids,
            threshold=self.config.cluster_similarity_threshold,
            min_size=self.config.min_cluster_size,
            max_size=self.config.max_cluster_size,
        )

        # Save clusters
        created = 0
        for cluster_ids in clusters[:max_clusters]:
            cluster_embeddings_array = embeddings_array[[ids.index(cid) for cid in cluster_ids]]

            centroid_id, centroid_emb = find_centroid(cluster_embeddings_array, cluster_ids)

            # Generate cluster ID
            cluster_id = f"CLST_{hashlib.md5(''.join(cluster_ids).encode()).hexdigest()[:12]}"

            # Calculate stability based on cluster size
            stability = 0.5 + 0.1 * min(5, len(cluster_ids))

            await db.execute(text("""
                INSERT INTO memory_clusters (
                    cluster_id, owner_tenant_id, member_ids, member_count,
                    centroid_segment_id, cluster_embedding, stability_score
                ) VALUES (
                    :cluster_id, :tenant_id, :members, :count,
                    :centroid, :embedding, :stability
                )
                ON CONFLICT (cluster_id) DO UPDATE SET
                    member_ids = :members,
                    member_count = :count,
                    stability_score = GREATEST(memory_clusters.stability_score, :stability),
                    updated_at = NOW()
            """), {
                "cluster_id": cluster_id,
                "tenant_id": tenant_id,
                "members": cluster_ids,
                "count": len(cluster_ids),
                "centroid": centroid_id,
                "embedding": self._format_vector(centroid_emb),
                "stability": stability,
            })

            # Update member memories
            for mid in cluster_ids:
                await db.execute(text("""
                    UPDATE assistant_memories
                    SET cluster_id = :cluster_id
                    WHERE memory_id = :mid
                """), {"cluster_id": cluster_id, "mid": mid})

            created += 1

        await db.commit()
        return len(ids), created

    async def _generate_abstractions(
        self,
        db: AsyncSession,
        tenant_id: str,
    ) -> int:
        """Generate abstractions for clusters without them.

        Returns: Number of abstractions generated
        """
        # Get clusters without abstractions
        result = await db.execute(text("""
            SELECT cluster_id, member_ids
            FROM memory_clusters
            WHERE abstraction_text IS NULL
            AND owner_tenant_id = :tenant_id
            AND status = 'active'
            AND member_count >= :min_size
            LIMIT 10
        """), {"tenant_id": tenant_id, "min_size": self.config.min_cluster_size})

        clusters = result.fetchall()
        if not clusters:
            return 0

        generated = 0

        for cluster in clusters:
            # Get member texts
            members_result = await db.execute(text("""
                SELECT normalized_text
                FROM assistant_memories
                WHERE memory_id = ANY(:ids)
            """), {"ids": cluster.member_ids})

            texts = [r.normalized_text for r in members_result.fetchall()]

            if not texts:
                continue

            # Generate abstraction
            try:
                abstraction = await self._call_llm_abstraction(texts)

                if abstraction:
                    # Generate embedding for abstraction
                    abstraction_emb = await self._embed_text(abstraction)

                    await db.execute(text("""
                        UPDATE memory_clusters SET
                            abstraction_text = :text,
                            abstraction_embedding = :embedding,
                            updated_at = NOW()
                        WHERE cluster_id = :cluster_id
                    """), {
                        "text": abstraction,
                        "embedding": self._format_vector(abstraction_emb) if abstraction_emb else None,
                        "cluster_id": cluster.cluster_id,
                    })

                    generated += 1

            except Exception as e:
                logger.warning(f"Failed to generate abstraction for {cluster.cluster_id}: {e}")

        await db.commit()
        return generated

    async def _apply_decay(
        self,
        db: AsyncSession,
        tenant_id: str,
    ) -> Tuple[int, int]:
        """Apply decay to unused memories.

        Returns: (decayed_count, archived_count)
        """
        decay_rate = self.config.decay_rate_per_day
        archive_threshold = self.config.archive_threshold

        # Apply decay to segments not accessed recently
        result = await db.execute(text("""
            UPDATE segments SET
                decay_score = GREATEST(:threshold,
                    COALESCE(decay_score, 1.0) * (1 - :rate *
                        EXTRACT(EPOCH FROM (NOW() - COALESCE(last_retrieved_at, created_at)))
                        / 86400.0
                    )
                ),
                updated_at = NOW()
            WHERE owner_tenant_id = :tenant_id
            AND COALESCE(last_retrieved_at, created_at) < NOW() - INTERVAL '7 days'
            AND COALESCE(decay_score, 1.0) > :threshold
            AND status = 'active'
            RETURNING segment_id
        """), {
            "tenant_id": tenant_id,
            "rate": decay_rate,
            "threshold": archive_threshold,
        })

        decayed_count = len(result.fetchall())

        # Archive heavily decayed segments
        result = await db.execute(text("""
            UPDATE segments
            SET status = 'archived', updated_at = NOW()
            WHERE owner_tenant_id = :tenant_id
            AND COALESCE(decay_score, 1.0) <= :threshold
            AND status = 'active'
            RETURNING segment_id
        """), {"tenant_id": tenant_id, "threshold": archive_threshold})

        archived_count = len(result.fetchall())

        # Also decay memory clusters
        await db.execute(text("""
            UPDATE memory_clusters SET
                current_decay = GREATEST(:threshold,
                    COALESCE(current_decay, 1.0) * (1 - :rate *
                        EXTRACT(EPOCH FROM (NOW() - last_accessed_at)) / 86400.0
                    )
                ),
                updated_at = NOW()
            WHERE owner_tenant_id = :tenant_id
            AND last_accessed_at < NOW() - INTERVAL '7 days'
            AND COALESCE(current_decay, 1.0) > :threshold
            AND status = 'active'
        """), {"tenant_id": tenant_id, "rate": decay_rate, "threshold": archive_threshold})

        await db.commit()
        return decayed_count, archived_count

    async def _generate_inferences(
        self,
        db: AsyncSession,
        tenant_id: str,
    ) -> int:
        """Generate inferences from stable clusters.

        Returns: Number of inferences generated
        """
        # Get stable clusters with abstractions
        result = await db.execute(text("""
            SELECT cluster_id, abstraction_text, stability_score
            FROM memory_clusters
            WHERE abstraction_text IS NOT NULL
            AND stability_score >= :threshold
            AND owner_tenant_id = :tenant_id
            AND status = 'active'
            ORDER BY stability_score DESC
            LIMIT 20
        """), {
            "tenant_id": tenant_id,
            "threshold": self.config.inference_confidence_threshold,
        })

        clusters = result.fetchall()
        if len(clusters) < 2:
            return 0

        # Try to generate inferences from cluster pairs
        generated = 0
        abstractions = [(c.cluster_id, c.abstraction_text) for c in clusters]

        for i in range(len(abstractions)):
            for j in range(i + 1, min(i + 5, len(abstractions))):
                try:
                    inference = await self._infer_from_pair(
                        abstractions[i][1],
                        abstractions[j][1],
                    )

                    if inference and inference.confidence >= self.config.inference_confidence_threshold:
                        # Store inference as a new claim or memory
                        await self._store_inference(
                            db,
                            inference,
                            tenant_id,
                            [abstractions[i][0], abstractions[j][0]],
                        )
                        generated += 1

                except Exception as e:
                    logger.debug(f"Inference generation failed: {e}")

        await db.commit()
        return generated

    async def _call_llm_abstraction(self, texts: List[str]) -> Optional[str]:
        """Call LLM to generate abstraction from texts."""
        try:
            from .prompt_service import get_default_prompt
            from .extraction import call_vllm

            variables = {"texts": texts[:10]}  # Limit input texts
            system_prompt, user_prompt, params = get_default_prompt(
                "abstraction_generation",
                variables,
            )

            response = await call_vllm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=params.get("temperature", 0.4),
                max_tokens=params.get("max_tokens", 500),
            )

            return response.strip() if response else None

        except Exception as e:
            logger.warning(f"LLM abstraction failed: {e}")
            return None

    async def _embed_text(self, text: str) -> Optional[np.ndarray]:
        """Embed text using Ollama."""
        try:
            from .extraction import embed_text_sync
            return embed_text_sync(text)
        except Exception as e:
            logger.warning(f"Text embedding failed: {e}")
            return None

    async def _infer_from_pair(
        self,
        abstraction1: str,
        abstraction2: str,
    ) -> Optional[GeneratedInference]:
        """Try to generate inference from two abstractions."""
        try:
            from .prompt_service import get_default_prompt
            from .extraction import call_vllm
            import json

            variables = {"facts": [abstraction1, abstraction2]}
            system_prompt, user_prompt, params = get_default_prompt(
                "inference_generation",
                variables,
            )

            response = await call_vllm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,
                max_tokens=500,
            )

            # Parse response
            import re
            json_match = re.search(r'\[[\s\S]*\]', response)
            if not json_match:
                return None

            data = json.loads(json_match.group())
            if not data:
                return None

            item = data[0]
            return GeneratedInference(
                inference_text=item.get("inference", ""),
                confidence=float(item.get("confidence", 0.5)),
                inference_type=item.get("inference_type", "inductive"),
                source_cluster_ids=[],
                premises=[abstraction1, abstraction2],
            )

        except Exception as e:
            logger.debug(f"Inference from pair failed: {e}")
            return None

    async def _store_inference(
        self,
        db: AsyncSession,
        inference: GeneratedInference,
        tenant_id: str,
        cluster_ids: List[str],
    ) -> None:
        """Store a generated inference."""
        import secrets

        # Store as a derived claim
        claim_id = f"CLM_INF_{secrets.token_hex(6)}"

        await db.execute(text("""
            INSERT INTO claims (
                claim_id, subject_text, predicate_text, object_text,
                confidence_score, claim_type, modality,
                owner_tenant_id, derived_from, derivation_type
            ) VALUES (
                :claim_id, :subject, 'inferred_relation', :object,
                :confidence, 'inference', 'possibility',
                :tenant_id, :derived_from, :derivation_type
            )
        """), {
            "claim_id": claim_id,
            "subject": inference.premises[0][:200] if inference.premises else "",
            "object": inference.inference_text[:500],
            "confidence": inference.confidence,
            "tenant_id": tenant_id,
            "derived_from": cluster_ids,
            "derivation_type": inference.inference_type,
        })

        # Update cluster inference_ids
        for cluster_id in cluster_ids:
            await db.execute(text("""
                UPDATE memory_clusters SET
                    inference_ids = COALESCE(inference_ids, ARRAY[]::text[]) || :claim_id,
                    updated_at = NOW()
                WHERE cluster_id = :cluster_id
            """), {"claim_id": claim_id, "cluster_id": cluster_id})

    def _parse_vector(self, vector_str: str) -> Optional[np.ndarray]:
        """Parse vector string to numpy array."""
        if not vector_str:
            return None

        try:
            # Handle PostgreSQL vector format [x,y,z]
            clean = vector_str.strip("[]")
            values = [float(x) for x in clean.split(",")]
            return np.array(values)
        except Exception:
            return None

    def _format_vector(self, vector: np.ndarray) -> str:
        """Format numpy array as PostgreSQL vector string."""
        return "[" + ",".join(str(x) for x in vector) + "]"


# =============================================================================
# REINFORCEMENT
# =============================================================================

async def reinforce_memory(
    db: AsyncSession,
    memory_id: str,
    boost: Optional[float] = None,
) -> bool:
    """Reinforce a memory (reset decay, boost stability).

    Args:
        db: Database session
        memory_id: Memory to reinforce
        boost: Optional stability boost (default from config)

    Returns:
        True if reinforced
    """
    config = get_config().consolidation
    boost = boost or config.reinforcement_boost

    result = await db.execute(text("""
        UPDATE assistant_memories SET
            decay_score = 1.0,
            stability_score = LEAST(1.0, COALESCE(stability_score, 0.5) + :boost),
            last_reinforced_at = NOW(),
            updated_at = NOW()
        WHERE memory_id = :mid
        RETURNING memory_id
    """), {"mid": memory_id, "boost": boost})

    success = result.fetchone() is not None

    if success:
        await db.commit()

        # Also reinforce the cluster if applicable
        await db.execute(text("""
            UPDATE memory_clusters SET
                current_decay = 1.0,
                repetition_count = repetition_count + 1,
                last_reinforced_at = NOW(),
                last_accessed_at = NOW(),
                access_count = access_count + 1,
                stability_score = LEAST(1.0, stability_score + :boost / 2),
                updated_at = NOW()
            WHERE cluster_id = (
                SELECT cluster_id FROM assistant_memories WHERE memory_id = :mid
            )
        """), {"mid": memory_id, "boost": boost})

        await db.commit()

    return success


# =============================================================================
# GLOBAL SERVICE INSTANCE
# =============================================================================

_consolidation_service: Optional[ConsolidationService] = None


def get_consolidation_service() -> ConsolidationService:
    """Get global consolidation service instance."""
    global _consolidation_service
    if _consolidation_service is None:
        _consolidation_service = ConsolidationService()
    return _consolidation_service
