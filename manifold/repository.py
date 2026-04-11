"""Repository layer for manifold data persistence.

Provides async database access for all manifold-related tables.
Uses asyncpg for PostgreSQL with pgvector support.
"""
import logging
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

logger = logging.getLogger("gami.manifold.repository")

# Type alias for database connection
# In production: asyncpg.Connection
Connection = Any


@dataclass
class ManifoldEmbeddingRecord:
    """A manifold embedding record from the database."""
    id: str
    object_id: str
    object_type: str
    manifold_type: str
    embedding: List[float]
    text_used: str
    created_at: datetime
    updated_at: Optional[datetime] = None


@dataclass
class PromotionScoreRecord:
    """A promotion score record from the database."""
    object_id: str
    object_type: str
    score: float
    importance: float
    retrieval_frequency: float
    source_diversity: float
    confidence: float
    novelty: float
    graph_centrality: float
    user_relevance: float
    should_promote: bool
    computed_at: datetime


@dataclass
class CanonicalClaimRecord:
    """A canonical claim record from the database."""
    id: str
    claim_id: str
    subject: str
    predicate: str
    object: str
    modality: str
    temporal_scope: Optional[str]
    qualifiers: Dict[str, Any]
    canonical_text: str
    created_at: datetime


@dataclass
class TemporalFeaturesRecord:
    """Temporal features record from the database."""
    id: str
    object_id: str
    object_type: str
    features: List[float]  # 12-dimensional
    timestamp: Optional[datetime]
    created_at: datetime


class ManifoldRepository:
    """Repository for manifold embeddings and related data."""

    def __init__(self, pool=None):
        """Initialize with connection pool.

        Args:
            pool: asyncpg connection pool (None for stub mode).
        """
        self.pool = pool

    async def get_connection(self) -> Connection:
        """Get a database connection."""
        if self.pool is None:
            raise RuntimeError("Database pool not initialized")
        return await self.pool.acquire()

    async def release_connection(self, conn: Connection):
        """Release a database connection."""
        if self.pool is not None:
            await self.pool.release(conn)

    # === Manifold Embeddings ===

    async def get_embedding(
        self,
        object_id: str,
        manifold_type: str,
    ) -> Optional[ManifoldEmbeddingRecord]:
        """Get embedding for an object."""
        if self.pool is None:
            return None

        conn = await self.get_connection()
        try:
            row = await conn.fetchrow(
                """
                SELECT id, object_id, object_type, manifold_type,
                       embedding, text_used, created_at, updated_at
                FROM manifold_embeddings
                WHERE object_id = $1 AND manifold_type = $2
                """,
                object_id,
                manifold_type,
            )
            if row:
                return ManifoldEmbeddingRecord(**dict(row))
            return None
        finally:
            await self.release_connection(conn)

    async def upsert_embedding(
        self,
        object_id: str,
        object_type: str,
        manifold_type: str,
        embedding: List[float],
        text_used: str,
    ) -> str:
        """Insert or update an embedding."""
        if self.pool is None:
            return ""

        conn = await self.get_connection()
        try:
            result = await conn.fetchval(
                """
                INSERT INTO manifold_embeddings (
                    object_id, object_type, manifold_type,
                    embedding, text_used, created_at
                ) VALUES ($1, $2, $3, $4, $5, NOW())
                ON CONFLICT (object_id, manifold_type) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    text_used = EXCLUDED.text_used,
                    updated_at = NOW()
                RETURNING id
                """,
                object_id,
                object_type,
                manifold_type,
                embedding,
                text_used[:1000],
            )
            return result
        finally:
            await self.release_connection(conn)

    async def search_by_embedding(
        self,
        embedding: List[float],
        manifold_type: str,
        limit: int = 20,
        tenant_id: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """Search for similar embeddings.

        Returns list of (object_id, similarity_score) tuples.
        """
        if self.pool is None:
            return []

        conn = await self.get_connection()
        try:
            # Using pgvector cosine distance
            query = """
                SELECT object_id, 1 - (embedding <=> $1::vector) as similarity
                FROM manifold_embeddings
                WHERE manifold_type = $2
            """
            params = [embedding, manifold_type]

            if tenant_id:
                query += " AND tenant_id = $3"
                params.append(tenant_id)

            query += " ORDER BY embedding <=> $1::vector LIMIT $" + str(len(params) + 1)
            params.append(limit)

            rows = await conn.fetch(query, *params)
            return [(row["object_id"], row["similarity"]) for row in rows]
        finally:
            await self.release_connection(conn)

    # === Promotion Scores ===

    async def get_promotion_score(
        self,
        object_id: str,
        object_type: str,
    ) -> Optional[PromotionScoreRecord]:
        """Get promotion score for an object."""
        if self.pool is None:
            return None

        conn = await self.get_connection()
        try:
            row = await conn.fetchrow(
                """
                SELECT object_id, object_type, score,
                       importance, retrieval_frequency, source_diversity,
                       confidence, novelty, graph_centrality, user_relevance,
                       should_promote, computed_at
                FROM promotion_scores
                WHERE object_id = $1 AND object_type = $2
                """,
                object_id,
                object_type,
            )
            if row:
                return PromotionScoreRecord(**dict(row))
            return None
        finally:
            await self.release_connection(conn)

    async def upsert_promotion_score(
        self,
        record: PromotionScoreRecord,
    ) -> None:
        """Insert or update a promotion score."""
        if self.pool is None:
            return

        conn = await self.get_connection()
        try:
            await conn.execute(
                """
                INSERT INTO promotion_scores (
                    object_id, object_type, score,
                    importance, retrieval_frequency, source_diversity,
                    confidence, novelty, graph_centrality, user_relevance,
                    should_promote, computed_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                ON CONFLICT (object_id, object_type) DO UPDATE SET
                    score = EXCLUDED.score,
                    importance = EXCLUDED.importance,
                    retrieval_frequency = EXCLUDED.retrieval_frequency,
                    source_diversity = EXCLUDED.source_diversity,
                    confidence = EXCLUDED.confidence,
                    novelty = EXCLUDED.novelty,
                    graph_centrality = EXCLUDED.graph_centrality,
                    user_relevance = EXCLUDED.user_relevance,
                    should_promote = EXCLUDED.should_promote,
                    computed_at = EXCLUDED.computed_at
                """,
                record.object_id,
                record.object_type,
                record.score,
                record.importance,
                record.retrieval_frequency,
                record.source_diversity,
                record.confidence,
                record.novelty,
                record.graph_centrality,
                record.user_relevance,
                record.should_promote,
                record.computed_at,
            )
        finally:
            await self.release_connection(conn)

    async def get_objects_for_promotion(
        self,
        object_type: str,
        limit: int = 100,
    ) -> List[str]:
        """Get object IDs that should be promoted."""
        if self.pool is None:
            return []

        conn = await self.get_connection()
        try:
            rows = await conn.fetch(
                """
                SELECT object_id FROM promotion_scores
                WHERE object_type = $1 AND should_promote = true
                ORDER BY score DESC
                LIMIT $2
                """,
                object_type,
                limit,
            )
            return [row["object_id"] for row in rows]
        finally:
            await self.release_connection(conn)

    # === Canonical Claims ===

    async def get_canonical_claim(
        self,
        claim_id: str,
    ) -> Optional[CanonicalClaimRecord]:
        """Get canonical form for a claim."""
        if self.pool is None:
            return None

        conn = await self.get_connection()
        try:
            row = await conn.fetchrow(
                """
                SELECT id, claim_id, subject, predicate, object,
                       modality, temporal_scope, qualifiers,
                       canonical_text, created_at
                FROM canonical_claims
                WHERE claim_id = $1
                """,
                claim_id,
            )
            if row:
                data = dict(row)
                if isinstance(data["qualifiers"], str):
                    data["qualifiers"] = json.loads(data["qualifiers"])
                return CanonicalClaimRecord(**data)
            return None
        finally:
            await self.release_connection(conn)

    async def upsert_canonical_claim(
        self,
        record: CanonicalClaimRecord,
    ) -> str:
        """Insert or update a canonical claim."""
        if self.pool is None:
            return ""

        conn = await self.get_connection()
        try:
            qualifiers_json = json.dumps(record.qualifiers)
            result = await conn.fetchval(
                """
                INSERT INTO canonical_claims (
                    claim_id, subject, predicate, object,
                    modality, temporal_scope, qualifiers,
                    canonical_text, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
                ON CONFLICT (claim_id) DO UPDATE SET
                    subject = EXCLUDED.subject,
                    predicate = EXCLUDED.predicate,
                    object = EXCLUDED.object,
                    modality = EXCLUDED.modality,
                    temporal_scope = EXCLUDED.temporal_scope,
                    qualifiers = EXCLUDED.qualifiers,
                    canonical_text = EXCLUDED.canonical_text,
                    updated_at = NOW()
                RETURNING id
                """,
                record.claim_id,
                record.subject,
                record.predicate,
                record.object,
                record.modality,
                record.temporal_scope,
                qualifiers_json,
                record.canonical_text,
            )
            return result
        finally:
            await self.release_connection(conn)

    # === Temporal Features ===

    async def get_temporal_features(
        self,
        object_id: str,
    ) -> Optional[TemporalFeaturesRecord]:
        """Get temporal features for an object."""
        if self.pool is None:
            return None

        conn = await self.get_connection()
        try:
            row = await conn.fetchrow(
                """
                SELECT id, object_id, object_type, features,
                       timestamp, created_at
                FROM temporal_features
                WHERE object_id = $1
                """,
                object_id,
            )
            if row:
                return TemporalFeaturesRecord(**dict(row))
            return None
        finally:
            await self.release_connection(conn)

    async def upsert_temporal_features(
        self,
        object_id: str,
        object_type: str,
        features: List[float],
        timestamp: Optional[datetime] = None,
    ) -> str:
        """Insert or update temporal features."""
        if self.pool is None:
            return ""

        conn = await self.get_connection()
        try:
            result = await conn.fetchval(
                """
                INSERT INTO temporal_features (
                    object_id, object_type, features, timestamp, created_at
                ) VALUES ($1, $2, $3, $4, NOW())
                ON CONFLICT (object_id) DO UPDATE SET
                    features = EXCLUDED.features,
                    timestamp = EXCLUDED.timestamp,
                    updated_at = NOW()
                RETURNING id
                """,
                object_id,
                object_type,
                features,
                timestamp,
            )
            return result
        finally:
            await self.release_connection(conn)

    # === Query Logging ===

    async def log_query(
        self,
        query: str,
        mode: str,
        tenant_id: str,
        results: List[str],
        latency_ms: float,
        cached: bool = False,
    ) -> None:
        """Log a query for analytics."""
        if self.pool is None:
            return

        conn = await self.get_connection()
        try:
            await conn.execute(
                """
                INSERT INTO query_logs (
                    query, mode, tenant_id, results,
                    latency_ms, cached, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, NOW())
                """,
                query[:500],
                mode,
                tenant_id,
                json.dumps([{"id": r} for r in results[:50]]),
                latency_ms,
                cached,
            )
        except Exception as e:
            logger.warning(f"Failed to log query: {e}")
        finally:
            await self.release_connection(conn)

    # === Shadow Comparisons ===

    async def log_shadow_comparison(
        self,
        query: str,
        tenant_id: str,
        result: str,
        overlap_ratio: float,
        new_latency_ms: float,
        old_latency_ms: float,
    ) -> None:
        """Log a shadow mode comparison."""
        if self.pool is None:
            return

        conn = await self.get_connection()
        try:
            await conn.execute(
                """
                INSERT INTO shadow_comparisons (
                    query, tenant_id, result, overlap_ratio,
                    new_latency_ms, old_latency_ms, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, NOW())
                """,
                query[:500],
                tenant_id,
                result,
                overlap_ratio,
                new_latency_ms,
                old_latency_ms,
            )
        except Exception as e:
            logger.warning(f"Failed to log shadow comparison: {e}")
        finally:
            await self.release_connection(conn)
