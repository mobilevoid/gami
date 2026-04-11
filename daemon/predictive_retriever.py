"""
Predictive retrieval for proactive context management.

Predicts what context will be needed based on:
1. Current conversation state
2. Active entities
3. Recent query patterns
4. Session history
"""

import logging
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from .state_classifier import ConversationState

logger = logging.getLogger("gami.daemon.predictive_retriever")


@dataclass
class PredictionResult:
    """Result of predictive retrieval."""
    segment_ids: List[str]
    entity_ids: List[str]
    topics: List[str]
    confidence: float
    prediction_source: str  # "state", "entity", "query_pattern", "graph"


@dataclass
class SessionContext:
    """Tracked context for a session."""
    session_id: str
    active_entities: List[str] = field(default_factory=list)
    active_topics: List[str] = field(default_factory=list)
    recent_queries: List[str] = field(default_factory=list)
    recent_segments: List[str] = field(default_factory=list)
    state_history: List[ConversationState] = field(default_factory=list)


class PredictiveRetriever:
    """Service for predictive context retrieval."""

    def __init__(
        self,
        max_predictions: int = 20,
        min_confidence: float = 0.4,
    ):
        """Initialize predictive retriever.

        Args:
            max_predictions: Maximum segments to predict
            min_confidence: Minimum confidence threshold
        """
        self.max_predictions = max_predictions
        self.min_confidence = min_confidence

        # In-memory session contexts
        self._sessions: Dict[str, SessionContext] = {}

    def get_session_context(self, session_id: str) -> SessionContext:
        """Get or create session context."""
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionContext(session_id=session_id)
        return self._sessions[session_id]

    def update_session_context(
        self,
        session_id: str,
        entities: Optional[List[str]] = None,
        topics: Optional[List[str]] = None,
        query: Optional[str] = None,
        segments: Optional[List[str]] = None,
        state: Optional[ConversationState] = None,
    ) -> None:
        """Update session context with new information."""
        ctx = self.get_session_context(session_id)

        if entities:
            # Add new entities, maintain recency (most recent last)
            for ent in entities:
                if ent in ctx.active_entities:
                    ctx.active_entities.remove(ent)
                ctx.active_entities.append(ent)
            # Keep last 20
            ctx.active_entities = ctx.active_entities[-20:]

        if topics:
            for topic in topics:
                if topic in ctx.active_topics:
                    ctx.active_topics.remove(topic)
                ctx.active_topics.append(topic)
            ctx.active_topics = ctx.active_topics[-10:]

        if query:
            ctx.recent_queries.append(query)
            ctx.recent_queries = ctx.recent_queries[-10:]

        if segments:
            ctx.recent_segments.extend(segments)
            ctx.recent_segments = ctx.recent_segments[-50:]

        if state:
            ctx.state_history.append(state)
            ctx.state_history = ctx.state_history[-10:]

    async def predict_needs(
        self,
        db: AsyncSession,
        session_id: str,
        current_state: ConversationState,
        tenant_id: str,
    ) -> PredictionResult:
        """Predict what context will be needed.

        Args:
            db: Database session
            session_id: Session ID
            current_state: Current conversation state
            tenant_id: Tenant ID

        Returns:
            PredictionResult with predicted segments
        """
        ctx = self.get_session_context(session_id)
        predictions: List[PredictionResult] = []

        # Strategy 1: State-based prediction
        state_pred = await self._predict_by_state(db, current_state, ctx, tenant_id)
        if state_pred:
            predictions.append(state_pred)

        # Strategy 2: Entity-based prediction
        if ctx.active_entities:
            entity_pred = await self._predict_by_entities(db, ctx.active_entities, tenant_id)
            if entity_pred:
                predictions.append(entity_pred)

        # Strategy 3: Query pattern prediction
        if ctx.recent_queries:
            query_pred = await self._predict_by_queries(db, ctx.recent_queries, tenant_id)
            if query_pred:
                predictions.append(query_pred)

        # Strategy 4: Graph expansion
        if ctx.recent_segments:
            graph_pred = await self._predict_by_graph(db, ctx.recent_segments[-5:], tenant_id)
            if graph_pred:
                predictions.append(graph_pred)

        # Merge predictions
        return self._merge_predictions(predictions)

    async def _predict_by_state(
        self,
        db: AsyncSession,
        state: ConversationState,
        ctx: SessionContext,
        tenant_id: str,
    ) -> Optional[PredictionResult]:
        """Predict based on conversation state using manifold-aware retrieval."""
        # Map states to query modes and primary manifolds
        state_config = {
            ConversationState.DEBUGGING: {
                "query_mode": "diagnostic",
                "primary_manifold": "PROCEDURE",
                "secondary_manifold": "TIME",
            },
            ConversationState.PLANNING: {
                "query_mode": "how_to",
                "primary_manifold": "PROCEDURE",
                "secondary_manifold": "TOPIC",
            },
            ConversationState.RECALLING: {
                "query_mode": "recall",
                "primary_manifold": "TOPIC",
                "secondary_manifold": "TIME",
            },
            ConversationState.EXPLORING: {
                "query_mode": "exploration",
                "primary_manifold": "TOPIC",
                "secondary_manifold": "CLAIM",
            },
            ConversationState.EXECUTING: {
                "query_mode": "procedural",
                "primary_manifold": "PROCEDURE",
                "secondary_manifold": "TOPIC",
            },
            ConversationState.CONFIRMING: {
                "query_mode": "verification",
                "primary_manifold": "CLAIM",
                "secondary_manifold": "EVIDENCE",
            },
        }

        config = state_config.get(state)
        if not config:
            return None

        query_mode = config["query_mode"]
        primary_manifold = config["primary_manifold"]

        # Try manifold-based prediction first (if manifold embeddings exist)
        try:
            manifold_result = await db.execute(text("""
                SELECT DISTINCT me.target_id as segment_id
                FROM manifold_embeddings me
                JOIN segments s ON me.target_id = s.segment_id
                WHERE me.manifold_type = :manifold
                AND me.target_type = 'segment'
                AND s.owner_tenant_id = :tenant_id
                AND s.importance_score > 0.5
                ORDER BY s.importance_score DESC
                LIMIT :limit
            """), {
                "manifold": primary_manifold,
                "tenant_id": tenant_id,
                "limit": self.max_predictions // 3,
            })

            manifold_segments = [row.segment_id for row in manifold_result.fetchall()]
        except Exception as e:
            logger.debug(f"Manifold prediction not available: {e}")
            manifold_segments = []

        # Fall back to/augment with retrieval log based prediction
        result = await db.execute(text("""
            SELECT DISTINCT unnest(segments_returned) as segment_id
            FROM retrieval_logs
            WHERE query_mode = :mode
            AND tenant_id = :tenant_id
            AND outcome_signal > 0
            ORDER BY segment_id
            LIMIT :limit
        """), {
            "mode": query_mode,
            "tenant_id": tenant_id,
            "limit": self.max_predictions // 2,
        })

        log_segments = [row.segment_id for row in result.fetchall()]

        # Merge results, prioritizing manifold-based segments
        all_segments = manifold_segments + [s for s in log_segments if s not in manifold_segments]
        segment_ids = all_segments[:self.max_predictions // 2]

        if segment_ids:
            confidence = 0.7 if manifold_segments else 0.6
            return PredictionResult(
                segment_ids=segment_ids,
                entity_ids=[],
                topics=[],
                confidence=confidence,
                prediction_source=f"state:{primary_manifold}",
            )

        return None

    async def _predict_by_entities(
        self,
        db: AsyncSession,
        entity_ids: List[str],
        tenant_id: str,
    ) -> Optional[PredictionResult]:
        """Predict based on active entities."""
        # Find segments mentioning these entities via provenance table
        result = await db.execute(text("""
            SELECT DISTINCT s.segment_id
            FROM segments s
            JOIN provenance p ON s.segment_id = p.segment_id
            WHERE p.target_type = 'entity'
            AND p.target_id = ANY(:entities)
            AND s.owner_tenant_id = :tenant_id
            AND s.storage_tier != 'cold'
            ORDER BY s.importance_score DESC NULLS LAST
            LIMIT :limit
        """), {
            "entities": entity_ids[-5:],  # Focus on most recent entities
            "tenant_id": tenant_id,
            "limit": self.max_predictions // 2,
        })

        segment_ids = [row.segment_id for row in result.fetchall()]

        if segment_ids:
            return PredictionResult(
                segment_ids=segment_ids,
                entity_ids=entity_ids[-5:],
                topics=[],
                confidence=0.7,
                prediction_source="entity",
            )

        return None

    async def _predict_by_queries(
        self,
        db: AsyncSession,
        recent_queries: List[str],
        tenant_id: str,
    ) -> Optional[PredictionResult]:
        """Predict based on recent query patterns."""
        # Find segments that were retrieved for similar queries
        # Use the most recent successful retrievals
        result = await db.execute(text("""
            SELECT segments_returned
            FROM retrieval_logs
            WHERE tenant_id = :tenant_id
            AND outcome_signal > 0
            AND processed_in_dream = FALSE
            ORDER BY created_at DESC
            LIMIT 5
        """), {"tenant_id": tenant_id})

        all_segments = set()
        for row in result.fetchall():
            if row.segments_returned:
                all_segments.update(row.segments_returned[:5])

        if all_segments:
            return PredictionResult(
                segment_ids=list(all_segments)[:self.max_predictions // 2],
                entity_ids=[],
                topics=[],
                confidence=0.5,
                prediction_source="query_pattern",
            )

        return None

    async def _predict_by_graph(
        self,
        db: AsyncSession,
        recent_segments: List[str],
        tenant_id: str,
    ) -> Optional[PredictionResult]:
        """Predict by expanding from recent segments via graph."""
        # Find entities in recent segments via provenance table
        result = await db.execute(text("""
            SELECT DISTINCT p.target_id as entity_id
            FROM provenance p
            WHERE p.segment_id = ANY(:segments)
            AND p.target_type = 'entity'
            LIMIT 10
        """), {"segments": recent_segments})

        entities = [row.entity_id for row in result.fetchall()]

        if not entities:
            return None

        # Expand via graph relations
        result = await db.execute(text("""
            SELECT DISTINCT
                CASE
                    WHEN r.source_entity_id = ANY(:entities) THEN r.target_entity_id
                    ELSE r.source_entity_id
                END as related_entity
            FROM relations r
            WHERE (r.source_entity_id = ANY(:entities) OR r.target_entity_id = ANY(:entities))
            AND r.owner_tenant_id = :tenant_id
            AND r.status = 'active'
            LIMIT 20
        """), {"entities": entities, "tenant_id": tenant_id})

        related_entities = [row.related_entity for row in result.fetchall()]

        if related_entities:
            # Get segments for related entities via provenance
            seg_result = await db.execute(text("""
                SELECT DISTINCT s.segment_id
                FROM segments s
                JOIN provenance p ON s.segment_id = p.segment_id
                WHERE p.target_type = 'entity'
                AND p.target_id = ANY(:entities)
                AND s.owner_tenant_id = :tenant_id
                AND s.storage_tier != 'cold'
                AND s.segment_id != ALL(:exclude)
                ORDER BY s.importance_score DESC NULLS LAST
                LIMIT :limit
            """), {
                "entities": related_entities,
                "tenant_id": tenant_id,
                "exclude": recent_segments,
                "limit": self.max_predictions // 3,
            })

            segment_ids = [row.segment_id for row in seg_result.fetchall()]

            if segment_ids:
                return PredictionResult(
                    segment_ids=segment_ids,
                    entity_ids=related_entities,
                    topics=[],
                    confidence=0.55,
                    prediction_source="graph",
                )

        return None

    def _merge_predictions(
        self,
        predictions: List[PredictionResult],
    ) -> PredictionResult:
        """Merge multiple prediction results."""
        if not predictions:
            return PredictionResult(
                segment_ids=[],
                entity_ids=[],
                topics=[],
                confidence=0.0,
                prediction_source="none",
            )

        if len(predictions) == 1:
            return predictions[0]

        # Merge with weighting by confidence
        segment_scores: Dict[str, float] = defaultdict(float)
        all_entities: Set[str] = set()
        all_topics: Set[str] = set()

        for pred in predictions:
            for seg_id in pred.segment_ids:
                segment_scores[seg_id] += pred.confidence
            all_entities.update(pred.entity_ids)
            all_topics.update(pred.topics)

        # Sort by score
        sorted_segments = sorted(
            segment_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        merged_segments = [s[0] for s in sorted_segments[:self.max_predictions]]

        # Average confidence
        avg_confidence = sum(p.confidence for p in predictions) / len(predictions)

        return PredictionResult(
            segment_ids=merged_segments,
            entity_ids=list(all_entities)[:20],
            topics=list(all_topics)[:10],
            confidence=min(1.0, avg_confidence),
            prediction_source="merged",
        )

    def clear_session(self, session_id: str) -> None:
        """Clear session context."""
        if session_id in self._sessions:
            del self._sessions[session_id]
