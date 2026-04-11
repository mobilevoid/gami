"""
Subconscious Daemon - Proactive context management for AI agents.

Runs as a separate process alongside MCP/API servers.
Watches conversations, predicts context needs, manages hot cache.

Features:
- State classification from message patterns
- Predictive retrieval based on conversation flow
- Hot context caching in Redis
- Proactive context injection
- Session tracking and statistics
"""

import asyncio
import json
import logging
import signal
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

import redis.asyncio as redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from manifold.config_v2 import ManifoldConfigV2, SubconsciousConfig, get_config
from .state_classifier import StateClassifier, ConversationState
from .predictive_retriever import PredictiveRetriever, PredictionResult
from .context_injector import ContextInjector, InjectionContext

logger = logging.getLogger("gami.daemon.subconscious")


@dataclass
class SessionState:
    """Tracked state for a session."""
    session_id: str
    tenant_id: str
    agent_id: Optional[str]
    current_state: ConversationState
    state_confidence: float
    active_entities: List[str]
    active_topics: List[str]
    hot_segment_ids: List[str]
    message_count: int
    last_activity: datetime


@dataclass
class DaemonStats:
    """Daemon runtime statistics."""
    started_at: datetime
    events_processed: int
    state_changes: int
    predictions_made: int
    injections_attempted: int
    injections_accepted: int
    active_sessions: int
    cache_hits: int
    cache_misses: int


class SubconsciousDaemon:
    """Main daemon process for proactive context management."""

    def __init__(self, config: Optional[ManifoldConfigV2] = None):
        """Initialize the daemon.

        Args:
            config: Manifold configuration
        """
        self.config = config or get_config()
        self.sub_config: SubconsciousConfig = self.config.subconscious

        # Components
        self.state_classifier = StateClassifier(
            patterns=self.sub_config.state_patterns,
        )
        self.predictive_retriever = PredictiveRetriever(
            max_predictions=self.sub_config.hot_context_size,
        )
        self.context_injector = ContextInjector(
            max_tokens=self.sub_config.injection_max_tokens,
            injection_threshold=self.sub_config.injection_threshold,
        )

        # Session state
        self.sessions: Dict[str, SessionState] = {}

        # Statistics
        self.stats = DaemonStats(
            started_at=datetime.now(),
            events_processed=0,
            state_changes=0,
            predictions_made=0,
            injections_attempted=0,
            injections_accepted=0,
            active_sessions=0,
            cache_hits=0,
            cache_misses=0,
        )

        # Redis connections
        self.redis: Optional[redis.Redis] = None
        self.pubsub: Optional[redis.client.PubSub] = None

        # Database engine
        self._engine = None
        self._session_factory = None

        # Control
        self._running = False
        self._shutdown_event = asyncio.Event()

    async def start(self) -> None:
        """Start the daemon."""
        if not self.sub_config.enabled:
            logger.info("Subconscious daemon disabled in config")
            return

        logger.info("Starting subconscious daemon...")

        # Initialize database connection
        db_url = self.config.database_url.replace("postgresql://", "postgresql+asyncpg://")
        self._engine = create_async_engine(db_url, pool_size=5, max_overflow=10)
        self._session_factory = sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False
        )

        # Connect to Redis
        self.redis = redis.from_url(
            self.config.redis_url,
            decode_responses=True,
        )
        self.pubsub = self.redis.pubsub()

        # Subscribe to events
        await self.pubsub.subscribe("gami:events")
        logger.info("Subscribed to gami:events channel")

        self._running = True

        # Setup signal handlers
        for sig in (signal.SIGINT, signal.SIGTERM):
            asyncio.get_event_loop().add_signal_handler(
                sig, lambda: asyncio.create_task(self.stop())
            )

        # Start main loops
        try:
            await asyncio.gather(
                self._event_loop(),
                self._maintenance_loop(),
                self._stats_loop(),
            )
        except asyncio.CancelledError:
            logger.info("Daemon tasks cancelled")

    async def stop(self) -> None:
        """Stop the daemon gracefully."""
        logger.info("Stopping subconscious daemon...")
        self._running = False
        self._shutdown_event.set()

        if self.pubsub:
            await self.pubsub.unsubscribe()
            await self.pubsub.close()

        if self.redis:
            await self.redis.close()

        if self._engine:
            await self._engine.dispose()

        logger.info("Subconscious daemon stopped")

    async def _event_loop(self) -> None:
        """Main event processing loop."""
        poll_interval = self.sub_config.poll_interval_ms / 1000

        while self._running:
            try:
                message = await self.pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=poll_interval,
                )

                if message and message["type"] == "message":
                    try:
                        event = json.loads(message["data"])
                        await self._handle_event(event)
                        self.stats.events_processed += 1
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid event JSON: {e}")

            except redis.ConnectionError:
                logger.error("Redis connection lost, reconnecting...")
                await asyncio.sleep(5)
                try:
                    self.redis = redis.from_url(self.config.redis_url, decode_responses=True)
                    self.pubsub = self.redis.pubsub()
                    await self.pubsub.subscribe("gami:events")
                except Exception as e:
                    logger.error(f"Reconnection failed: {e}")

            except Exception as e:
                logger.error(f"Event loop error: {e}")
                await asyncio.sleep(1)

    async def _handle_event(self, event: Dict[str, Any]) -> None:
        """Handle incoming event."""
        event_type = event.get("type")
        session_id = event.get("session_id")

        if not session_id:
            return

        tenant_id = event.get("tenant_id", "*")
        agent_id = event.get("agent_id")

        if event_type == "message":
            await self._on_message(session_id, tenant_id, agent_id, event)
        elif event_type == "query":
            await self._on_query(session_id, tenant_id, agent_id, event)
        elif event_type == "session_start":
            await self._on_session_start(session_id, tenant_id, agent_id, event)
        elif event_type == "session_end":
            await self._on_session_end(session_id)
        elif event_type == "feedback":
            await self._on_feedback(session_id, event)

    async def _on_message(
        self,
        session_id: str,
        tenant_id: str,
        agent_id: Optional[str],
        event: Dict[str, Any],
    ) -> None:
        """Handle new message in conversation."""
        message = event.get("message", "")
        message_type = event.get("message_type", "user")

        # Get or create session state
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionState(
                session_id=session_id,
                tenant_id=tenant_id,
                agent_id=agent_id,
                current_state=ConversationState.IDLE,
                state_confidence=0.3,
                active_entities=[],
                active_topics=[],
                hot_segment_ids=[],
                message_count=0,
                last_activity=datetime.now(),
            )

        session = self.sessions[session_id]
        session.message_count += 1
        session.last_activity = datetime.now()

        # Only classify on user messages
        if message_type == "user" and message:
            # Classify state
            classification = await self.state_classifier.classify([message])

            # Check for state change
            if classification.state != session.current_state:
                await self._log_event(session_id, "state_change", {
                    "previous_state": session.current_state.value,
                    "new_state": classification.state.value,
                    "confidence": classification.confidence,
                })
                self.stats.state_changes += 1

            session.current_state = classification.state
            session.state_confidence = classification.confidence

            # Update active entities from event
            if event.get("entities"):
                session.active_entities = event["entities"][-20:]

            # Predictive retrieval if enabled
            if self.sub_config.predictive_retrieval_enabled:
                await self._predict_and_cache(session)

    async def _on_query(
        self,
        session_id: str,
        tenant_id: str,
        agent_id: Optional[str],
        event: Dict[str, Any],
    ) -> None:
        """Handle retrieval query event."""
        query = event.get("query", "")
        results = event.get("results", [])

        # Update session context
        self.predictive_retriever.update_session_context(
            session_id,
            query=query,
            segments=[r.get("segment_id") for r in results],
        )

    async def _on_session_start(
        self,
        session_id: str,
        tenant_id: str,
        agent_id: Optional[str],
        event: Dict[str, Any],
    ) -> None:
        """Handle session start."""
        self.sessions[session_id] = SessionState(
            session_id=session_id,
            tenant_id=tenant_id,
            agent_id=agent_id,
            current_state=ConversationState.IDLE,
            state_confidence=0.3,
            active_entities=[],
            active_topics=[],
            hot_segment_ids=[],
            message_count=0,
            last_activity=datetime.now(),
        )

        self.stats.active_sessions = len(self.sessions)

        # Record in database
        async with self._session_factory() as db:
            await db.execute(text("""
                INSERT INTO sessions (session_id, tenant_id, agent_id, conversation_state)
                VALUES (:session_id, :tenant_id, :agent_id, 'idle')
                ON CONFLICT (session_id) DO UPDATE SET
                    last_activity_at = NOW(),
                    ended_at = NULL
            """), {
                "session_id": session_id,
                "tenant_id": tenant_id,
                "agent_id": agent_id,
            })
            await db.commit()

    async def _on_session_end(self, session_id: str) -> None:
        """Handle session end."""
        if session_id in self.sessions:
            del self.sessions[session_id]

        self.predictive_retriever.clear_session(session_id)
        self.context_injector.clear_session(session_id)

        # Clear Redis cache
        await self.redis.delete(f"gami:session:{session_id}:hot")

        # Update database
        async with self._session_factory() as db:
            await db.execute(text("""
                UPDATE sessions SET ended_at = NOW()
                WHERE session_id = :session_id
            """), {"session_id": session_id})
            await db.commit()

        self.stats.active_sessions = len(self.sessions)

    async def _on_feedback(self, session_id: str, event: Dict[str, Any]) -> None:
        """Handle feedback event."""
        feedback_type = event.get("feedback_type")
        segment_ids = event.get("segment_ids", [])

        # Update predictive retriever context
        if feedback_type in ("confirmed", "helpful", "used"):
            self.predictive_retriever.update_session_context(
                session_id,
                segments=segment_ids,
            )

    async def _predict_and_cache(self, session: SessionState) -> None:
        """Run predictive retrieval and cache results."""
        async with self._session_factory() as db:
            prediction = await self.predictive_retriever.predict_needs(
                db=db,
                session_id=session.session_id,
                current_state=session.current_state,
                tenant_id=session.tenant_id,
            )

            if not prediction.segment_ids:
                return

            self.stats.predictions_made += 1

            # Load segment content
            segments = await self._load_segments(db, prediction.segment_ids)

            if segments:
                session.hot_segment_ids = prediction.segment_ids

                # Cache in Redis
                await self._cache_hot_context(session.session_id, segments)

                await self._log_event(session.session_id, "preload", {
                    "segment_count": len(segments),
                    "prediction_source": prediction.prediction_source,
                    "confidence": prediction.confidence,
                })

    async def _load_segments(
        self,
        db: AsyncSession,
        segment_ids: List[str],
    ) -> List[Dict[str, Any]]:
        """Load segment content from database."""
        if not segment_ids:
            return []

        result = await db.execute(text("""
            SELECT
                s.segment_id, s.raw_text, s.segment_type, s.importance_score,
                src.source_type
            FROM segments s
            LEFT JOIN sources src ON s.source_id = src.source_id
            WHERE s.segment_id = ANY(:ids)
            AND s.status = 'active'
        """), {"ids": segment_ids})

        return [
            {
                "segment_id": row.segment_id,
                "text": row.raw_text,
                "segment_type": row.segment_type,
                "score": row.importance_score or 0.5,
                "source_type": row.source_type,
            }
            for row in result.fetchall()
        ]

    async def _cache_hot_context(
        self,
        session_id: str,
        segments: List[Dict[str, Any]],
    ) -> None:
        """Cache hot context in Redis."""
        cache_key = f"gami:session:{session_id}:hot"
        await self.redis.setex(
            cache_key,
            self.sub_config.hot_context_ttl_seconds,
            json.dumps(segments),
        )

    async def get_hot_context(self, session_id: str) -> List[Dict[str, Any]]:
        """Get cached hot context for a session."""
        cache_key = f"gami:session:{session_id}:hot"
        data = await self.redis.get(cache_key)

        if data:
            self.stats.cache_hits += 1
            return json.loads(data)

        self.stats.cache_misses += 1
        return []

    async def prepare_injection(
        self,
        session_id: str,
    ) -> Optional[InjectionContext]:
        """Prepare context injection for a session."""
        if not self.sub_config.injection_enabled:
            return None

        session = self.sessions.get(session_id)
        if not session:
            return None

        hot_context = await self.get_hot_context(session_id)
        if not hot_context:
            return None

        self.stats.injections_attempted += 1

        injection = await self.context_injector.prepare_injection(
            session_id=session_id,
            hot_context=hot_context,
            current_state=session.current_state,
            agent_id=session.agent_id,
        )

        if injection:
            self.stats.injections_accepted += 1

        return injection

    async def _log_event(
        self,
        session_id: str,
        event_type: str,
        data: Dict[str, Any],
    ) -> None:
        """Log subconscious event to database."""
        try:
            async with self._session_factory() as db:
                await db.execute(text("""
                    INSERT INTO subconscious_events (
                        session_id, event_type,
                        detected_state, state_confidence,
                        predicted_entities, preloaded_segment_ids,
                        prediction_confidence
                    ) VALUES (
                        :session_id, :event_type,
                        :state, :state_conf,
                        :entities, :segments,
                        :pred_conf
                    )
                """), {
                    "session_id": session_id,
                    "event_type": event_type,
                    "state": data.get("new_state") or data.get("detected_state"),
                    "state_conf": data.get("confidence") or data.get("state_confidence"),
                    "entities": data.get("predicted_entities"),
                    "segments": data.get("preloaded_segment_ids"),
                    "pred_conf": data.get("prediction_confidence"),
                })
                await db.commit()
        except Exception as e:
            logger.debug(f"Failed to log event: {e}")

    async def _maintenance_loop(self) -> None:
        """Background maintenance tasks."""
        while self._running:
            try:
                # Clean up stale sessions
                stale_threshold = datetime.now() - timedelta(hours=1)
                stale_sessions = [
                    sid for sid, session in self.sessions.items()
                    if session.last_activity < stale_threshold
                ]

                for sid in stale_sessions:
                    await self._on_session_end(sid)

                if stale_sessions:
                    logger.info(f"Cleaned up {len(stale_sessions)} stale sessions")

                await asyncio.sleep(60)

            except Exception as e:
                logger.error(f"Maintenance loop error: {e}")
                await asyncio.sleep(60)

    async def _stats_loop(self) -> None:
        """Periodic statistics logging."""
        while self._running:
            try:
                await asyncio.sleep(300)  # Every 5 minutes

                logger.info(
                    f"Stats: events={self.stats.events_processed}, "
                    f"sessions={self.stats.active_sessions}, "
                    f"predictions={self.stats.predictions_made}, "
                    f"injections={self.stats.injections_accepted}/{self.stats.injections_attempted}, "
                    f"cache_hits={self.stats.cache_hits}"
                )

            except Exception as e:
                logger.debug(f"Stats loop error: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get current daemon statistics."""
        return {
            **asdict(self.stats),
            "started_at": self.stats.started_at.isoformat(),
            "uptime_seconds": (datetime.now() - self.stats.started_at).total_seconds(),
        }


# =============================================================================
# ENTRY POINT
# =============================================================================

async def main():
    """Main entry point for the daemon."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    config = get_config()
    daemon = SubconsciousDaemon(config)

    try:
        await daemon.start()
    except KeyboardInterrupt:
        await daemon.stop()


if __name__ == "__main__":
    asyncio.run(main())
