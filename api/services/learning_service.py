"""
Retrieval learning service.

Implements learning from user feedback on retrieval results:
1. Logs all retrievals with context
2. Collects outcome signals (confirmed, corrected, ignored, etc.)
3. Adjusts segment importance scores using bandit-style updates
4. Integrates with dream cycle for batch processing
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import math

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from manifold.config_v2 import ManifoldConfigV2, LearningConfig, get_config

logger = logging.getLogger("gami.learning_service")


# =============================================================================
# OUTCOME SIGNAL DEFINITIONS
# =============================================================================

OUTCOME_SIGNALS = {
    # Positive signals
    "confirmed": 1.0,        # User explicitly confirmed result is correct
    "used": 0.8,             # Result was cited/used in response
    "helpful": 0.7,          # User indicated result was helpful
    "continued": 0.3,        # User continued conversation without correction
    "acknowledged": 0.2,     # User acknowledged seeing result

    # Negative signals
    "ignored": -0.2,         # Result was ignored/not used
    "unhelpful": -0.5,       # User indicated not helpful
    "corrected": -0.8,       # User provided correction
    "wrong": -1.0,           # User explicitly said result was wrong

    # Neutral
    "ambiguous": 0.0,        # Unclear if result was helpful
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class RetrievalLog:
    """Log entry for a retrieval operation."""
    log_id: str
    session_id: Optional[str]
    query_text: str
    query_mode: Optional[str]
    segments_returned: List[str]
    scores_returned: List[float]
    tenant_id: str
    agent_id: Optional[str]
    user_id: Optional[str]
    created_at: datetime
    outcome_type: Optional[str] = None
    outcome_signal: Optional[float] = None
    correction_text: Optional[str] = None


@dataclass
class LearningStats:
    """Statistics from learning process."""
    logs_processed: int
    segments_updated: int
    positive_signals: int
    negative_signals: int
    adjustments_made: Dict[str, float]  # segment_id -> adjustment


# =============================================================================
# RETRIEVAL LOGGER
# =============================================================================

class RetrievalLogger:
    """Service for logging retrieval operations."""

    async def log_retrieval(
        self,
        db: AsyncSession,
        query_text: str,
        segments_returned: List[str],
        scores_returned: List[float],
        tenant_id: str,
        session_id: Optional[str] = None,
        query_mode: Optional[str] = None,
        agent_id: Optional[str] = None,
        user_id: Optional[str] = None,
        query_embedding: Optional[List[float]] = None,
    ) -> str:
        """Log a retrieval operation.

        Args:
            db: Database session
            query_text: The query text
            segments_returned: List of segment IDs returned
            scores_returned: Corresponding scores
            tenant_id: Tenant ID
            session_id: Session ID for tracking
            query_mode: Query mode used
            agent_id: Agent performing retrieval
            user_id: User performing retrieval
            query_embedding: Optional query embedding for analysis

        Returns:
            Log ID
        """
        import secrets
        log_id = f"RL_{secrets.token_hex(8)}"

        # Format embedding if provided
        embedding_str = None
        if query_embedding:
            embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

        await db.execute(text("""
            INSERT INTO retrieval_logs (
                log_id, session_id, query_text, query_embedding, query_mode,
                segments_returned, scores_returned,
                tenant_id, agent_id, user_id
            ) VALUES (
                :log_id, :session_id, :query, :embedding, :mode,
                :segments, :scores,
                :tenant_id, :agent_id, :user_id
            )
        """), {
            "log_id": log_id,
            "session_id": session_id,
            "query": query_text[:1000],  # Truncate long queries
            "embedding": embedding_str,
            "mode": query_mode,
            "segments": segments_returned,
            "scores": scores_returned,
            "tenant_id": tenant_id,
            "agent_id": agent_id,
            "user_id": user_id,
        })

        await db.commit()
        return log_id

    async def record_outcome(
        self,
        db: AsyncSession,
        log_id: Optional[str] = None,
        session_id: Optional[str] = None,
        outcome_type: str = "ambiguous",
        correction_text: Optional[str] = None,
    ) -> bool:
        """Record outcome for a retrieval.

        Args:
            db: Database session
            log_id: Specific log ID to update
            session_id: Session ID to find most recent log
            outcome_type: Type of outcome (see OUTCOME_SIGNALS)
            correction_text: Optional correction provided by user

        Returns:
            True if outcome was recorded
        """
        outcome_signal = OUTCOME_SIGNALS.get(outcome_type, 0.0)

        if log_id:
            result = await db.execute(text("""
                UPDATE retrieval_logs SET
                    outcome_type = :outcome_type,
                    outcome_signal = :outcome_signal,
                    correction_text = :correction
                WHERE log_id = :log_id AND outcome_type IS NULL
                RETURNING log_id
            """), {
                "log_id": log_id,
                "outcome_type": outcome_type,
                "outcome_signal": outcome_signal,
                "correction": correction_text,
            })
        elif session_id:
            # Update most recent log for session using subquery
            result = await db.execute(text("""
                UPDATE retrieval_logs SET
                    outcome_type = :outcome_type,
                    outcome_signal = :outcome_signal,
                    correction_text = :correction
                WHERE log_id = (
                    SELECT log_id FROM retrieval_logs
                    WHERE session_id = :session_id
                    AND outcome_type IS NULL
                    AND created_at > NOW() - INTERVAL '1 hour'
                    ORDER BY created_at DESC
                    LIMIT 1
                )
                RETURNING log_id
            """), {
                "session_id": session_id,
                "outcome_type": outcome_type,
                "outcome_signal": outcome_signal,
                "correction": correction_text,
            })
        else:
            logger.warning("record_outcome called without log_id or session_id")
            return False

        success = result.fetchone() is not None
        if success:
            await db.commit()
            logger.debug(f"Recorded outcome {outcome_type} for retrieval")

        return success

    async def get_session_logs(
        self,
        db: AsyncSession,
        session_id: str,
        limit: int = 50,
    ) -> List[RetrievalLog]:
        """Get retrieval logs for a session.

        Args:
            db: Database session
            session_id: Session to get logs for
            limit: Maximum logs to return

        Returns:
            List of RetrievalLog objects
        """
        result = await db.execute(text("""
            SELECT
                log_id, session_id, query_text, query_mode,
                segments_returned, scores_returned,
                tenant_id, agent_id, user_id,
                outcome_type, outcome_signal, correction_text,
                created_at
            FROM retrieval_logs
            WHERE session_id = :session_id
            ORDER BY created_at DESC
            LIMIT :limit
        """), {"session_id": session_id, "limit": limit})

        return [
            RetrievalLog(
                log_id=row.log_id,
                session_id=row.session_id,
                query_text=row.query_text,
                query_mode=row.query_mode,
                segments_returned=row.segments_returned or [],
                scores_returned=row.scores_returned or [],
                tenant_id=row.tenant_id,
                agent_id=row.agent_id,
                user_id=row.user_id,
                outcome_type=row.outcome_type,
                outcome_signal=row.outcome_signal,
                correction_text=row.correction_text,
                created_at=row.created_at,
            )
            for row in result.fetchall()
        ]


# =============================================================================
# LEARNING ANALYZER
# =============================================================================

class LearningAnalyzer:
    """Service for analyzing retrieval logs and adjusting scores."""

    def __init__(self, config: Optional[ManifoldConfigV2] = None):
        """Initialize learning analyzer.

        Args:
            config: Manifold configuration
        """
        self._config = config

    @property
    def config(self) -> LearningConfig:
        """Get learning config."""
        if self._config is None:
            self._config = get_config()
        return self._config.learning

    async def process_learning_signals(
        self,
        db: AsyncSession,
        max_logs: int = 1000,
        tenant_id: Optional[str] = None,
    ) -> LearningStats:
        """Process unprocessed retrieval logs and adjust segment scores.

        This is the main learning function, called during dream cycle.

        Args:
            db: Database session
            max_logs: Maximum logs to process
            tenant_id: Filter by tenant

        Returns:
            LearningStats with processing results
        """
        if not self.config.enabled:
            return LearningStats(0, 0, 0, 0, {})

        # Get unprocessed logs with outcomes
        params = {"limit": max_logs}
        tenant_filter = ""
        if tenant_id:
            tenant_filter = "AND tenant_id = :tenant_id"
            params["tenant_id"] = tenant_id

        result = await db.execute(text(f"""
            SELECT log_id, segments_returned, outcome_signal
            FROM retrieval_logs
            WHERE NOT processed_in_dream
            AND outcome_signal IS NOT NULL
            {tenant_filter}
            ORDER BY created_at
            LIMIT :limit
        """), params)

        logs = result.fetchall()

        if not logs:
            logger.info("No new learning signals to process")
            return LearningStats(0, 0, 0, 0, {})

        # Aggregate signals per segment
        segment_signals: Dict[str, List[float]] = defaultdict(list)
        positive_count = 0
        negative_count = 0

        for log_id, segments, signal in logs:
            if not segments:
                continue

            for seg_id in segments:
                segment_signals[seg_id].append(signal)

            if signal > 0:
                positive_count += 1
            elif signal < 0:
                negative_count += 1

        # Calculate and apply adjustments
        adjustments = {}
        segments_updated = 0

        alpha = self.config.bandit_alpha
        max_adj = self.config.max_adjustment
        min_obs = self.config.min_observations
        pos_weight = self.config.positive_signal_weight
        neg_weight = self.config.negative_signal_weight

        for seg_id, signals in segment_signals.items():
            if len(signals) < min_obs:
                continue

            # Weighted average of signals (negatives weighted more heavily)
            pos_signals = [s for s in signals if s > 0]
            neg_signals = [s for s in signals if s < 0]

            pos_contribution = sum(pos_signals) / len(pos_signals) if pos_signals else 0
            neg_contribution = sum(neg_signals) / len(neg_signals) if neg_signals else 0

            weighted_signal = (
                pos_contribution * pos_weight + neg_contribution * neg_weight
            ) / (pos_weight + neg_weight)

            # Calculate adjustment with learning rate
            adjustment = alpha * weighted_signal
            adjustment = max(-max_adj, min(max_adj, adjustment))

            # Apply adjustment
            await db.execute(text("""
                UPDATE segments SET
                    importance_score = GREATEST(0.0, LEAST(1.0,
                        COALESCE(importance_score, 0.5) + :adj
                    )),
                    stability_score = GREATEST(0.0, LEAST(1.0,
                        COALESCE(stability_score, 0.5) + ABS(:adj) * 0.1
                    )),
                    last_reinforced_at = NOW(),
                    updated_at = NOW()
                WHERE segment_id = :sid
            """), {"adj": adjustment, "sid": seg_id})

            adjustments[seg_id] = adjustment
            segments_updated += 1

        # Mark logs as processed
        log_ids = [l[0] for l in logs]
        await db.execute(text("""
            UPDATE retrieval_logs SET processed_in_dream = TRUE
            WHERE log_id = ANY(:ids)
        """), {"ids": log_ids})

        await db.commit()

        stats = LearningStats(
            logs_processed=len(logs),
            segments_updated=segments_updated,
            positive_signals=positive_count,
            negative_signals=negative_count,
            adjustments_made=adjustments,
        )

        logger.info(
            f"Learning: processed {stats.logs_processed} logs, "
            f"updated {stats.segments_updated} segments "
            f"(+{stats.positive_signals}/-{stats.negative_signals})"
        )

        return stats

    async def get_segment_learning_history(
        self,
        db: AsyncSession,
        segment_id: str,
        days: int = 30,
    ) -> Dict[str, Any]:
        """Get learning history for a segment.

        Args:
            db: Database session
            segment_id: Segment to analyze
            days: Days of history

        Returns:
            Learning history and statistics
        """
        result = await db.execute(text("""
            SELECT
                outcome_type,
                outcome_signal,
                query_text,
                created_at
            FROM retrieval_logs
            WHERE :segment_id = ANY(segments_returned)
            AND outcome_signal IS NOT NULL
            AND created_at > NOW() - INTERVAL ':days days'
            ORDER BY created_at DESC
        """.replace(":days", str(days))), {"segment_id": segment_id})

        history = []
        total_positive = 0
        total_negative = 0
        total_neutral = 0

        for row in result.fetchall():
            history.append({
                "outcome_type": row.outcome_type,
                "outcome_signal": row.outcome_signal,
                "query": row.query_text[:100],
                "timestamp": row.created_at.isoformat(),
            })

            if row.outcome_signal > 0:
                total_positive += 1
            elif row.outcome_signal < 0:
                total_negative += 1
            else:
                total_neutral += 1

        # Get current segment stats
        seg_result = await db.execute(text("""
            SELECT importance_score, stability_score, last_reinforced_at
            FROM segments
            WHERE segment_id = :sid
        """), {"sid": segment_id})

        seg_row = seg_result.fetchone()

        return {
            "segment_id": segment_id,
            "total_retrievals": len(history),
            "positive_outcomes": total_positive,
            "negative_outcomes": total_negative,
            "neutral_outcomes": total_neutral,
            "current_importance": seg_row.importance_score if seg_row else None,
            "current_stability": seg_row.stability_score if seg_row else None,
            "last_reinforced": seg_row.last_reinforced_at.isoformat() if seg_row and seg_row.last_reinforced_at else None,
            "history": history[:20],  # Last 20 retrievals
        }


# =============================================================================
# FEEDBACK INFERENCE
# =============================================================================

class FeedbackInference:
    """Infer feedback from user actions when not explicitly provided."""

    def infer_outcome_from_response(
        self,
        user_message: str,
        assistant_response: str,
        retrieved_segments: List[str],
        cited_segments: List[str],
    ) -> Tuple[str, float]:
        """Infer outcome signal from conversation.

        Args:
            user_message: User's message
            assistant_response: Assistant's response
            retrieved_segments: Segments that were retrieved
            cited_segments: Segments that were cited in response

        Returns:
            Tuple of (outcome_type, confidence)
        """
        # Check for explicit feedback patterns
        user_lower = user_message.lower()

        # Negative patterns
        negative_patterns = [
            "that's wrong", "not correct", "incorrect", "no that's not",
            "actually", "you're wrong", "that doesn't", "that's not what",
            "i said", "i meant", "not what i asked"
        ]

        for pattern in negative_patterns:
            if pattern in user_lower:
                return "corrected", 0.85

        # Positive patterns
        positive_patterns = [
            "thanks", "perfect", "exactly", "that's right", "correct",
            "yes", "great", "helpful", "that works"
        ]

        for pattern in positive_patterns:
            if pattern in user_lower:
                return "confirmed", 0.75

        # Citation-based inference
        if cited_segments:
            citation_rate = len(set(cited_segments) & set(retrieved_segments)) / len(retrieved_segments)
            if citation_rate > 0.5:
                return "used", 0.7
            elif citation_rate > 0:
                return "continued", 0.5

        # If no clear signal, return ambiguous
        return "ambiguous", 0.3


# =============================================================================
# GLOBAL SERVICE INSTANCES
# =============================================================================

_retrieval_logger: Optional[RetrievalLogger] = None
_learning_analyzer: Optional[LearningAnalyzer] = None
_feedback_inference: Optional[FeedbackInference] = None


def get_retrieval_logger() -> RetrievalLogger:
    """Get global retrieval logger instance."""
    global _retrieval_logger
    if _retrieval_logger is None:
        _retrieval_logger = RetrievalLogger()
    return _retrieval_logger


def get_learning_analyzer() -> LearningAnalyzer:
    """Get global learning analyzer instance."""
    global _learning_analyzer
    if _learning_analyzer is None:
        _learning_analyzer = LearningAnalyzer()
    return _learning_analyzer


def get_feedback_inference() -> FeedbackInference:
    """Get global feedback inference instance."""
    global _feedback_inference
    if _feedback_inference is None:
        _feedback_inference = FeedbackInference()
    return _feedback_inference
