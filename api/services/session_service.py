"""Session tracking service for GAMI.

Tracks conversation sessions for subconscious daemon and analytics.
Sessions are created on first activity and updated on subsequent interactions.
"""

import logging
from datetime import datetime, timezone
from typing import Optional, List

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger("gami.services.session_service")


async def ensure_session(
    db: AsyncSession,
    session_id: str,
    tenant_id: str,
    agent_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> bool:
    """Create session if not exists, or update last_activity_at.

    Args:
        db: Database session
        session_id: Unique session identifier
        tenant_id: Tenant ID
        agent_id: Optional agent ID
        user_id: Optional user ID

    Returns:
        True if new session created, False if existing updated
    """
    now = datetime.now(timezone.utc)

    # Check if exists
    result = await db.execute(
        text("SELECT id FROM sessions WHERE session_id = :sid"),
        {"sid": session_id}
    )
    existing = result.fetchone()

    if existing:
        # Update existing session
        await db.execute(
            text("""
                UPDATE sessions
                SET last_activity_at = :now,
                    message_count = message_count + 1
                WHERE session_id = :sid
            """),
            {"now": now, "sid": session_id}
        )
        await db.commit()
        return False
    else:
        # Create new session
        await db.execute(
            text("""
                INSERT INTO sessions (
                    session_id, tenant_id, agent_id, user_id,
                    conversation_state, state_confidence,
                    started_at, last_activity_at, message_count
                ) VALUES (
                    :sid, :tid, :aid, :uid,
                    'idle', 0.5,
                    :now, :now, 1
                )
            """),
            {
                "sid": session_id,
                "tid": tenant_id,
                "aid": agent_id,
                "uid": user_id,
                "now": now,
            }
        )
        await db.commit()
        logger.debug("Created new session: %s (tenant=%s)", session_id, tenant_id)
        return True


async def update_session_state(
    db: AsyncSession,
    session_id: str,
    state: str,
    confidence: float = 0.5,
    active_entities: Optional[List[str]] = None,
    active_topics: Optional[List[str]] = None,
) -> None:
    """Update session conversation state.

    Args:
        db: Database session
        session_id: Session identifier
        state: New conversation state (idle, debugging, planning, etc.)
        confidence: State confidence (0-1)
        active_entities: List of active entity IDs
        active_topics: List of active topics
    """
    now = datetime.now(timezone.utc)

    await db.execute(
        text("""
            UPDATE sessions
            SET conversation_state = :state,
                state_confidence = :conf,
                active_entities = :entities,
                active_topics = :topics,
                last_activity_at = :now
            WHERE session_id = :sid
        """),
        {
            "state": state,
            "conf": confidence,
            "entities": active_entities,
            "topics": active_topics,
            "now": now,
            "sid": session_id,
        }
    )
    await db.commit()


async def record_retrieval(
    db: AsyncSession,
    session_id: str,
) -> None:
    """Increment retrieval count for session.

    Args:
        db: Database session
        session_id: Session identifier
    """
    await db.execute(
        text("""
            UPDATE sessions
            SET retrieval_count = retrieval_count + 1,
                last_activity_at = NOW()
            WHERE session_id = :sid
        """),
        {"sid": session_id}
    )
    await db.commit()


async def record_learning_signal(
    db: AsyncSession,
    session_id: str,
    positive: bool,
) -> None:
    """Record learning signal (positive or negative feedback).

    Args:
        db: Database session
        session_id: Session identifier
        positive: True for positive signal, False for negative
    """
    if positive:
        col = "learning_signals_positive"
    else:
        col = "learning_signals_negative"

    await db.execute(
        text(f"""
            UPDATE sessions
            SET {col} = {col} + 1,
                last_activity_at = NOW()
            WHERE session_id = :sid
        """),
        {"sid": session_id}
    )
    await db.commit()


async def end_session(
    db: AsyncSession,
    session_id: str,
) -> None:
    """Mark session as ended.

    Args:
        db: Database session
        session_id: Session identifier
    """
    now = datetime.now(timezone.utc)

    await db.execute(
        text("""
            UPDATE sessions
            SET ended_at = :now
            WHERE session_id = :sid
        """),
        {"now": now, "sid": session_id}
    )
    await db.commit()
    logger.debug("Ended session: %s", session_id)


async def get_active_sessions(
    db: AsyncSession,
    tenant_id: Optional[str] = None,
    limit: int = 100,
) -> list:
    """Get active (not ended) sessions.

    Args:
        db: Database session
        tenant_id: Optional tenant filter
        limit: Maximum sessions to return

    Returns:
        List of session dicts
    """
    query = """
        SELECT session_id, tenant_id, agent_id, user_id,
               conversation_state, state_confidence,
               message_count, retrieval_count,
               started_at, last_activity_at
        FROM sessions
        WHERE ended_at IS NULL
    """
    params = {"limit": limit}

    if tenant_id:
        query += " AND tenant_id = :tid"
        params["tid"] = tenant_id

    query += " ORDER BY last_activity_at DESC LIMIT :limit"

    result = await db.execute(text(query), params)
    rows = result.fetchall()

    return [
        {
            "session_id": r[0],
            "tenant_id": r[1],
            "agent_id": r[2],
            "user_id": r[3],
            "conversation_state": r[4],
            "state_confidence": r[5],
            "message_count": r[6],
            "retrieval_count": r[7],
            "started_at": r[8].isoformat() if r[8] else None,
            "last_activity_at": r[9].isoformat() if r[9] else None,
        }
        for r in rows
    ]
