"""Redis event publisher for subconscious daemon integration.

Publishes events to gami:events channel for the subconscious daemon to consume.
Events include: message, query, session_start, session_end, feedback
"""

import json
import logging
from datetime import datetime
from typing import Optional, List, Any

import redis

logger = logging.getLogger("gami.events")

# Redis connection (lazy init)
_redis_client: Optional[redis.Redis] = None


def _get_redis() -> redis.Redis:
    """Get or create Redis client."""
    global _redis_client
    if _redis_client is None:
        from api.config import settings
        _redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
    return _redis_client


def publish_event(
    event_type: str,
    session_id: Optional[str] = None,
    tenant_id: str = "claude-opus",
    agent_id: Optional[str] = None,
    **kwargs
) -> bool:
    """Publish event to gami:events channel.

    Args:
        event_type: Type of event (message, query, session_start, session_end, feedback)
        session_id: Session identifier
        tenant_id: Tenant identifier
        agent_id: Agent identifier
        **kwargs: Additional event data

    Returns:
        True if published successfully
    """
    try:
        event = {
            "type": event_type,
            "session_id": session_id,
            "tenant_id": tenant_id,
            "agent_id": agent_id,
            "timestamp": datetime.utcnow().isoformat(),
            **kwargs
        }

        r = _get_redis()
        r.publish("gami:events", json.dumps(event))
        logger.debug("Published event: %s (session=%s)", event_type, session_id)
        return True

    except Exception as e:
        logger.warning("Failed to publish event: %s", e)
        return False


def publish_query_event(
    session_id: str,
    query: str,
    tenant_id: str = "claude-opus",
    agent_id: Optional[str] = None,
    results: Optional[List[dict]] = None,
) -> bool:
    """Publish a query/recall event."""
    return publish_event(
        event_type="query",
        session_id=session_id,
        tenant_id=tenant_id,
        agent_id=agent_id,
        query=query,
        results=results or [],
    )


def publish_message_event(
    session_id: str,
    message: str,
    message_type: str = "user",
    tenant_id: str = "claude-opus",
    agent_id: Optional[str] = None,
    entities: Optional[List[str]] = None,
) -> bool:
    """Publish a message event."""
    return publish_event(
        event_type="message",
        session_id=session_id,
        tenant_id=tenant_id,
        agent_id=agent_id,
        message=message,
        message_type=message_type,
        entities=entities or [],
    )


def publish_session_start(
    session_id: str,
    tenant_id: str = "claude-opus",
    agent_id: Optional[str] = None,
) -> bool:
    """Publish session start event."""
    return publish_event(
        event_type="session_start",
        session_id=session_id,
        tenant_id=tenant_id,
        agent_id=agent_id,
    )


def publish_session_end(session_id: str) -> bool:
    """Publish session end event."""
    return publish_event(
        event_type="session_end",
        session_id=session_id,
    )


def publish_feedback_event(
    session_id: str,
    feedback_type: str,
    segment_ids: Optional[List[str]] = None,
) -> bool:
    """Publish feedback event."""
    return publish_event(
        event_type="feedback",
        session_id=session_id,
        feedback_type=feedback_type,
        segment_ids=segment_ids or [],
    )
