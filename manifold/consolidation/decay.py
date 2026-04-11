"""Temporal decay for memory consolidation.

Implements forgetting curves for unreferenced memories:
- Segments and clusters decay over time if not accessed
- Reinforcement (access) resets decay
- Heavily decayed items are archived (moved to cold storage)
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import List, Tuple, Optional
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger("gami.consolidation.decay")


@dataclass
class DecayConfig:
    """Configuration for decay behavior."""
    decay_rate_per_day: float = 0.01  # 1% decay per day of inactivity
    archive_threshold: float = 0.1    # Archive when decay_score < 0.1
    reinforcement_boost: float = 0.2  # Boost on access
    min_age_days: int = 7             # Don't decay items younger than this
    max_decay_per_run: float = 0.1    # Max decay to apply in single run


def calculate_decay_score(
    current_score: float,
    last_accessed: datetime,
    now: Optional[datetime] = None,
    config: Optional[DecayConfig] = None,
) -> float:
    """Calculate new decay score based on time since last access.

    Args:
        current_score: Current decay_score (0.0 to 1.0)
        last_accessed: When item was last accessed
        now: Current time (defaults to now)
        config: Decay configuration

    Returns:
        New decay score (0.0 to 1.0)
    """
    if config is None:
        config = DecayConfig()

    if now is None:
        now = datetime.now(timezone.utc)

    # Ensure timezone-aware comparison
    if last_accessed.tzinfo is None:
        last_accessed = last_accessed.replace(tzinfo=timezone.utc)

    days_since_access = (now - last_accessed).total_seconds() / 86400.0

    # Don't decay young items
    if days_since_access < config.min_age_days:
        return current_score

    # Calculate decay amount
    effective_days = days_since_access - config.min_age_days
    decay_amount = effective_days * config.decay_rate_per_day

    # Cap decay per run
    decay_amount = min(decay_amount, config.max_decay_per_run)

    new_score = max(0.0, current_score - decay_amount)
    return round(new_score, 4)


def reinforce_item(
    current_score: float,
    config: Optional[DecayConfig] = None,
) -> float:
    """Boost decay score when item is accessed (reinforcement).

    Args:
        current_score: Current decay_score
        config: Decay configuration

    Returns:
        New decay score (capped at 1.0)
    """
    if config is None:
        config = DecayConfig()

    new_score = min(1.0, current_score + config.reinforcement_boost)
    return round(new_score, 4)


def apply_decay(
    db: Session,
    table: str = "segments",
    tenant_id: Optional[str] = None,
    config: Optional[DecayConfig] = None,
    dry_run: bool = False,
) -> Tuple[int, int]:
    """Apply decay to items that haven't been accessed recently.

    Args:
        db: Database session
        table: Table to apply decay to ('segments' or 'memory_clusters')
        tenant_id: Optional tenant filter
        config: Decay configuration
        dry_run: If True, don't actually update

    Returns:
        Tuple of (decayed_count, archived_count)
    """
    if config is None:
        config = DecayConfig()

    now = datetime.now(timezone.utc)
    min_age = now - timedelta(days=config.min_age_days)

    # Build query based on table
    if table == "segments":
        access_col = "last_retrieved_at"
        id_col = "segment_id"
        tenant_col = "owner_tenant_id"
    elif table == "memory_clusters":
        access_col = "last_accessed_at"
        id_col = "cluster_id"
        tenant_col = "owner_tenant_id"
    else:
        raise ValueError(f"Unknown table: {table}")

    # Find items to decay
    where_clause = f"""
        WHERE decay_score > :threshold
        AND status = 'active'
        AND COALESCE({access_col}, created_at) < :min_age
    """
    params = {
        "threshold": config.archive_threshold,
        "min_age": min_age,
    }

    if tenant_id:
        where_clause += f" AND {tenant_col} = :tenant_id"
        params["tenant_id"] = tenant_id

    # Get items needing decay
    query = f"""
        SELECT {id_col}, decay_score, COALESCE({access_col}, created_at) as last_access
        FROM {table}
        {where_clause}
        LIMIT 1000
    """

    result = db.execute(text(query), params)
    rows = result.fetchall()

    if not rows:
        logger.debug("No items to decay in %s", table)
        return (0, 0)

    decayed_count = 0
    archived_count = 0

    for row in rows:
        item_id = row[0]
        current_score = row[1] or 1.0
        last_access = row[2]

        new_score = calculate_decay_score(current_score, last_access, now, config)

        if new_score == current_score:
            continue

        if not dry_run:
            if new_score <= config.archive_threshold:
                # Archive the item
                db.execute(
                    text(f"UPDATE {table} SET status = 'archived', decay_score = :score WHERE {id_col} = :id"),
                    {"score": new_score, "id": item_id}
                )
                archived_count += 1
            else:
                # Just update decay score
                db.execute(
                    text(f"UPDATE {table} SET decay_score = :score WHERE {id_col} = :id"),
                    {"score": new_score, "id": item_id}
                )
                decayed_count += 1
        else:
            if new_score <= config.archive_threshold:
                archived_count += 1
            else:
                decayed_count += 1

    if not dry_run:
        db.commit()

    logger.info(
        "Decay applied to %s: %d decayed, %d archived (dry_run=%s)",
        table, decayed_count, archived_count, dry_run
    )

    return (decayed_count, archived_count)


def archive_decayed(
    db: Session,
    table: str = "segments",
    tenant_id: Optional[str] = None,
    config: Optional[DecayConfig] = None,
) -> int:
    """Archive items that have decayed below threshold.

    Args:
        db: Database session
        table: Table to process
        tenant_id: Optional tenant filter
        config: Decay configuration

    Returns:
        Number of items archived
    """
    if config is None:
        config = DecayConfig()

    if table == "segments":
        id_col = "segment_id"
        tenant_col = "owner_tenant_id"
    elif table == "memory_clusters":
        id_col = "cluster_id"
        tenant_col = "owner_tenant_id"
    else:
        raise ValueError(f"Unknown table: {table}")

    where_clause = """
        WHERE decay_score <= :threshold
        AND status = 'active'
    """
    params = {"threshold": config.archive_threshold}

    if tenant_id:
        where_clause += f" AND {tenant_col} = :tenant_id"
        params["tenant_id"] = tenant_id

    result = db.execute(
        text(f"UPDATE {table} SET status = 'archived' {where_clause} RETURNING {id_col}"),
        params
    )

    archived_ids = [row[0] for row in result.fetchall()]
    db.commit()

    if archived_ids:
        logger.info("Archived %d items from %s", len(archived_ids), table)

    return len(archived_ids)


def record_access(
    db: Session,
    item_id: str,
    table: str = "segments",
) -> None:
    """Record that an item was accessed (for retrieval tracking).

    Updates last_retrieved_at/last_accessed_at and boosts decay_score.

    Args:
        db: Database session
        item_id: ID of accessed item
        table: Table containing the item
    """
    config = DecayConfig()
    now = datetime.now(timezone.utc)

    if table == "segments":
        access_col = "last_retrieved_at"
        id_col = "segment_id"
        count_col = "retrieval_count"
    elif table == "memory_clusters":
        access_col = "last_accessed_at"
        id_col = "cluster_id"
        count_col = "access_count"
    else:
        return

    # Get current decay score
    result = db.execute(
        text(f"SELECT decay_score FROM {table} WHERE {id_col} = :id"),
        {"id": item_id}
    )
    row = result.fetchone()

    if row:
        current_score = row[0] or 1.0
        new_score = reinforce_item(current_score, config)

        db.execute(
            text(f"""
                UPDATE {table}
                SET {access_col} = :now,
                    decay_score = :score,
                    {count_col} = COALESCE({count_col}, 0) + 1
                WHERE {id_col} = :id
            """),
            {"now": now, "score": new_score, "id": item_id}
        )
