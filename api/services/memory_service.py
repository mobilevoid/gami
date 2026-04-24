"""Assistant memory service for GAMI.

Manages persistent assistant memories — remembering, forgetting, updating,
and retrieving context for sessions. Memories are never deleted, only archived.
"""
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from api.llm.embeddings import embed_text
from api.services.db import AsyncSessionLocal

logger = logging.getLogger("gami.services.memory_service")

# Similarity threshold for duplicate detection
DUPLICATE_THRESHOLD = 0.92
# Minimum importance to store a memory
IMPORTANCE_FLOOR = 0.3
# Keywords that indicate credential content
CREDENTIAL_KEYWORDS = [
    "password", "token", "secret", "api_key", "apikey", "api key",
    "credential", "ssh key", "private key", "certificate",
]


def _make_id() -> str:
    return f"MEM_{uuid.uuid4().hex[:16]}"


def _detect_sensitivity(text_content: str) -> str:
    """Detect if text contains credentials or sensitive info."""
    lower = text_content.lower()
    for kw in CREDENTIAL_KEYWORDS:
        if kw in lower:
            return "credential"
    return "normal"


async def remember(
    tenant_id: str,
    text_content: str,
    memory_type: str = "fact",
    subject: str = "general",
    importance: float = 0.5,
    source_info: Optional[dict] = None,
    session_id: Optional[str] = None,
    db: Optional[AsyncSession] = None,
) -> dict:
    """Store a new assistant memory.

    Args:
        tenant_id: Owner tenant.
        text_content: The memory text to store.
        memory_type: Type of memory (fact, preference, instruction, context, observation).
        subject: Subject/topic identifier.
        importance: Importance score 0-1.
        source_info: Optional dict with provenance info.
        session_id: Optional session to link the memory to.
        db: Optional existing session.

    Returns:
        Dict with memory_id, status, and any flags (duplicate, contradiction).
    """
    # Importance floor
    if importance < IMPORTANCE_FLOOR:
        return {
            "stored": False,
            "reason": f"Importance {importance} below floor {IMPORTANCE_FLOOR}",
        }

    # Validate memory_type
    valid_types = {"fact", "preference", "instruction", "context", "observation", "credential"}
    if memory_type not in valid_types:
        memory_type = "fact"

    # Detect sensitivity
    sensitivity = _detect_sensitivity(text_content)
    if sensitivity == "credential":
        memory_type = "credential"

    # Get embedding for duplicate/contradiction check
    embedding = await embed_text(text_content)
    vec_str = "[" + ",".join(str(v) for v in embedding) + "]"

    close_session = False
    if db is None:
        db = AsyncSessionLocal()
        close_session = True

    try:
        now = datetime.now(timezone.utc)

        # Duplicate check: find similar active memories
        dup_result = await db.execute(
            text("""
                SELECT memory_id, normalized_text,
                       1 - (embedding <=> CAST(:vec AS vector)) AS similarity
                FROM assistant_memories
                WHERE owner_tenant_id = :tid
                  AND status IN ('active', 'provisional')
                  AND embedding IS NOT NULL
                ORDER BY embedding <=> CAST(:vec AS vector)
                LIMIT 3
            """),
            {"vec": vec_str, "tid": tenant_id},
        )
        similar = dup_result.fetchall()

        for sim in similar:
            if float(sim[2]) >= DUPLICATE_THRESHOLD:
                # Duplicate found — just bump confirmation
                await db.execute(
                    text("""
                        UPDATE assistant_memories
                        SET confirmation_count = confirmation_count + 1,
                            last_confirmed_at = :now,
                            stability_score = LEAST(1.0, stability_score + 0.1),
                            updated_at = :now
                        WHERE memory_id = :mid
                    """),
                    {"mid": sim[0], "now": now},
                )
                await db.commit()
                return {
                    "stored": False,
                    "reason": "duplicate",
                    "existing_memory_id": sim[0],
                    "similarity": float(sim[2]),
                }

        # No duplicate — create new memory
        memory_id = _make_id()
        source_json = json.dumps(source_info) if source_info else "{}"

        await db.execute(
            text("""
                INSERT INTO assistant_memories
                (memory_id, owner_tenant_id, memory_type, subject_id,
                 normalized_text, canonical_form_json, embedding,
                 importance_score, stability_score, recall_score,
                 confirmation_count, sensitivity, status, version,
                 created_at, updated_at)
                VALUES (:mid, :tid, :mtype, :subject,
                        :text, CAST(:source AS jsonb), CAST(:vec AS vector),
                        :importance, 0.3, 0.0,
                        0, :sensitivity, 'provisional', 1,
                        :now, :now)
            """),
            {
                "mid": memory_id,
                "tid": tenant_id,
                "mtype": memory_type,
                "subject": subject,
                "text": text_content,
                "source": source_json,
                "vec": vec_str,
                "importance": importance,
                "sensitivity": sensitivity,
                "now": now,
            },
        )

        # Link to session if provided
        if session_id:
            await db.execute(
                text("""
                    UPDATE sessions
                    SET memory_ids_retrieved = memory_ids_retrieved || CAST(:mid_arr AS jsonb),
                        last_active_at = :now
                    WHERE session_id = :sid
                """),
                {"mid_arr": json.dumps([memory_id]), "sid": session_id, "now": now},
            )

        await db.commit()

        return {
            "stored": True,
            "memory_id": memory_id,
            "memory_type": memory_type,
            "sensitivity": sensitivity,
            "status": "provisional",
        }

    except Exception as exc:
        logger.error("Failed to store memory: %s", exc, exc_info=True)
        await db.rollback()
        raise
    finally:
        if close_session:
            await db.close()


async def forget(
    memory_id: str,
    reason: str = "user_requested",
    db: Optional[AsyncSession] = None,
) -> dict:
    """Archive a memory (never truly delete).

    Args:
        memory_id: The memory to archive.
        reason: Why it's being archived.

    Returns:
        Dict with status.
    """
    close_session = False
    if db is None:
        db = AsyncSessionLocal()
        close_session = True

    try:
        now = datetime.now(timezone.utc)
        result = await db.execute(
            text("""
                UPDATE assistant_memories
                SET status = 'archived',
                    updated_at = :now
                WHERE memory_id = :mid
                  AND status != 'archived'
                RETURNING memory_id, normalized_text
            """),
            {"mid": memory_id, "now": now},
        )
        row = result.fetchone()
        await db.commit()

        if not row:
            return {"archived": False, "reason": "not found or already archived"}

        logger.info("Archived memory %s: %s", memory_id, reason)
        return {
            "archived": True,
            "memory_id": memory_id,
            "reason": reason,
        }
    finally:
        if close_session:
            await db.close()


async def update_memory(
    memory_id: str,
    new_text: str,
    reason: str = "correction",
    db: Optional[AsyncSession] = None,
) -> dict:
    """Update a memory by creating a new version and superseding the old one.

    The old memory is marked as superseded (not deleted).
    """
    close_session = False
    if db is None:
        db = AsyncSessionLocal()
        close_session = True

    try:
        now = datetime.now(timezone.utc)

        # Fetch old memory
        old_result = await db.execute(
            text("""
                SELECT memory_id, owner_tenant_id, memory_type, subject_id,
                       importance_score, confirmation_count, version, sensitivity
                FROM assistant_memories
                WHERE memory_id = :mid AND status IN ('active', 'provisional')
            """),
            {"mid": memory_id},
        )
        old = old_result.fetchone()
        if not old:
            return {"updated": False, "reason": "memory not found or not active"}

        # Create new version
        new_id = _make_id()
        new_embedding = await embed_text(new_text)
        vec_str = "[" + ",".join(str(v) for v in new_embedding) + "]"
        new_version = old[6] + 1
        sensitivity = _detect_sensitivity(new_text) or old[7]

        await db.execute(
            text("""
                INSERT INTO assistant_memories
                (memory_id, owner_tenant_id, memory_type, subject_id,
                 normalized_text, canonical_form_json, embedding,
                 importance_score, stability_score, recall_score,
                 confirmation_count, sensitivity, status, version,
                 created_at, updated_at)
                VALUES (:mid, :tid, :mtype, :subject,
                        :text, '{}', CAST(:vec AS vector),
                        :importance, 0.5, 0.0,
                        :conf_count, :sensitivity, 'active', :version,
                        :now, :now)
            """),
            {
                "mid": new_id,
                "tid": old[1],
                "mtype": old[2],
                "subject": old[3],
                "text": new_text,
                "vec": vec_str,
                "importance": old[4],
                "conf_count": old[5],
                "sensitivity": sensitivity,
                "version": new_version,
                "now": now,
            },
        )

        # Supersede old memory
        await db.execute(
            text("""
                UPDATE assistant_memories
                SET status = 'superseded',
                    superseded_by_id = :new_id,
                    updated_at = :now
                WHERE memory_id = :mid
            """),
            {"new_id": new_id, "mid": memory_id, "now": now},
        )

        await db.commit()

        return {
            "updated": True,
            "old_memory_id": memory_id,
            "new_memory_id": new_id,
            "version": new_version,
            "reason": reason,
        }
    except Exception as exc:
        logger.error("Failed to update memory: %s", exc, exc_info=True)
        await db.rollback()
        raise
    finally:
        if close_session:
            await db.close()


async def confirm(
    memory_id: str,
    db: Optional[AsyncSession] = None,
) -> dict:
    """Confirm a memory, incrementing confirmation_count and updating stability."""
    close_session = False
    if db is None:
        db = AsyncSessionLocal()
        close_session = True

    try:
        now = datetime.now(timezone.utc)
        result = await db.execute(
            text("""
                UPDATE assistant_memories
                SET confirmation_count = confirmation_count + 1,
                    stability_score = LEAST(1.0, stability_score + 0.1),
                    status = CASE
                        WHEN status = 'provisional' AND confirmation_count >= 1
                        THEN 'active'
                        ELSE status
                    END,
                    last_confirmed_at = :now,
                    updated_at = :now
                WHERE memory_id = :mid
                  AND status IN ('active', 'provisional')
                RETURNING memory_id, confirmation_count, status, stability_score
            """),
            {"mid": memory_id, "now": now},
        )
        row = result.fetchone()
        await db.commit()

        if not row:
            return {"confirmed": False, "reason": "not found or not active"}

        return {
            "confirmed": True,
            "memory_id": row[0],
            "confirmation_count": row[1],
            "status": row[2],
            "stability_score": float(row[3]),
        }
    finally:
        if close_session:
            await db.close()


async def get_context(
    session_id: str,
    tenant_id: str,
    max_memories: int = 20,
    db: Optional[AsyncSession] = None,
) -> dict:
    """Get active session context: recently retrieved memories + relevant active memories.

    Returns a combined context with session-specific and general memories.
    """
    close_session = False
    if db is None:
        db = AsyncSessionLocal()
        close_session = True

    try:
        # Get session info
        session_result = await db.execute(
            text("""
                SELECT session_id, tenant_id, memory_ids_retrieved,
                       entity_anchors, topic_anchors
                FROM sessions
                WHERE session_id = :sid
            """),
            {"sid": session_id},
        )
        session = session_result.fetchone()

        session_memories = []
        if session:
            memory_ids = session[2] if isinstance(session[2], list) else json.loads(session[2] or "[]")
            if memory_ids:
                placeholders = ", ".join(f":m{i}" for i in range(len(memory_ids[:20])))
                params = {f"m{i}": mid for i, mid in enumerate(memory_ids[:20])}
                mem_result = await db.execute(
                    text(f"""
                        SELECT memory_id, memory_type, subject_id,
                               normalized_text, importance_score, status
                        FROM assistant_memories
                        WHERE memory_id IN ({placeholders})
                          AND status IN ('active', 'provisional')
                        ORDER BY importance_score DESC
                    """),
                    params,
                )
                for row in mem_result.fetchall():
                    session_memories.append({
                        "memory_id": row[0],
                        "memory_type": row[1],
                        "subject": row[2],
                        "text": row[3],
                        "importance": float(row[4]),
                        "status": row[5],
                    })

        # Get top active memories for this tenant (general context)
        remaining = max_memories - len(session_memories)
        general_memories = []
        if remaining > 0:
            exclude_ids = [m["memory_id"] for m in session_memories]
            exclude_clause = ""
            params = {"tid": tenant_id, "lim": remaining}
            if exclude_ids:
                placeholders = ", ".join(f":ex{i}" for i in range(len(exclude_ids)))
                params.update({f"ex{i}": eid for i, eid in enumerate(exclude_ids)})
                exclude_clause = f"AND memory_id NOT IN ({placeholders})"

            gen_result = await db.execute(
                text(f"""
                    SELECT memory_id, memory_type, subject_id,
                           normalized_text, importance_score, status
                    FROM assistant_memories
                    WHERE owner_tenant_id = :tid
                      AND status IN ('active', 'provisional')
                      {exclude_clause}
                    ORDER BY importance_score DESC, confirmation_count DESC
                    LIMIT :lim
                """),
                params,
            )
            for row in gen_result.fetchall():
                general_memories.append({
                    "memory_id": row[0],
                    "memory_type": row[1],
                    "subject": row[2],
                    "text": row[3],
                    "importance": float(row[4]),
                    "status": row[5],
                })

        return {
            "session_id": session_id,
            "tenant_id": tenant_id,
            "session_memories": session_memories,
            "general_memories": general_memories,
            "total": len(session_memories) + len(general_memories),
        }
    finally:
        if close_session:
            await db.close()
