"""Mem0-style memory operation classification.

Classifies incoming memories as ADD/UPDATE/DELETE/NOOP to avoid
storing duplicate or redundant information.

The classifier:
1. Embeds the new memory
2. Finds similar existing memories
3. Uses similarity thresholds or LLM to decide the operation
4. Returns the appropriate action to take
"""
import logging
import os
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List
import requests

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger("gami.services.memory_classifier")


class OperationType(str, Enum):
    """Memory operation types."""
    ADD = "ADD"         # Store as new memory
    UPDATE = "UPDATE"   # Update existing memory
    DELETE = "DELETE"   # Archive target memory
    NOOP = "NOOP"       # Skip - duplicate/redundant


@dataclass
class SimilarMemory:
    """A similar existing memory."""
    memory_id: str
    text: str
    similarity: float
    memory_type: str
    subject_id: str


@dataclass
class MemoryOperation:
    """Result of memory classification."""
    type: OperationType
    new_text: str
    target_id: Optional[str] = None
    target_text: Optional[str] = None
    similarity: float = 0.0
    reason: str = ""
    confidence: float = 0.5


class MemoryOperationClassifier:
    """Classify new memories against existing ones.

    Uses a two-stage approach:
    1. Fast similarity check via embeddings
    2. LLM-based decision for ambiguous cases

    Thresholds (configurable via env):
    - EXACT_THRESHOLD (0.95): Above this = NOOP (duplicate)
    - SIMILAR_THRESHOLD (0.75): Above this = LLM decides UPDATE vs ADD
    - Below SIMILAR_THRESHOLD = ADD (new information)
    """

    def __init__(
        self,
        exact_threshold: float = None,
        similar_threshold: float = None,
        vllm_url: str = None,
    ):
        from api.config import settings
        self.exact_threshold = exact_threshold or settings.MEMORY_EXACT_THRESHOLD
        self.similar_threshold = similar_threshold or settings.MEMORY_SIMILAR_THRESHOLD
        self.vllm_url = vllm_url or settings.VLLM_URL

    async def classify(
        self,
        new_text: str,
        tenant_id: str,
        db: AsyncSession,
        memory_type: str = None,
        subject_id: str = None,
    ) -> MemoryOperation:
        """Classify a new memory and determine the appropriate operation.

        Args:
            new_text: The new memory text
            tenant_id: Tenant ID for memory scope
            db: Database session
            memory_type: Optional memory type filter
            subject_id: Optional subject filter

        Returns:
            MemoryOperation with the recommended action
        """
        # 1. Embed new memory
        from api.llm.embeddings import embed_text
        try:
            embedding = await embed_text(new_text)
        except Exception as e:
            logger.error(f"Failed to embed new memory: {e}")
            # Can't classify without embedding, default to ADD
            return MemoryOperation(
                type=OperationType.ADD,
                new_text=new_text,
                reason="Embedding failed, defaulting to ADD",
                confidence=0.3,
            )

        vec_str = "[" + ",".join(str(v) for v in embedding) + "]"

        # 2. Find similar existing memories
        similar = await self._find_similar(
            db, vec_str, tenant_id, memory_type, subject_id
        )

        # 3. No similar memories → ADD
        if not similar:
            logger.debug(f"No similar memories found, operation=ADD")
            return MemoryOperation(
                type=OperationType.ADD,
                new_text=new_text,
                reason="No similar existing memories",
                confidence=0.9,
            )

        top = similar[0]
        logger.debug(f"Top similar: {top.memory_id} (sim={top.similarity:.3f})")

        # 4. Very high similarity → NOOP (duplicate)
        if top.similarity >= self.exact_threshold:
            logger.debug(f"Exact match detected, operation=NOOP")
            return MemoryOperation(
                type=OperationType.NOOP,
                new_text=new_text,
                target_id=top.memory_id,
                target_text=top.text,
                similarity=top.similarity,
                reason=f"Duplicate memory (similarity={top.similarity:.3f})",
                confidence=0.95,
            )

        # 5. Moderately similar → LLM decides
        if top.similarity >= self.similar_threshold:
            decision = await self._llm_decide(new_text, top.text)
            return MemoryOperation(
                type=decision["type"],
                new_text=new_text,
                target_id=top.memory_id,
                target_text=top.text,
                similarity=top.similarity,
                reason=decision["reason"],
                confidence=decision["confidence"],
            )

        # 6. Low similarity → ADD
        return MemoryOperation(
            type=OperationType.ADD,
            new_text=new_text,
            similarity=top.similarity,
            reason=f"Low similarity to existing ({top.similarity:.3f} < {self.similar_threshold})",
            confidence=0.85,
        )

    async def _find_similar(
        self,
        db: AsyncSession,
        vec_str: str,
        tenant_id: str,
        memory_type: str = None,
        subject_id: str = None,
        limit: int = 5,
    ) -> List[SimilarMemory]:
        """Find similar existing memories."""
        # Build filter clause
        filters = ["owner_tenant_id = :tid", "status = 'active'"]
        params = {"vec": vec_str, "tid": tenant_id, "lim": limit}

        if memory_type:
            filters.append("memory_type = :mtype")
            params["mtype"] = memory_type
        if subject_id:
            filters.append("subject_id = :sid")
            params["sid"] = subject_id

        where_clause = " AND ".join(filters)

        result = await db.execute(text(f"""
            SELECT memory_id, normalized_text, memory_type, subject_id,
                   1 - (embedding <=> CAST(:vec AS vector)) as similarity
            FROM assistant_memories
            WHERE {where_clause}
            AND embedding IS NOT NULL
            ORDER BY embedding <=> CAST(:vec AS vector)
            LIMIT :lim
        """), params)

        return [
            SimilarMemory(
                memory_id=r.memory_id,
                text=r.normalized_text,
                similarity=float(r.similarity),
                memory_type=r.memory_type,
                subject_id=r.subject_id,
            )
            for r in result.fetchall()
        ]

    async def _llm_decide(
        self,
        new_text: str,
        existing_text: str,
    ) -> dict:
        """Use LLM to decide between ADD/UPDATE/NOOP."""
        prompt = f"""Compare these two memories and decide the operation:

EXISTING MEMORY:
{existing_text[:800]}

NEW MEMORY:
{new_text[:800]}

Decide which operation to perform:
- UPDATE: The NEW memory corrects, updates, or supersedes the EXISTING one
- ADD: The NEW memory provides distinct/complementary information
- NOOP: The NEW memory is redundant with EXISTING (same info, different words)

Reply with ONLY one word: UPDATE, ADD, or NOOP"""

        try:
            r = requests.post(
                f"{self.vllm_url}/chat/completions",
                json={
                    "model": os.getenv("GAMI_EXTRACTION_MODEL", "qwen35-27b-unredacted"),
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 20,
                    "temperature": 0.1,
                },
                timeout=30,
            )

            if r.status_code == 200:
                response = r.json()["choices"][0]["message"]["content"].strip().upper()

                if "UPDATE" in response:
                    return {
                        "type": OperationType.UPDATE,
                        "reason": "LLM: new memory updates existing",
                        "confidence": 0.8,
                    }
                elif "NOOP" in response:
                    return {
                        "type": OperationType.NOOP,
                        "reason": "LLM: memories are redundant",
                        "confidence": 0.8,
                    }
                else:  # ADD or unclear
                    return {
                        "type": OperationType.ADD,
                        "reason": "LLM: memories are complementary",
                        "confidence": 0.8,
                    }

        except Exception as e:
            logger.warning(f"LLM decision failed: {e}")

        # Default to ADD if LLM unavailable
        return {
            "type": OperationType.ADD,
            "reason": "LLM unavailable, defaulting to ADD",
            "confidence": 0.5,
        }


async def log_memory_operation(
    db: AsyncSession,
    operation: MemoryOperation,
    tenant_id: str,
    agent_id: str = None,
    result_memory_id: str = None,
    embedding: List[float] = None,
) -> str:
    """Log a memory operation to the tracking table.

    Returns the operation_id.
    """
    vec_str = None
    if embedding:
        vec_str = "[" + ",".join(str(v) for v in embedding) + "]"

    result = await db.execute(text("""
        INSERT INTO memory_operations (
            operation_type, new_memory_text, new_memory_embedding,
            target_memory_id, target_memory_text, similarity_score,
            decision_reason, llm_confidence,
            executed, executed_at, result_memory_id,
            tenant_id, agent_id
        ) VALUES (
            :op_type, :new_text, CAST(:vec AS vector),
            :target_id, :target_text, :similarity,
            :reason, :confidence,
            :executed, CASE WHEN :executed THEN NOW() ELSE NULL END, :result_id,
            :tenant_id, :agent_id
        )
        RETURNING operation_id
    """), {
        "op_type": operation.type.value,
        "new_text": operation.new_text,
        "vec": vec_str,
        "target_id": operation.target_id,
        "target_text": operation.target_text[:500] if operation.target_text else None,
        "similarity": operation.similarity,
        "reason": operation.reason,
        "confidence": operation.confidence,
        "executed": result_memory_id is not None,
        "result_id": result_memory_id,
        "tenant_id": tenant_id,
        "agent_id": agent_id,
    })

    row = result.fetchone()
    return row.operation_id if row else None
