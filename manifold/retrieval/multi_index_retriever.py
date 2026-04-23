"""Multi-index retrieval with query routing.

Retrieves from multiple indexes based on routing decisions and fuses results.
Each index type has specialized search logic optimized for its data structure.
"""
import logging
from dataclasses import dataclass
from typing import List, Dict, Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from .query_routing import QueryRouting, IndexType

logger = logging.getLogger("gami.retrieval.multi_index")


@dataclass
class IndexResult:
    """Result from a single index search."""
    item_id: str
    item_type: str
    text: str
    score: float
    index_source: IndexType
    metadata: Dict[str, Any]


class MultiIndexRetriever:
    """Retrieves from multiple indexes based on routing decision."""

    async def retrieve(
        self,
        db: AsyncSession,
        query: str,
        query_embedding: List[float],
        routing: QueryRouting,
        tenant_ids: List[str],
        limit_per_index: int = 15,
    ) -> List[IndexResult]:
        """Query multiple indexes and fuse results.

        Args:
            db: Database session
            query: Search query text
            query_embedding: Query embedding vector
            routing: Routing decision from QueryRouter
            tenant_ids: Tenant scope
            limit_per_index: Max results per index

        Returns:
            Fused and deduplicated results from all queried indexes
        """
        all_results: List[IndexResult] = []
        vec_str = "[" + ",".join(str(v) for v in query_embedding) + "]"

        for index, weight in routing.index_weights.items():
            if weight < 0.1:
                continue

            try:
                results = await self._query_index(
                    db, query, vec_str, index, tenant_ids, limit_per_index
                )

                # Apply routing weight to scores
                for r in results:
                    r.score *= weight

                all_results.extend(results)
                logger.debug(f"Retrieved {len(results)} from {index.value}")

            except Exception as e:
                logger.warning(f"Error querying {index.value}: {e}")

        # Fuse and deduplicate results
        return self._fuse_results(all_results, limit=limit_per_index * 2)

    async def _query_index(
        self,
        db: AsyncSession,
        _query: str,  # Reserved for future full-text search within indexes
        vec_str: str,
        index: IndexType,
        tenant_ids: List[str],
        limit: int,
    ) -> List[IndexResult]:
        """Query a specific index."""
        if index == IndexType.CAUSAL:
            return await self._query_causal(db, vec_str, tenant_ids, limit)
        elif index == IndexType.PROCEDURES:
            return await self._query_procedures(db, vec_str, tenant_ids, limit)
        elif index == IndexType.ENTITIES:
            return await self._query_entities(db, vec_str, tenant_ids, limit)
        elif index == IndexType.RELATIONS:
            return await self._query_relations(db, vec_str, tenant_ids, limit)
        elif index == IndexType.CLAIMS:
            return await self._query_claims(db, vec_str, tenant_ids, limit)
        elif index == IndexType.MEMORIES:
            return await self._query_memories(db, vec_str, tenant_ids, limit)
        elif index == IndexType.CLUSTERS:
            return await self._query_clusters(db, vec_str, tenant_ids, limit)
        else:  # SEGMENTS - handled by main retrieval pipeline
            return []

    async def _query_causal(
        self,
        db: AsyncSession,
        vec_str: str,
        tenant_ids: List[str],
        limit: int,
    ) -> List[IndexResult]:
        """Query causal_relations table."""
        result = await db.execute(text("""
            SELECT cr.causal_id, cr.cause_text, cr.effect_text, cr.causal_type,
                   cr.strength_score, cr.source_segment_id,
                   s.embedding <=> CAST(:vec AS vector) AS distance
            FROM causal_relations cr
            LEFT JOIN segments s ON cr.source_segment_id = s.segment_id
            WHERE cr.owner_tenant_id = ANY(:tids)
            AND cr.status = 'active'
            AND s.embedding IS NOT NULL
            ORDER BY s.embedding <=> CAST(:vec AS vector)
            LIMIT :lim
        """), {"vec": vec_str, "tids": tenant_ids, "lim": limit})

        return [
            IndexResult(
                item_id=r.causal_id,
                item_type="causal_relation",
                text=f"Cause: {r.cause_text}\nEffect: {r.effect_text}",
                score=max(0, 1.0 - float(r.distance)) * float(r.strength_score or 0.5),
                index_source=IndexType.CAUSAL,
                metadata={
                    "causal_type": r.causal_type,
                    "source_segment_id": r.source_segment_id,
                },
            )
            for r in result.fetchall()
        ]

    async def _query_procedures(
        self,
        db: AsyncSession,
        vec_str: str,
        tenant_ids: List[str],
        limit: int,
    ) -> List[IndexResult]:
        """Query workflow memories (primary) and legacy procedures (fallback)."""
        results = []

        # Primary: Query workflow memories
        try:
            mem_result = await db.execute(text("""
                SELECT memory_id, normalized_text, importance_score,
                       embedding <=> CAST(:vec AS vector) AS distance
                FROM assistant_memories
                WHERE owner_tenant_id = ANY(:tids)
                AND memory_type = 'workflow'
                AND status = 'active'
                AND embedding IS NOT NULL
                ORDER BY embedding <=> CAST(:vec AS vector)
                LIMIT :lim
            """), {"vec": vec_str, "tids": tenant_ids, "lim": limit})

            for r in mem_result.fetchall():
                results.append(IndexResult(
                    item_id=r.memory_id,
                    item_type="workflow",
                    text=r.normalized_text,
                    score=max(0, 1.0 - float(r.distance)) * float(r.importance_score or 0.5),
                    index_source=IndexType.PROCEDURES,
                    metadata={"source": "workflow_memory"},
                ))
        except Exception as e:
            logger.debug(f"Workflow memory query failed: {e}")

        # Fallback: Query legacy procedures if we need more results
        if len(results) < limit:
            try:
                proc_result = await db.execute(text("""
                    SELECT procedure_id, name, description, category, steps,
                           success_rate, confidence,
                           embedding <=> CAST(:vec AS vector) AS distance
                    FROM procedures
                    WHERE owner_tenant_id = ANY(:tids)
                    AND status = 'active'
                    AND embedding IS NOT NULL
                    ORDER BY embedding <=> CAST(:vec AS vector)
                    LIMIT :lim
                """), {"vec": vec_str, "tids": tenant_ids, "lim": limit - len(results)})

                for r in proc_result.fetchall():
                    # Format steps for display
                    steps_text = ""
                    if r.steps:
                        import json
                        steps = json.loads(r.steps) if isinstance(r.steps, str) else r.steps
                        if isinstance(steps, list):
                            steps_text = "\n".join(
                                f"{i+1}. {s.get('action', s)}"
                                for i, s in enumerate(steps[:5])
                            )

                    results.append(IndexResult(
                        item_id=r.procedure_id,
                        item_type="procedure",
                        text=f"**{r.name}**\n{r.description or ''}\n\nSteps:\n{steps_text}",
                        score=max(0, 1.0 - float(r.distance)) * float(r.confidence or 0.5),
                        index_source=IndexType.PROCEDURES,
                        metadata={
                            "category": r.category,
                            "success_rate": float(r.success_rate) if r.success_rate else 0.5,
                            "source": "legacy_procedure",
                        },
                    ))
            except Exception as e:
                logger.debug(f"Legacy procedures query failed: {e}")

        return results

    async def _query_entities(
        self,
        db: AsyncSession,
        vec_str: str,
        tenant_ids: List[str],
        limit: int,
    ) -> List[IndexResult]:
        """Query entities table."""
        result = await db.execute(text("""
            SELECT entity_id, canonical_name, entity_type, description,
                   importance_score, mention_count,
                   embedding <=> CAST(:vec AS vector) AS distance
            FROM entities
            WHERE owner_tenant_id = ANY(:tids)
            AND status = 'active'
            AND embedding IS NOT NULL
            ORDER BY embedding <=> CAST(:vec AS vector)
            LIMIT :lim
        """), {"vec": vec_str, "tids": tenant_ids, "lim": limit})

        return [
            IndexResult(
                item_id=r.entity_id,
                item_type="entity",
                text=f"**{r.canonical_name}** ({r.entity_type})\n{r.description or ''}",
                score=max(0, 1.0 - float(r.distance)),
                index_source=IndexType.ENTITIES,
                metadata={
                    "entity_type": r.entity_type,
                    "importance": float(r.importance_score) if r.importance_score else 0.5,
                    "mentions": r.mention_count or 0,
                },
            )
            for r in result.fetchall()
        ]

    async def _query_relations(
        self,
        db: AsyncSession,
        vec_str: str,
        tenant_ids: List[str],
        limit: int,
    ) -> List[IndexResult]:
        """Query relations table."""
        result = await db.execute(text("""
            SELECT r.relation_id, r.relation_type,
                   e1.canonical_name as source_name, e1.entity_type as source_type,
                   e2.canonical_name as target_name, e2.entity_type as target_type,
                   r.confidence, r.evidence_count,
                   s.embedding <=> CAST(:vec AS vector) AS distance
            FROM relations r
            LEFT JOIN entities e1 ON r.source_entity_id = e1.entity_id
            LEFT JOIN entities e2 ON r.target_entity_id = e2.entity_id
            LEFT JOIN segments s ON r.source_segment_id = s.segment_id
            WHERE r.owner_tenant_id = ANY(:tids)
            AND r.status = 'active'
            AND s.embedding IS NOT NULL
            ORDER BY s.embedding <=> CAST(:vec AS vector)
            LIMIT :lim
        """), {"vec": vec_str, "tids": tenant_ids, "lim": limit})

        return [
            IndexResult(
                item_id=r.relation_id,
                item_type="relation",
                text=f"{r.source_name} --[{r.relation_type}]--> {r.target_name}",
                score=max(0, 1.0 - float(r.distance)) * float(r.confidence or 0.5),
                index_source=IndexType.RELATIONS,
                metadata={
                    "relation_type": r.relation_type,
                    "source_entity": r.source_name,
                    "target_entity": r.target_name,
                    "evidence_count": r.evidence_count or 0,
                },
            )
            for r in result.fetchall()
        ]

    async def _query_claims(
        self,
        db: AsyncSession,
        vec_str: str,
        tenant_ids: List[str],
        limit: int,
    ) -> List[IndexResult]:
        """Query claims table."""
        result = await db.execute(text("""
            SELECT claim_id, summary_text, predicate, confidence,
                   support_count, modality,
                   embedding <=> CAST(:vec AS vector) AS distance
            FROM claims
            WHERE owner_tenant_id = ANY(:tids)
            AND status = 'active'
            AND embedding IS NOT NULL
            ORDER BY embedding <=> CAST(:vec AS vector)
            LIMIT :lim
        """), {"vec": vec_str, "tids": tenant_ids, "lim": limit})

        return [
            IndexResult(
                item_id=r.claim_id,
                item_type="claim",
                text=r.summary_text or f"[{r.predicate}]",
                score=max(0, 1.0 - float(r.distance)) * float(r.confidence or 0.5),
                index_source=IndexType.CLAIMS,
                metadata={
                    "predicate": r.predicate,
                    "modality": r.modality,
                    "support_count": r.support_count or 0,
                },
            )
            for r in result.fetchall()
        ]

    async def _query_memories(
        self,
        db: AsyncSession,
        vec_str: str,
        tenant_ids: List[str],
        limit: int,
    ) -> List[IndexResult]:
        """Query assistant_memories table."""
        result = await db.execute(text("""
            SELECT memory_id, normalized_text, memory_type, subject_id,
                   importance_score, stability_score,
                   embedding <=> CAST(:vec AS vector) AS distance
            FROM assistant_memories
            WHERE owner_tenant_id = ANY(:tids)
            AND status = 'active'
            AND embedding IS NOT NULL
            ORDER BY embedding <=> CAST(:vec AS vector)
            LIMIT :lim
        """), {"vec": vec_str, "tids": tenant_ids, "lim": limit})

        return [
            IndexResult(
                item_id=r.memory_id,
                item_type="memory",
                text=r.normalized_text,
                score=max(0, 1.0 - float(r.distance)) * float(r.importance_score or 0.5),
                index_source=IndexType.MEMORIES,
                metadata={
                    "memory_type": r.memory_type,
                    "subject_id": r.subject_id,
                    "stability": float(r.stability_score) if r.stability_score else 0.5,
                },
            )
            for r in result.fetchall()
        ]

    async def _query_clusters(
        self,
        db: AsyncSession,
        vec_str: str,
        tenant_ids: List[str],
        limit: int,
    ) -> List[IndexResult]:
        """Query memory_clusters abstractions."""
        result = await db.execute(text("""
            SELECT cluster_id, cluster_name, abstraction_text,
                   member_count, stability_score,
                   abstraction_embedding <=> CAST(:vec AS vector) AS distance
            FROM memory_clusters
            WHERE owner_tenant_id = ANY(:tids)
            AND status = 'active'
            AND abstraction_embedding IS NOT NULL
            ORDER BY abstraction_embedding <=> CAST(:vec AS vector)
            LIMIT :lim
        """), {"vec": vec_str, "tids": tenant_ids, "lim": limit})

        return [
            IndexResult(
                item_id=r.cluster_id,
                item_type="cluster",
                text=r.abstraction_text or f"[Cluster: {r.cluster_name}]",
                score=max(0, 1.0 - float(r.distance)) * float(r.stability_score or 0.5),
                index_source=IndexType.CLUSTERS,
                metadata={
                    "cluster_name": r.cluster_name,
                    "member_count": r.member_count or 0,
                },
            )
            for r in result.fetchall()
        ]

    def _fuse_results(
        self,
        results: List[IndexResult],
        limit: int,
    ) -> List[IndexResult]:
        """Fuse results from multiple indexes, deduplicating by content similarity."""
        if not results:
            return []

        # Group by item_id (same item from different indexes)
        by_id: Dict[str, List[IndexResult]] = {}
        for r in results:
            key = f"{r.item_type}:{r.item_id}"
            by_id.setdefault(key, []).append(r)

        # Take max score for duplicates, boost items appearing in multiple indexes
        fused = []
        for key, items in by_id.items():
            best = max(items, key=lambda x: x.score)
            # Boost items that appear in multiple indexes (cross-index confirmation)
            if len(items) > 1:
                best.score *= (1.0 + 0.1 * len(items))
            fused.append(best)

        # Sort by score descending
        fused.sort(key=lambda x: -x.score)

        return fused[:limit]
