"""Graph sync service for GAMI.

Syncs entities, relations, and claims from relational tables into the
Apache AGE graph. Called after extraction completes. Idempotent.
"""
import json
import logging
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from api.services.db import AsyncSessionLocal

logger = logging.getLogger("gami.services.graph_sync")

GRAPH_NAME = "gami_graph"


async def _age_exec(db: AsyncSession, cypher: str) -> bool:
    """Execute a Cypher mutation (CREATE/MERGE) via AGE. Returns success."""
    sql = f"""
        SELECT * FROM cypher('{GRAPH_NAME}', $$
            {cypher}
        $$) AS (result agtype);
    """
    try:
        await db.execute(text("LOAD 'age'"))
        await db.execute(text("SET search_path = ag_catalog, public"))
        await db.execute(text(sql))
        return True
    except Exception as exc:
        logger.warning("AGE exec failed: %s\nCypher: %.200s", exc, cypher)
        return False


async def _node_exists(db: AsyncSession, entity_id: str) -> bool:
    """Check if a node with the given entity_id exists in the graph."""
    sql = f"""
        SELECT * FROM cypher('{GRAPH_NAME}', $$
            MATCH (n {{entity_id: '{entity_id}'}})
            RETURN count(n)
        $$) AS (cnt agtype);
    """
    try:
        await db.execute(text("LOAD 'age'"))
        await db.execute(text("SET search_path = ag_catalog, public"))
        result = await db.execute(text(sql))
        row = result.fetchone()
        if row:
            cnt = row[0]
            if isinstance(cnt, str):
                cnt = json.loads(cnt)
            return int(cnt) > 0
        return False
    except Exception:
        return False


async def _edge_exists(
    db: AsyncSession,
    from_id: str,
    to_id: str,
    rel_type: str,
) -> bool:
    """Check if an edge exists between two nodes."""
    sql = f"""
        SELECT * FROM cypher('{GRAPH_NAME}', $$
            MATCH (a {{entity_id: '{from_id}'}})-[r:{rel_type}]->(b {{entity_id: '{to_id}'}})
            RETURN count(r)
        $$) AS (cnt agtype);
    """
    try:
        await db.execute(text("LOAD 'age'"))
        await db.execute(text("SET search_path = ag_catalog, public"))
        result = await db.execute(text(sql))
        row = result.fetchone()
        if row:
            cnt = row[0]
            if isinstance(cnt, str):
                cnt = json.loads(cnt)
            return int(cnt) > 0
        return False
    except Exception:
        return False


def _escape_cypher(s: str) -> str:
    """Escape a string for Cypher property values."""
    if not s:
        return ""
    return s.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"')


async def sync_entity(
    db: AsyncSession,
    entity_id: str,
    canonical_name: str,
    entity_type: str,
    description: str = "",
    importance: float = 0.5,
    tenant_id: str = "",
) -> bool:
    """Sync a single entity to the AGE graph. Idempotent."""
    if await _node_exists(db, entity_id):
        # Update properties
        name_esc = _escape_cypher(canonical_name)
        desc_esc = _escape_cypher(description)
        cypher = (
            f"MATCH (n {{entity_id: '{entity_id}'}}) "
            f"SET n.name = '{name_esc}', "
            f"n.entity_type = '{entity_type}', "
            f"n.description = '{desc_esc}', "
            f"n.importance = {importance} "
            f"RETURN n"
        )
        return await _age_exec(db, cypher)

    # Create new node
    name_esc = _escape_cypher(canonical_name)
    desc_esc = _escape_cypher(description)
    cypher = (
        f"CREATE (n:Entity {{"
        f"entity_id: '{entity_id}', "
        f"name: '{name_esc}', "
        f"entity_type: '{entity_type}', "
        f"description: '{desc_esc}', "
        f"importance: {importance}, "
        f"tenant_id: '{tenant_id}'"
        f"}}) RETURN n"
    )
    return await _age_exec(db, cypher)


async def sync_relation(
    db: AsyncSession,
    from_entity_id: str,
    to_entity_id: str,
    relation_type: str,
    confidence: float = 0.7,
    weight: float = 1.0,
) -> bool:
    """Sync a relation to the AGE graph as an edge. Idempotent."""
    # Ensure both nodes exist
    if not await _node_exists(db, from_entity_id):
        logger.warning("Source node %s not in graph, skipping relation", from_entity_id)
        return False
    if not await _node_exists(db, to_entity_id):
        logger.warning("Target node %s not in graph, skipping relation", to_entity_id)
        return False

    if await _edge_exists(db, from_entity_id, to_entity_id, relation_type):
        return True  # Already exists

    cypher = (
        f"MATCH (a {{entity_id: '{from_entity_id}'}}), (b {{entity_id: '{to_entity_id}'}}) "
        f"CREATE (a)-[r:{relation_type} {{confidence: {confidence}, weight: {weight}}}]->(b) "
        f"RETURN r"
    )
    return await _age_exec(db, cypher)


async def sync_claim_node(
    db: AsyncSession,
    claim_id: str,
    summary_text: str,
    subject_entity_id: Optional[str] = None,
    object_entity_id: Optional[str] = None,
    confidence: float = 0.7,
    tenant_id: str = "",
) -> bool:
    """Sync a claim as a node in the graph, linked to subject/object entities."""
    if await _node_exists(db, claim_id):
        return True

    text_esc = _escape_cypher(summary_text[:200])
    cypher = (
        f"CREATE (c:Claim {{"
        f"entity_id: '{claim_id}', "
        f"summary: '{text_esc}', "
        f"confidence: {confidence}, "
        f"tenant_id: '{tenant_id}'"
        f"}}) RETURN c"
    )
    if not await _age_exec(db, cypher):
        return False

    # Link to subject entity
    if subject_entity_id and await _node_exists(db, subject_entity_id):
        await _age_exec(
            db,
            f"MATCH (s {{entity_id: '{subject_entity_id}'}}), (c {{entity_id: '{claim_id}'}}) "
            f"CREATE (c)-[r:ABOUT {{confidence: {confidence}}}]->(s) RETURN r",
        )

    # Link to object entity
    if object_entity_id and await _node_exists(db, object_entity_id):
        await _age_exec(
            db,
            f"MATCH (o {{entity_id: '{object_entity_id}'}}), (c {{entity_id: '{claim_id}'}}) "
            f"CREATE (c)-[r:REFERENCES {{confidence: {confidence}}}]->(o) RETURN r",
        )

    return True


async def sync_all_from_db(
    tenant_id: Optional[str] = None,
    batch_size: int = 100,
) -> dict:
    """Sync all entities, relations, and claims from relational DB to the graph.

    This is a bulk sync meant to be called periodically or after extraction.
    Idempotent — safe to re-run.
    """
    stats = {"entities_synced": 0, "relations_synced": 0, "claims_synced": 0, "errors": 0}

    async with AsyncSessionLocal() as db:
        # Sync entities
        where_clause = "WHERE owner_tenant_id = :tid" if tenant_id else ""
        params = {"tid": tenant_id} if tenant_id else {}

        result = await db.execute(
            text(f"""
                SELECT entity_id, canonical_name, entity_type,
                       description, importance_score, owner_tenant_id
                FROM entities
                {where_clause}
                ORDER BY importance_score DESC
                LIMIT :batch
            """),
            {**params, "batch": batch_size},
        )
        entities = result.fetchall()

        for ent in entities:
            ok = await sync_entity(
                db,
                entity_id=ent[0],
                canonical_name=ent[1],
                entity_type=ent[2],
                description=ent[3] or "",
                importance=ent[4],
                tenant_id=ent[5],
            )
            if ok:
                stats["entities_synced"] += 1
            else:
                stats["errors"] += 1

        await db.commit()

        # Sync relations
        result = await db.execute(
            text(f"""
                SELECT from_node_id, to_node_id, relation_type,
                       confidence, weight
                FROM relations
                {where_clause}
                LIMIT :batch
            """),
            {**params, "batch": batch_size},
        )
        relations = result.fetchall()

        for rel in relations:
            ok = await sync_relation(
                db,
                from_entity_id=rel[0],
                to_entity_id=rel[1],
                relation_type=rel[2],
                confidence=rel[3],
                weight=rel[4],
            )
            if ok:
                stats["relations_synced"] += 1
            else:
                stats["errors"] += 1

        await db.commit()

        # Sync claims
        result = await db.execute(
            text(f"""
                SELECT claim_id, summary_text, subject_entity_id,
                       object_entity_id, confidence, owner_tenant_id
                FROM claims
                {where_clause}
                LIMIT :batch
            """),
            {**params, "batch": batch_size},
        )
        claims = result.fetchall()

        for clm in claims:
            ok = await sync_claim_node(
                db,
                claim_id=clm[0],
                summary_text=clm[1] or "",
                subject_entity_id=clm[2],
                object_entity_id=clm[3],
                confidence=clm[4],
                tenant_id=clm[5],
            )
            if ok:
                stats["claims_synced"] += 1
            else:
                stats["errors"] += 1

        await db.commit()

    logger.info("Graph sync complete: %s", stats)
    return stats
