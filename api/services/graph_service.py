"""Graph service for GAMI using Apache AGE.

Provides Cypher-based graph queries over entities, relations, and claims
stored in the gami_graph AGE graph within PostgreSQL.
"""
import json
import logging
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from api.services.db import AsyncSessionLocal

logger = logging.getLogger("gami.services.graph_service")

GRAPH_NAME = "gami_graph"


async def _age_query(
    db: AsyncSession,
    cypher: str,
    columns: list[str],
) -> list[dict]:
    """Execute a Cypher query via AGE and return results as dicts.

    Args:
        db: Async database session.
        cypher: Cypher query string (no dollar-quoting needed).
        columns: Expected column names and AGE types, e.g. ["name agtype", "id agtype"].

    Returns:
        List of dicts with column names as keys.
    """
    col_names = [c.split()[0] for c in columns]
    col_spec = ", ".join(columns)

    sql = f"""
        SELECT * FROM cypher('{GRAPH_NAME}', $$
            {cypher}
        $$) AS ({col_spec});
    """

    try:
        # AGE requires LOAD and search_path to be set
        await db.execute(text("LOAD 'age'"))
        await db.execute(text("SET search_path = ag_catalog, public"))
        result = await db.execute(text(sql))
        rows = result.fetchall()
    except Exception as exc:
        logger.error("AGE query failed: %s\nCypher: %s", exc, cypher)
        return []

    results = []
    for row in rows:
        d = {}
        for i, col_name in enumerate(col_names):
            val = row[i]
            # AGE returns agtype strings — try to parse as JSON
            if isinstance(val, str):
                try:
                    val = json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    pass
            d[col_name] = val
        results.append(d)

    return results


async def get_neighborhood(
    entity_id: str,
    depth: int = 2,
    limit: int = 20,
    db: Optional[AsyncSession] = None,
) -> dict:
    """Get the neighborhood of an entity in the graph.

    Returns the entity, its direct neighbors, and the edges between them,
    up to the specified depth.
    """
    close_session = False
    if db is None:
        db = AsyncSessionLocal()
        close_session = True

    try:
        depth = min(depth, 3)  # Cap depth to avoid explosion

        # Get center node
        center_results = await _age_query(
            db,
            f"MATCH (n {{entity_id: '{entity_id}'}}) RETURN n",
            ["n agtype"],
        )

        # Get neighbors up to depth
        neighbors_results = await _age_query(
            db,
            f"""MATCH (n {{entity_id: '{entity_id}'}})-[r*1..{depth}]-(m)
                RETURN DISTINCT m, r
                LIMIT {limit}""",
            ["m agtype", "r agtype"],
        )

        # Get edges
        edges_results = await _age_query(
            db,
            f"""MATCH (n {{entity_id: '{entity_id}'}})-[r]-(m)
                RETURN n.entity_id, type(r), m.entity_id, r
                LIMIT {limit * 2}""",
            ["from_id agtype", "rel_type agtype", "to_id agtype", "r agtype"],
        )

        return {
            "center": center_results[0] if center_results else None,
            "neighbors": [r.get("m") for r in neighbors_results],
            "edges": [
                {
                    "from_id": e.get("from_id"),
                    "relation_type": e.get("rel_type"),
                    "to_id": e.get("to_id"),
                    "properties": e.get("r"),
                }
                for e in edges_results
            ],
            "node_count": len(neighbors_results),
            "edge_count": len(edges_results),
        }
    finally:
        if close_session:
            await db.close()


async def find_path(
    from_entity_id: str,
    to_entity_id: str,
    max_depth: int = 5,
    db: Optional[AsyncSession] = None,
) -> dict:
    """Find the shortest path between two entities in the graph."""
    close_session = False
    if db is None:
        db = AsyncSessionLocal()
        close_session = True

    try:
        max_depth = min(max_depth, 6)

        results = await _age_query(
            db,
            f"""MATCH p = shortestPath(
                    (a {{entity_id: '{from_entity_id}'}})-[*..{max_depth}]-
                    (b {{entity_id: '{to_entity_id}'}})
                )
                RETURN nodes(p), relationships(p), length(p)""",
            ["nodes agtype", "rels agtype", "path_length agtype"],
        )

        if not results:
            return {
                "found": False,
                "from_entity_id": from_entity_id,
                "to_entity_id": to_entity_id,
                "nodes": [],
                "edges": [],
                "path_length": -1,
            }

        path = results[0]
        return {
            "found": True,
            "from_entity_id": from_entity_id,
            "to_entity_id": to_entity_id,
            "nodes": path.get("nodes", []),
            "edges": path.get("rels", []),
            "path_length": path.get("path_length", 0),
        }
    finally:
        if close_session:
            await db.close()


async def get_cluster(
    cluster_id: str,
    db: Optional[AsyncSession] = None,
) -> dict:
    """Get a topic cluster with its members from the relational tables.

    Clusters are stored in the 'clusters' table, not in the AGE graph,
    since they are aggregation constructs.
    """
    close_session = False
    if db is None:
        db = AsyncSessionLocal()
        close_session = True

    try:
        result = await db.execute(
            text("""
                SELECT cluster_id, owner_tenant_id, cluster_type,
                       canonical_node_id, member_ids, summary_id,
                       confidence, status, created_at
                FROM clusters
                WHERE cluster_id = :cid
            """),
            {"cid": cluster_id},
        )
        row = result.fetchone()
        if not row:
            return {"found": False, "cluster_id": cluster_id}

        member_ids = row[4] if isinstance(row[4], list) else json.loads(row[4]) if row[4] else []

        # Fetch member entities
        members = []
        if member_ids:
            placeholders = ", ".join(f"'{mid}'" for mid in member_ids[:50])
            ent_result = await db.execute(
                text(f"""
                    SELECT entity_id, canonical_name, entity_type,
                           description, importance_score
                    FROM entities
                    WHERE entity_id IN ({placeholders})
                """),
            )
            for er in ent_result.fetchall():
                members.append({
                    "entity_id": er[0],
                    "name": er[1],
                    "type": er[2],
                    "description": er[3],
                    "importance": er[4],
                })

        # Fetch summary if exists
        summary_text = None
        if row[5]:
            sum_result = await db.execute(
                text("SELECT summary_text FROM summaries WHERE summary_id = :sid"),
                {"sid": row[5]},
            )
            sum_row = sum_result.fetchone()
            if sum_row:
                summary_text = sum_row[0]

        return {
            "found": True,
            "cluster_id": row[0],
            "owner_tenant_id": row[1],
            "cluster_type": row[2],
            "canonical_node_id": row[3],
            "member_count": len(member_ids),
            "members": members,
            "summary": summary_text,
            "confidence": row[6],
            "status": row[7],
            "created_at": row[8].isoformat() if row[8] else None,
        }
    finally:
        if close_session:
            await db.close()


async def get_entity_by_name(
    name: str,
    tenant_id: str = "claude-opus",
    db: Optional[AsyncSession] = None,
) -> Optional[dict]:
    """Look up an entity by canonical name (case-insensitive)."""
    close_session = False
    if db is None:
        db = AsyncSessionLocal()
        close_session = True

    try:
        result = await db.execute(
            text("""
                SELECT entity_id, canonical_name, entity_type,
                       description, importance_score, mention_count
                FROM entities
                WHERE owner_tenant_id = :tid
                  AND LOWER(canonical_name) = LOWER(:name)
                LIMIT 1
            """),
            {"tid": tenant_id, "name": name},
        )
        row = result.fetchone()
        if not row:
            return None
        return {
            "entity_id": row[0],
            "name": row[1],
            "type": row[2],
            "description": row[3],
            "importance": row[4],
            "mention_count": row[5],
        }
    finally:
        if close_session:
            await db.close()
