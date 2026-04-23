#!/usr/bin/env python3
"""GAMI MCP Server — exposes all GAMI operations as MCP tools.

Provides memory recall, search, remember, forget, update, cite, verify,
context lookup, source ingestion, graph exploration, and admin stats
via the Model Context Protocol.

Usage:
    python -m mcp.server                    # stdio transport (default)
    python -m mcp.server --transport sse    # SSE transport on port 9001
"""
import asyncio
import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

# Ensure GAMI root is in path
GAMI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if GAMI_ROOT not in sys.path:
    sys.path.insert(0, GAMI_ROOT)

# mcp package (installed via pip) — no namespace conflict since this
# directory is mcp_tools/, not mcp/
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from api.config import settings
from mcp_tools.tool_definitions import TOOL_DEFINITIONS

logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL))
logger = logging.getLogger("gami.mcp")

# Create MCP server
server = Server("gami-memory")


# ---------------------------------------------------------------------------
# Tool listing
# ---------------------------------------------------------------------------

@server.list_tools()
async def list_tools() -> list[Tool]:
    """Return all available GAMI tools."""
    tools = []
    for name, defn in TOOL_DEFINITIONS.items():
        tools.append(
            Tool(
                name=defn["name"],
                description=defn["description"],
                inputSchema=defn["inputSchema"],
            )
        )
    return tools


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Dispatch tool calls to GAMI operations."""
    import time

    start_time = time.perf_counter()
    success = False
    error_type = None

    try:
        handler = TOOL_HANDLERS.get(name)
        if not handler:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

        result = await handler(arguments)
        success = True
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

    except Exception as exc:
        logger.error("Tool %s failed: %s", name, exc, exc_info=True)
        error_type = type(exc).__name__
        return [TextContent(
            type="text",
            text=json.dumps({"error": str(exc), "tool": name}),
        )]

    finally:
        latency = time.perf_counter() - start_time
        # Track MCP tool metrics
        try:
            from manifold.metrics import track_mcp_tool
            track_mcp_tool(
                tool=name,
                latency=latency,
                success=success,
                error_type=error_type,
            )
        except Exception as metric_err:
            logger.debug(f"Metrics tracking failed: {metric_err}")

        # Log slow tool calls
        if latency > 5.0:
            logger.warning(f"Slow MCP tool: {name} took {latency:.2f}s")


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

async def _memory_recall(args: dict) -> dict:
    """Recall memories with token budget — searches memories, entities, claims, AND segments."""
    from api.services.retrieval import recall

    query = args["query"]
    tenant_id = args.get("tenant_id", "claude-opus")
    tenant_ids = args.get("tenant_ids")
    max_tokens = args.get("max_tokens", 2000)
    session_id = args.get("session_id")
    agent_id = args.get("agent_id", "mcp-client")

    # Default to 'factual' to skip slow Ollama classification (~30s).
    # Claude already knows what it's looking for; override with mode param if needed.
    mode = args.get("mode", "factual")

    # Phase 5: Bi-temporal query support
    event_after = args.get("event_after")
    event_before = args.get("event_before")
    ingested_after = args.get("ingested_after")
    ingested_before = args.get("ingested_before")

    # Phase 3: Compression detail level
    detail_level = args.get("detail_level", "normal")

    result = await recall(
        query=query,
        tenant_id=tenant_id,
        tenant_ids=tenant_ids,
        max_tokens=max_tokens,
        mode=mode,
        session_id=session_id,
        agent_id=agent_id,
        event_after=event_after,
        event_before=event_before,
        ingested_after=ingested_after,
        ingested_before=ingested_before,
        detail_level=detail_level,
    )

    # Format for MCP response
    citations = []
    for ev in result.evidence:
        citations.append({
            "segment_id": ev.item_id,
            "source_id": ev.metadata.get("source_id", ev.metadata.get("source_type", "")),
            "item_type": ev.item_type,
            "score": ev.effective_score,
        })

    response = {
        "context": result.context_text,
        "citations": citations,
        "total_tokens": result.total_tokens_used,
        "results_used": len(result.evidence),
        "mode": result.mode,
        "search_ms": result.search_ms,
    }

    # Phase 7: Include contradiction info
    if result.has_contradictions:
        response["has_contradictions"] = True
        response["needs_resolution"] = result.needs_resolution
        response["contradictions"] = [
            {
                "group_id": c.group_id,
                "predicate": c.predicate,
                "claim_ids": c.claim_ids,
                "values": c.values,
                "confidences": c.confidences,
                "status": c.status,
                "proposal_id": c.proposal_id,
            }
            for c in result.contradictions
        ]

    return response


async def _memory_remember(args: dict) -> dict:
    """Store a new assistant memory with Mem0-style consolidation.

    If consolidation is enabled (default), classifies the memory as:
    - ADD: Store as new memory
    - UPDATE: Update existing similar memory
    - NOOP: Skip storage (duplicate detected)
    """
    from api.llm.embeddings import embed_text
    from api.services.db import AsyncSessionLocal
    from api.config import settings

    text_content = args["text"]
    memory_type = args.get("memory_type", "fact")
    subject_id = args.get("subject_id", "general")
    tenant_id = args.get("tenant_id", "claude-opus")
    importance = args.get("importance", 0.5)
    skip_consolidation = args.get("skip_consolidation", False)
    agent_id = args.get("agent_id")

    async with AsyncSessionLocal() as db:
        from sqlalchemy import text as sql_text

        # Memory consolidation (if enabled)
        if settings.MEMORY_CONSOLIDATION_ENABLED and not skip_consolidation:
            try:
                from api.services.memory_classifier import (
                    MemoryOperationClassifier,
                    OperationType,
                    log_memory_operation,
                )

                classifier = MemoryOperationClassifier()
                operation = await classifier.classify(
                    text_content, tenant_id, db, memory_type, subject_id
                )

                # Handle NOOP (duplicate)
                if operation.type == OperationType.NOOP:
                    await log_memory_operation(
                        db, operation, tenant_id, agent_id
                    )
                    await db.commit()
                    return {
                        "status": "noop",
                        "reason": operation.reason,
                        "existing_memory_id": operation.target_id,
                        "similarity": operation.similarity,
                        "tenant_id": tenant_id,
                    }

                # Handle UPDATE (supersede existing)
                if operation.type == OperationType.UPDATE:
                    # Use existing update logic
                    result = await _memory_update({
                        "memory_id": operation.target_id,
                        "new_text": text_content,
                        "reason": operation.reason,
                    })
                    await log_memory_operation(
                        db, operation, tenant_id, agent_id,
                        result_memory_id=result.get("new_memory_id")
                    )
                    await db.commit()
                    return {
                        "status": "updated",
                        "reason": operation.reason,
                        "old_memory_id": operation.target_id,
                        "new_memory_id": result.get("new_memory_id"),
                        "similarity": operation.similarity,
                        "tenant_id": tenant_id,
                    }

                # Log ADD operation (will be executed below)
                await log_memory_operation(
                    db, operation, tenant_id, agent_id
                )

            except Exception as e:
                logger.warning(f"Memory consolidation failed: {e}, proceeding with ADD")

        # Standard ADD flow
        embedding = await embed_text(text_content)
        vec_str = "[" + ",".join(str(v) for v in embedding) + "]"

        memory_id = f"MEM_{uuid.uuid4().hex[:16]}"
        now = datetime.now(timezone.utc)

        await db.execute(
            sql_text(
                "INSERT INTO assistant_memories "
                "(memory_id, owner_tenant_id, memory_type, subject_id, "
                "normalized_text, embedding, importance_score, "
                "status, created_at, updated_at) "
                "VALUES (:mid, :tid, :mtype, :subj, :txt, "
                "CAST(:vec AS vector), :imp, 'active', :now, :now)"
            ),
            {
                "mid": memory_id,
                "tid": tenant_id,
                "mtype": memory_type,
                "subj": subject_id,
                "txt": text_content,
                "vec": vec_str,
                "imp": importance,
                "now": now,
            },
        )
        await db.commit()

    return {
        "memory_id": memory_id,
        "status": "stored",
        "memory_type": memory_type,
        "tenant_id": tenant_id,
    }


async def _memory_forget(args: dict) -> dict:
    """Mark a memory as archived."""
    from api.services.db import AsyncSessionLocal
    from sqlalchemy import text as sql_text

    memory_id = args["memory_id"]
    reason = args.get("reason", "user_requested")
    now = datetime.now(timezone.utc)

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            sql_text(
                "UPDATE assistant_memories SET status = 'archived', "
                "updated_at = :now WHERE memory_id = :mid AND status != 'archived' "
                "RETURNING memory_id"
            ),
            {"mid": memory_id, "now": now},
        )
        row = result.fetchone()
        await db.commit()

    if not row:
        return {"status": "not_found", "memory_id": memory_id}

    return {"status": "archived", "memory_id": memory_id, "reason": reason}


async def _memory_update(args: dict) -> dict:
    """Update a memory by creating a new version."""
    from api.llm.embeddings import embed_text
    from api.services.db import AsyncSessionLocal
    from sqlalchemy import text as sql_text

    old_id = args["memory_id"]
    new_text = args["new_text"]
    reason = args.get("reason", "correction")

    # Embed new text
    embedding = await embed_text(new_text)
    vec_str = "[" + ",".join(str(v) for v in embedding) + "]"

    now = datetime.now(timezone.utc)
    new_id = f"MEM_{uuid.uuid4().hex[:16]}"

    async with AsyncSessionLocal() as db:
        # Fetch old memory
        old = await db.execute(
            sql_text(
                "SELECT memory_id, owner_tenant_id, memory_type, subject_id, "
                "importance_score, version FROM assistant_memories "
                "WHERE memory_id = :mid"
            ),
            {"mid": old_id},
        )
        old_row = old.fetchone()

        if not old_row:
            return {"status": "not_found", "memory_id": old_id}

        # Create new version
        await db.execute(
            sql_text(
                "INSERT INTO assistant_memories "
                "(memory_id, owner_tenant_id, memory_type, subject_id, "
                "normalized_text, embedding, importance_score, "
                "status, version, created_at, updated_at) "
                "VALUES (:mid, :tid, :mtype, :subj, :txt, "
                "CAST(:vec AS vector), :imp, 'active', :ver, :now, :now)"
            ),
            {
                "mid": new_id,
                "tid": old_row[1],
                "mtype": old_row[2],
                "subj": old_row[3],
                "txt": new_text,
                "vec": vec_str,
                "imp": old_row[4],
                "ver": (old_row[5] or 1) + 1,
                "now": now,
            },
        )

        # Supersede old memory
        await db.execute(
            sql_text(
                "UPDATE assistant_memories SET status = 'superseded', "
                "superseded_by_id = :new_id, updated_at = :now "
                "WHERE memory_id = :old_id"
            ),
            {"new_id": new_id, "old_id": old_id, "now": now},
        )

        await db.commit()

    return {
        "status": "updated",
        "old_memory_id": old_id,
        "new_memory_id": new_id,
        "reason": reason,
    }


async def _memory_cite(args: dict) -> dict:
    """Get provenance for a target object."""
    from api.services.db import AsyncSessionLocal
    from sqlalchemy import text as sql_text

    target_id = args["target_id"]
    target_type = args.get("target_type", "memory")

    # Map target_type to provenance target_type values
    prov_type_map = {
        "memory": "memory",
        "claim": "claim",
        "entity": "entity",
        "event": "event",
    }
    prov_type = prov_type_map.get(target_type, target_type)

    async with AsyncSessionLocal() as db:
        rows = await db.execute(
            sql_text(
                "SELECT p.provenance_id, p.source_id, p.segment_id, "
                "       p.extraction_method, p.confidence, p.created_at, "
                "       s.title, s.source_type, s.source_uri "
                "FROM provenance p "
                "LEFT JOIN sources s ON p.source_id = s.source_id "
                "WHERE p.target_id = :tid AND p.target_type = :ttype "
                "ORDER BY p.created_at DESC"
            ),
            {"tid": target_id, "ttype": prov_type},
        )
        results = rows.fetchall()

    return {
        "target_id": target_id,
        "target_type": target_type,
        "citations": [
            {
                "provenance_id": r[0],
                "source_id": r[1],
                "segment_id": r[2],
                "extraction_method": r[3],
                "confidence": r[4],
                "created_at": r[5].isoformat() if r[5] else None,
                "source_title": r[6],
                "source_type": r[7],
                "source_uri": r[8],
            }
            for r in results
        ],
        "total": len(results),
    }


async def _memory_verify(args: dict) -> dict:
    """Verify a statement against known claims."""
    from api.llm.embeddings import embed_text
    from api.search.hybrid import hybrid_search
    from api.services.db import AsyncSessionLocal
    from sqlalchemy import text as sql_text

    statement = args["statement"]
    tenant_id = args.get("tenant_id", "claude-opus")

    # Search for related claims
    embedding = await embed_text(statement)
    vec_str = "[" + ",".join(str(v) for v in embedding) + "]"

    async with AsyncSessionLocal() as db:
        # Vector search against claims
        claim_rows = await db.execute(
            sql_text(
                "SELECT claim_id, summary_text, confidence, modality, "
                "       contradiction_group_id, "
                "       1 - (embedding <=> CAST(:vec AS vector)) AS similarity "
                "FROM claims "
                "WHERE owner_tenant_id = :tid "
                "  AND embedding IS NOT NULL "
                "  AND status = 'active' "
                "ORDER BY embedding <=> CAST(:vec AS vector) "
                "LIMIT 10"
            ),
            {"vec": vec_str, "tid": tenant_id},
        )
        claims = claim_rows.fetchall()

    supporting = []
    contradicting = []
    related = []

    for c in claims:
        claim_data = {
            "claim_id": c[0],
            "summary": c[1],
            "confidence": c[2],
            "modality": c[3],
            "similarity": float(c[5]),
        }

        if c[5] > 0.85:
            if c[3] == "negated":
                contradicting.append(claim_data)
            else:
                supporting.append(claim_data)
        elif c[5] > 0.6:
            related.append(claim_data)

    has_contradiction = bool(contradicting)
    has_support = bool(supporting)

    if has_support and not has_contradiction:
        verdict = "supported"
    elif has_contradiction and not has_support:
        verdict = "contradicted"
    elif has_support and has_contradiction:
        verdict = "contested"
    else:
        verdict = "unverified"

    return {
        "statement": statement,
        "verdict": verdict,
        "supporting_claims": supporting,
        "contradicting_claims": contradicting,
        "related_claims": related,
    }


async def _memory_search(args: dict) -> dict:
    """Hybrid search over memory segments."""
    from api.llm.embeddings import embed_text
    from api.search.hybrid import hybrid_search, lexical_search, vector_search
    from api.services.db import AsyncSessionLocal

    query = args["query"]
    tenant_ids = args.get("tenant_ids") or [args.get("tenant_id", "claude-opus")]
    limit = args.get("limit", 20)
    mode = args.get("search_mode", "hybrid")

    query_embedding = None
    if mode != "lexical":
        query_embedding = await embed_text(query)

    async with AsyncSessionLocal() as db:
        if mode == "lexical":
            results = await lexical_search(db, query, tenant_ids, limit=limit)
        elif mode == "vector":
            results = await vector_search(db, query_embedding, tenant_ids, limit=limit)
        else:
            results = await hybrid_search(
                db, query, query_embedding, tenant_ids, limit=limit
            )

    return {
        "query": query,
        "results": [
            {
                "segment_id": r["segment_id"],
                "text": r["text"][:500],
                "source_id": r["source_id"],
                "segment_type": r["segment_type"],
                "score": r.get("combined_score") or r.get("similarity") or r.get("rank", 0),
            }
            for r in results
        ],
        "total": len(results),
    }


async def _memory_context(args: dict) -> dict:
    """Get context for an entity."""
    from api.services.db import AsyncSessionLocal
    from sqlalchemy import text as sql_text

    entity_id = args.get("entity_id")
    entity_name = args.get("entity_name")
    tenant_id = args.get("tenant_id", "claude-opus")

    async with AsyncSessionLocal() as db:
        # Resolve entity
        if entity_name and not entity_id:
            row = await db.execute(
                sql_text(
                    "SELECT entity_id FROM entities "
                    "WHERE owner_tenant_id = :tid "
                    "AND LOWER(canonical_name) = :name "
                    "AND status != 'archived' "
                    "LIMIT 1"
                ),
                {"tid": tenant_id, "name": entity_name.lower()},
            )
            found = row.fetchone()
            if found:
                entity_id = found[0]

        if not entity_id:
            return {"error": "Entity not found", "query": entity_name or "none"}

        # Fetch entity details
        ent_row = await db.execute(
            sql_text(
                "SELECT entity_id, canonical_name, entity_type, description, "
                "       importance_score, mention_count, source_count "
                "FROM entities WHERE entity_id = :eid"
            ),
            {"eid": entity_id},
        )
        entity = ent_row.fetchone()

        if not entity:
            return {"error": "Entity not found", "entity_id": entity_id}

        result = {
            "entity_id": entity[0],
            "name": entity[1],
            "type": entity[2],
            "description": entity[3],
            "importance": entity[4],
            "mentions": entity[5],
            "sources": entity[6],
        }

        # Claims
        if args.get("include_claims", True):
            claims = await db.execute(
                sql_text(
                    "SELECT claim_id, predicate, summary_text, confidence "
                    "FROM claims WHERE status = 'active' "
                    "AND (subject_entity_id = :eid OR object_entity_id = :eid) "
                    "ORDER BY confidence DESC LIMIT 20"
                ),
                {"eid": entity_id},
            )
            result["claims"] = [
                {
                    "claim_id": c[0],
                    "predicate": c[1],
                    "summary": c[2],
                    "confidence": c[3],
                }
                for c in claims.fetchall()
            ]

        # Relations
        if args.get("include_relations", True):
            rels = await db.execute(
                sql_text(
                    "SELECT r.relation_type, r.confidence, "
                    "       CASE WHEN r.from_node_id = :eid THEN r.to_node_id "
                    "            ELSE r.from_node_id END AS other_id, "
                    "       CASE WHEN r.from_node_id = :eid THEN 'outgoing' "
                    "            ELSE 'incoming' END AS direction "
                    "FROM relations r "
                    "WHERE r.status = 'active' "
                    "AND (r.from_node_id = :eid OR r.to_node_id = :eid) "
                    "ORDER BY r.confidence DESC LIMIT 30"
                ),
                {"eid": entity_id},
            )
            rel_list = rels.fetchall()

            # Resolve names
            other_ids = list({r[2] for r in rel_list})
            name_map = {}
            if other_ids:
                names = await db.execute(
                    sql_text(
                        "SELECT entity_id, canonical_name FROM entities "
                        "WHERE entity_id = ANY(:ids)"
                    ),
                    {"ids": other_ids},
                )
                name_map = {n[0]: n[1] for n in names.fetchall()}

            result["relations"] = [
                {
                    "relation_type": r[0],
                    "confidence": r[1],
                    "other_entity": name_map.get(r[2], r[2]),
                    "other_id": r[2],
                    "direction": r[3],
                }
                for r in rel_list
            ]

    return result


async def _ingest_source(args: dict) -> dict:
    """Ingest a file via the API."""
    import httpx

    file_path = args["file_path"]
    source_type = args.get("source_type", "markdown")
    tenant_id = args.get("tenant_id", "shared")
    title = args.get("title")

    data = {
        "file_path": file_path,
        "source_type": source_type,
        "tenant_id": tenant_id,
        "metadata_json": "{}",
    }
    if title:
        data["title"] = title

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"http://127.0.0.1:{settings.API_PORT}/api/v1/ingest/source",
            data=data,
            timeout=120.0,
        )
        resp.raise_for_status()
        return resp.json()


async def _graph_explore(args: dict) -> dict:
    """Explore the knowledge graph from an entity."""
    from api.services.db import AsyncSessionLocal
    from sqlalchemy import text as sql_text

    entity_id = args.get("entity_id")
    entity_name = args.get("entity_name")
    tenant_id = args.get("tenant_id", "claude-opus")
    depth = min(args.get("depth", 2), 4)
    limit = min(args.get("limit", 50), 200)
    rel_filter = args.get("relation_types")

    async with AsyncSessionLocal() as db:
        # Resolve name to ID
        if entity_name and not entity_id:
            row = await db.execute(
                sql_text(
                    "SELECT entity_id FROM entities "
                    "WHERE owner_tenant_id = :tid "
                    "AND LOWER(canonical_name) = :name "
                    "AND status != 'archived' LIMIT 1"
                ),
                {"tid": tenant_id, "name": entity_name.lower()},
            )
            found = row.fetchone()
            if found:
                entity_id = found[0]

        if not entity_id:
            return {"error": "Entity not found"}

        nodes = {}
        edges = []
        frontier = {entity_id}

        for hop in range(1, depth + 1):
            if not frontier or len(nodes) >= limit:
                break

            frontier_list = list(frontier)
            next_frontier = set()

            # Build relation query
            rel_where = ""
            params = {"ids": frontier_list, "tid": tenant_id}
            if rel_filter:
                rel_where = "AND r.relation_type = ANY(:rtypes) "
                params["rtypes"] = rel_filter

            rels = await db.execute(
                sql_text(
                    f"SELECT r.relation_id, r.relation_type, r.confidence, "
                    f"       r.from_node_id, r.to_node_id "
                    f"FROM relations r "
                    f"WHERE r.owner_tenant_id = :tid "
                    f"  AND r.status = 'active' "
                    f"  AND (r.from_node_id = ANY(:ids) OR r.to_node_id = ANY(:ids)) "
                    f"  {rel_where}"
                    f"LIMIT 200"
                ),
                params,
            )

            for r in rels.fetchall():
                edges.append({
                    "relation_id": r[0],
                    "type": r[1],
                    "confidence": r[2],
                    "from": r[3],
                    "to": r[4],
                })
                for nid in (r[3], r[4]):
                    if nid not in nodes:
                        next_frontier.add(nid)

            # Add discovered nodes
            if next_frontier:
                new_ids = list(next_frontier - set(nodes.keys()))
                if new_ids:
                    ent_rows = await db.execute(
                        sql_text(
                            "SELECT entity_id, canonical_name, entity_type, "
                            "       importance_score "
                            "FROM entities WHERE entity_id = ANY(:ids)"
                        ),
                        {"ids": new_ids},
                    )
                    for e in ent_rows.fetchall():
                        nodes[e[0]] = {
                            "entity_id": e[0],
                            "name": e[1],
                            "type": e[2],
                            "importance": e[3],
                            "hop": hop,
                        }

            frontier = next_frontier - set(nodes.keys())

        # Also add frontier entities
        for eid in frontier_list:
            if eid not in nodes:
                ent = await db.execute(
                    sql_text(
                        "SELECT entity_id, canonical_name, entity_type, importance_score "
                        "FROM entities WHERE entity_id = :eid"
                    ),
                    {"eid": eid},
                )
                row = ent.fetchone()
                if row:
                    nodes[row[0]] = {
                        "entity_id": row[0],
                        "name": row[1],
                        "type": row[2],
                        "importance": row[3],
                        "hop": 0,
                    }

    return {
        "root_entity_id": entity_id,
        "nodes": list(nodes.values())[:limit],
        "edges": edges[:limit * 2],
        "total_nodes": len(nodes),
        "total_edges": len(edges),
    }


async def _admin_stats(args: dict) -> dict:
    """Get system statistics."""
    from api.services.db import AsyncSessionLocal
    from sqlalchemy import text as sql_text

    tenant_id = args.get("tenant_id")

    async with AsyncSessionLocal() as db:
        if tenant_id:
            where = "WHERE owner_tenant_id = :tid"
            params = {"tid": tenant_id}
        else:
            where = ""
            params = {}

        counts = {}
        for table, label in [
            ("segments", "segments"),
            ("entities", "entities"),
            ("claims", "claims"),
            ("relations", "relations"),
            ("events", "events"),
            ("sources", "sources"),
            ("assistant_memories", "memories"),
        ]:
            try:
                col = "owner_tenant_id" if table != "relations" else "owner_tenant_id"
                q = f"SELECT COUNT(*) FROM {table} {where}"
                row = await db.execute(sql_text(q), params)
                counts[label] = row.scalar()
            except Exception:
                counts[label] = 0

        # Storage tier breakdown
        tier_counts = {}
        for table in ["segments", "entities", "claims", "relations"]:
            try:
                rows = await db.execute(
                    sql_text(
                        f"SELECT storage_tier, COUNT(*) FROM {table} "
                        f"{where} GROUP BY storage_tier"
                    ),
                    params,
                )
                tier_counts[table] = {r[0]: r[1] for r in rows.fetchall()}
            except Exception:
                tier_counts[table] = {}

        # Tenant count
        tenant_row = await db.execute(sql_text("SELECT COUNT(*) FROM tenants"))
        counts["tenants"] = tenant_row.scalar()

        # Pending proposals
        prop_row = await db.execute(
            sql_text("SELECT COUNT(*) FROM proposed_changes WHERE status = 'pending'")
        )
        counts["pending_proposals"] = prop_row.scalar()

    return {
        "counts": counts,
        "storage_tiers": tier_counts,
        "tenant_filter": tenant_id,
    }


async def _get_unprocessed_segments(args: dict) -> dict:
    """Get segments that haven't been processed for entity extraction yet."""
    from api.services.db import AsyncSessionLocal
    from sqlalchemy import text as sql_text

    limit = args.get("limit", 20)
    tenant_id = args.get("tenant_id", "claude-opus")
    min_length = args.get("min_length", 200)
    max_length = args.get("max_length", 3000)

    async with AsyncSessionLocal() as db:
        rows = await db.execute(sql_text("""
            SELECT s.segment_id, s.text, s.source_id
            FROM segments s
            WHERE length(s.text) BETWEEN :minl AND :maxl
            AND s.owner_tenant_id = :tid
            AND s.segment_type NOT IN ('tool_call', 'tool_result', 'chunk')
            AND NOT EXISTS (
                SELECT 1 FROM provenance p WHERE p.segment_id = s.segment_id
            )
            ORDER BY s.created_at DESC
            LIMIT :lim
        """), {"tid": tenant_id, "minl": min_length, "maxl": max_length, "lim": limit})

        segments = []
        for row in rows:
            segments.append({
                "segment_id": row.segment_id,
                "text": row.text[:2000],
                "source_id": row.source_id,
            })

        # Also get total count
        total = await db.execute(sql_text("""
            SELECT count(*) FROM segments s
            WHERE length(s.text) BETWEEN :minl AND :maxl
            AND s.owner_tenant_id = :tid
            AND s.segment_type NOT IN ('tool_call', 'tool_result', 'chunk')
            AND NOT EXISTS (SELECT 1 FROM provenance p WHERE p.segment_id = s.segment_id)
        """), {"tid": tenant_id, "minl": min_length, "maxl": max_length})

    return {
        "segments": segments,
        "returned": len(segments),
        "total_unprocessed": total.scalar(),
    }


async def _store_extractions(args: dict) -> dict:
    """Store extracted entities and provenance from an agent's analysis."""
    import hashlib
    from api.services.db import AsyncSessionLocal
    from sqlalchemy import text as sql_text

    segment_id = args["segment_id"]
    source_id = args.get("source_id", "")
    entities = args.get("entities", [])
    tenant_id = args.get("tenant_id", "claude-opus")

    def gen_id(prefix, name):
        h = hashlib.md5(name.encode()).hexdigest()[:8]
        clean = "".join(c if c.isalnum() or c == '_' else '_' for c in name)[:30]
        return f"{prefix}_{clean}_{h}"

    stored = 0
    async with AsyncSessionLocal() as db:
        for ent in entities:
            name = ent.get("name", "").strip()
            etype = ent.get("type", "technology").lower()
            desc = ent.get("description", "")
            if not name or len(name) < 2:
                continue

            eid = gen_id("ENT", f"{etype}_{name}")
            await db.execute(sql_text("""
                INSERT INTO entities (entity_id, owner_tenant_id, entity_type, canonical_name,
                    description, status, first_seen_at, last_seen_at, source_count, mention_count)
                VALUES (:eid, :tid, :etype, :name, :desc, 'active', NOW(), NOW(), 1, 1)
                ON CONFLICT (entity_id) DO UPDATE SET
                    mention_count = entities.mention_count + 1, last_seen_at = NOW()
            """), {"eid": eid, "tid": tenant_id, "etype": etype, "name": name, "desc": desc})

            prov_id = gen_id("PROV", f"{eid}_{segment_id}")
            await db.execute(sql_text("""
                INSERT INTO provenance (provenance_id, target_type, target_id, source_id,
                    segment_id, extraction_method, extractor_version, confidence)
                VALUES (:pid, 'entity', :eid, :src, :seg, 'agent_extract', 'haiku_v1', 0.85)
                ON CONFLICT (provenance_id) DO NOTHING
            """), {"pid": prov_id, "eid": eid, "src": source_id, "seg": segment_id})
            stored += 1

        await db.commit()

    return {"stored": stored, "segment_id": segment_id}


async def _ingest_file(args: dict) -> dict:
    """Ingest a file into GAMI from a local path."""
    import subprocess

    file_path = args["file_path"]
    tenant_id = args.get("tenant_id", "claude-opus")
    source_type = args.get("source_type", "markdown")

    import os
    if not os.path.isfile(file_path):
        return {"status": "error", "message": f"File not found: {file_path}"}

    file_size = os.path.getsize(file_path)
    if file_size > 50 * 1024 * 1024:  # 50MB limit
        return {"status": "error", "message": f"File too large: {file_size} bytes (max 50MB)"}

    # Use the ingest API
    try:
        import httpx
        async with httpx.AsyncClient(timeout=120) as client:
            with open(file_path, "rb") as f:
                resp = await client.post(
                    "http://127.0.0.1:9090/api/v1/ingest/source",
                    files={"file": (os.path.basename(file_path), f)},
                    data={"tenant_id": tenant_id, "source_type": source_type},
                )
                if resp.status_code == 200:
                    result = resp.json()
                    return {
                        "status": "ingested",
                        "source_id": result.get("source_id", ""),
                        "segments": result.get("segment_count", 0),
                        "file": file_path,
                        "size": file_size,
                    }
                else:
                    # Fallback: direct parse + insert
                    pass
    except Exception:
        pass

    # Fallback: read file and insert directly
    try:
        with open(file_path, "r", errors="replace") as f:
            content = f.read()

        from api.services.db import AsyncSessionLocal
        from sqlalchemy import text as sql_text
        import hashlib
        import uuid

        source_id = f"SRC_{source_type.upper()}_{hashlib.md5(file_path.encode()).hexdigest()[:12]}"
        fname = os.path.basename(file_path)

        async with AsyncSessionLocal() as db:
            # Create source
            await db.execute(sql_text("""
                INSERT INTO sources (source_id, owner_tenant_id, source_type, title,
                    file_path, file_size_bytes, checksum, parse_status)
                VALUES (:sid, :tid, :stype, :title, :path, :size, :cksum, 'parsed')
                ON CONFLICT (source_id) DO NOTHING
            """), {
                "sid": source_id, "tid": tenant_id, "stype": source_type,
                "title": fname, "path": file_path, "size": file_size,
                "cksum": hashlib.md5(content[:10000].encode()).hexdigest(),
            })

            # Split into segments (simple paragraph split)
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip() and len(p.strip()) > 20]
            seg_count = 0
            for i, para in enumerate(paragraphs):
                seg_id = f"SEG_{source_id}_{i}"
                await db.execute(sql_text("""
                    INSERT INTO segments (segment_id, source_id, owner_tenant_id, text,
                        segment_type, ordinal, depth, token_count)
                    VALUES (:sid, :src, :tid, :txt, 'paragraph', :ord, 0, :tc)
                    ON CONFLICT (segment_id) DO NOTHING
                """), {
                    "sid": seg_id, "src": source_id, "tid": tenant_id,
                    "txt": para, "ord": i, "tc": len(para) // 4,
                })
                seg_count += 1

            await db.commit()

        return {
            "status": "ingested",
            "source_id": source_id,
            "segments": seg_count,
            "file": file_path,
            "size": file_size,
            "method": "direct_parse",
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def _dream_haiku(args: dict) -> dict:
    """Run dream-like knowledge synthesis using Haiku instead of vLLM.

    For machines without a GPU. Processes extraction, summarization,
    and entity resolution using Claude Code Haiku agent via OAuth billing.
    """
    import subprocess
    import shutil

    limit = args.get("limit", 50)
    phases = args.get("phases", "all")  # all, extract, summarize

    if not shutil.which("claude"):
        return {"status": "error", "message": "Claude CLI not found"}

    # Get unprocessed count
    from api.services.db import AsyncSessionLocal
    from sqlalchemy import text as sql_text
    async with AsyncSessionLocal() as db:
        row = await db.execute(sql_text(
            "SELECT count(*) FROM segments s "
            "WHERE length(s.text) BETWEEN 200 AND 3000 "
            "AND s.owner_tenant_id = 'claude-opus' "
            "AND s.segment_type NOT IN ('tool_call', 'tool_result', 'chunk') "
            "AND NOT EXISTS (SELECT 1 FROM provenance p WHERE p.segment_id = s.segment_id)"
        ))
        unprocessed = row.scalar()

    if unprocessed == 0:
        return {"status": "complete", "message": "All segments already processed", "unprocessed": 0}

    actual_limit = min(limit, unprocessed)

    # Launch process_segments.sh with the limit
    try:
        env = dict(__import__("os").environ)
        env["GAMI_LIMIT"] = str(actual_limit)
        proc = subprocess.Popen(
            ["/opt/gami/cli/process_segments.sh"],
            stdout=open("/tmp/gami-dream-haiku.log", "a"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )
        est_minutes = actual_limit * 12 / 60  # ~12s per segment
        return {
            "status": "started",
            "pid": proc.pid,
            "unprocessed": unprocessed,
            "processing": actual_limit,
            "estimated_minutes": round(est_minutes, 1),
            "message": f"Haiku dream started — processing {actual_limit} of {unprocessed} segments. "
                       f"ETA: ~{est_minutes:.0f} min. Rate: ~300 segments/hour. "
                       f"Monitor: tail -f /tmp/gami-dream-haiku.log",
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def _run_haiku_extraction(args: dict) -> dict:
    """Trigger the Haiku agent extraction pipeline.

    Spawns process_segments.sh in the background, which:
    1. Ensures Ollama is running for embeddings
    2. Launches a Claude Code Haiku agent to extract entities
    Uses OAuth billing — no API key needed, works without GPU.
    """
    import subprocess
    import shutil

    limit = args.get("limit", 10)
    script = "/opt/gami/cli/process_segments.sh"

    # Check prerequisites
    checks = {}

    # Check Ollama
    try:
        import requests as req
        r = req.get("http://localhost:11434/api/tags", timeout=3)
        checks["ollama"] = "running" if r.status_code == 200 else "down"
    except Exception:
        checks["ollama"] = "unreachable"

    # Check claude CLI
    checks["claude_cli"] = "found" if shutil.which("claude") else "not_found"

    # Check vLLM (to report GPU status)
    try:
        import requests as req
        r = req.get("http://localhost:8000/v1/models", timeout=3)
        checks["vllm_gpu"] = "running" if r.status_code == 200 else "down"
    except Exception:
        checks["vllm_gpu"] = "unavailable"

    # Check unprocessed count
    from api.services.db import AsyncSessionLocal
    from sqlalchemy import text as sql_text
    async with AsyncSessionLocal() as db:
        row = await db.execute(sql_text(
            "SELECT count(*) FROM segments s "
            "WHERE length(s.text) BETWEEN 200 AND 3000 "
            "AND s.owner_tenant_id = 'claude-opus' "
            "AND s.segment_type NOT IN ('tool_call', 'tool_result', 'chunk') "
            "AND NOT EXISTS (SELECT 1 FROM provenance p WHERE p.segment_id = s.segment_id)"
        ))
        checks["unprocessed_segments"] = row.scalar()

    if checks["claude_cli"] == "not_found":
        return {
            "status": "error",
            "message": "Claude CLI not found. Install Claude Code to use Haiku extraction.",
            "checks": checks,
        }

    if checks["unprocessed_segments"] == 0:
        return {
            "status": "skipped",
            "message": "No unprocessed segments to extract from.",
            "checks": checks,
        }

    # Launch the extraction script in the background
    try:
        proc = subprocess.Popen(
            [script],
            stdout=open("/tmp/gami-process-segments.log", "a"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        return {
            "status": "started",
            "pid": proc.pid,
            "message": f"Haiku extraction pipeline started (PID {proc.pid}). "
                       f"{checks['unprocessed_segments']} segments to process. "
                       f"GPU: {checks['vllm_gpu']}, Ollama: {checks['ollama']}. "
                       f"Monitor: tail -f /tmp/gami-process-segments.log",
            "checks": checks,
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to start extraction: {e}",
            "checks": checks,
        }


async def _memory_correct(args: dict) -> dict:
    """Correct wrong information in GAMI — fixes memories, entities, and claims in real-time.

    When you discover that stored information is wrong (wrong password, wrong IP,
    outdated fact, incorrect entity description), call this to fix it immediately.
    The old value is archived for audit trail, never deleted.
    """
    from api.services.db import AsyncSessionLocal
    from sqlalchemy import text as sql_text
    import hashlib

    item_type = args["item_type"]  # memory, entity, claim
    search_text = args.get("search_text", "")  # text to find the wrong item
    wrong_value = args.get("wrong_value", "")  # what's wrong
    correct_value = args["correct_value"]  # what it should be
    reason = args.get("reason", "Corrected during conversation")
    tenant_id = args.get("tenant_id", "claude-opus")

    # Check knowledge tier — refuse to correct reference/historical content
    async with AsyncSessionLocal() as db:
        tier_row = await db.execute(sql_text(
            "SELECT config_json->>'knowledge_tier' as tier FROM tenants WHERE tenant_id = :tid"
        ), {"tid": tenant_id})
        tier = (tier_row.scalar() or "operational")

    if tier in ("reference", "reference-technical"):
        return {
            "status": "refused",
            "message": (
                f"Cannot correct items in '{tenant_id}' — this is a {tier} knowledge tier. "
                f"Reference/historical content is preserved as-is to maintain source integrity. "
                f"It represents what the source claims, not operational truth. "
                f"To correct operational data, target the 'claude-opus' or 'shared' tenant instead."
            ),
            "knowledge_tier": tier,
        }

    corrections = []

    async with AsyncSessionLocal() as db:
        if item_type == "memory":
            # Find memories matching the search text
            rows = await db.execute(sql_text("""
                SELECT memory_id, normalized_text, subject_id
                FROM assistant_memories
                WHERE owner_tenant_id = :tid AND status = 'active'
                AND (normalized_text ILIKE :search OR subject_id ILIKE :search)
                LIMIT 5
            """), {"tid": tenant_id, "search": f"%{search_text}%"})

            for row in rows:
                if wrong_value and wrong_value.lower() not in row.normalized_text.lower():
                    continue

                # Archive old version
                archive_id = f"ARCHIVE_{row.memory_id}_{hashlib.md5(row.normalized_text.encode()).hexdigest()[:8]}"
                await db.execute(sql_text("""
                    INSERT INTO proposed_changes (proposal_id, proposer_tenant_id, change_type,
                        target_type, target_id, proposed_state_json, reason, confidence, status,
                        reviewed_by, reviewed_at)
                    VALUES (:pid, :tid, 'correction', 'assistant_memory', :mid,
                        :state, :reason, 1.0, 'approved', 'claude-realtime', NOW())
                    ON CONFLICT (proposal_id) DO NOTHING
                """), {
                    "pid": archive_id, "tid": tenant_id, "mid": row.memory_id,
                    "state": __import__("json").dumps({
                        "old_value": row.normalized_text,
                        "new_value": correct_value,
                        "wrong_value": wrong_value,
                    }),
                    "reason": reason,
                })

                # Update the memory
                await db.execute(sql_text("""
                    UPDATE assistant_memories
                    SET normalized_text = :new_text, updated_at = NOW(),
                        confirmation_count = confirmation_count + 1,
                        stability_score = LEAST(1.0, stability_score + 0.1)
                    WHERE memory_id = :mid
                """), {"new_text": correct_value, "mid": row.memory_id})

                # Re-embed
                try:
                    from api.llm.embeddings import embed_text
                    emb = await embed_text(correct_value[:2000])
                    vec = "[" + ",".join(str(v) for v in emb) + "]"
                    await db.execute(sql_text(
                        "UPDATE assistant_memories SET embedding = CAST(:v AS vector) WHERE memory_id = :mid"
                    ), {"v": vec, "mid": row.memory_id})
                except Exception:
                    pass

                corrections.append({
                    "type": "memory",
                    "id": row.memory_id,
                    "old": row.normalized_text[:100],
                    "new": correct_value[:100],
                })

        elif item_type == "entity":
            rows = await db.execute(sql_text("""
                SELECT entity_id, canonical_name, description
                FROM entities
                WHERE owner_tenant_id = :tid AND status = 'active'
                AND (canonical_name ILIKE :search OR description ILIKE :search)
                LIMIT 5
            """), {"tid": tenant_id, "search": f"%{search_text}%"})

            for row in rows:
                await db.execute(sql_text("""
                    UPDATE entities SET description = :desc, updated_at = NOW()
                    WHERE entity_id = :eid
                """), {"desc": correct_value, "eid": row.entity_id})

                # Re-embed
                try:
                    from api.llm.embeddings import embed_text
                    full_text = f"{row.canonical_name}: {correct_value}"
                    emb = await embed_text(full_text[:2000])
                    vec = "[" + ",".join(str(v) for v in emb) + "]"
                    await db.execute(sql_text(
                        "UPDATE entities SET embedding = CAST(:v AS vector) WHERE entity_id = :eid"
                    ), {"v": vec, "eid": row.entity_id})
                except Exception:
                    pass

                corrections.append({
                    "type": "entity",
                    "id": row.entity_id,
                    "name": row.canonical_name,
                    "old_desc": (row.description or "")[:100],
                    "new_desc": correct_value[:100],
                })

        elif item_type == "claim":
            rows = await db.execute(sql_text("""
                SELECT claim_id, summary_text, predicate
                FROM claims
                WHERE owner_tenant_id = :tid AND status = 'active'
                AND (summary_text ILIKE :search OR predicate ILIKE :search)
                LIMIT 5
            """), {"tid": tenant_id, "search": f"%{search_text}%"})

            for row in rows:
                if wrong_value and wrong_value.lower() not in (row.summary_text or "").lower():
                    continue

                # Mark old claim as superseded
                await db.execute(sql_text("""
                    UPDATE claims SET status = 'corrected', updated_at = NOW()
                    WHERE claim_id = :cid
                """), {"cid": row.claim_id})

                # Create corrected claim
                new_id = f"CLM_corrected_{hashlib.md5(correct_value.encode()).hexdigest()[:12]}"
                await db.execute(sql_text("""
                    INSERT INTO claims (claim_id, owner_tenant_id, predicate,
                        summary_text, confidence, modality, status, superseded_by_id)
                    VALUES (:cid, :tid, :pred, :txt, 0.95, 'corrected', 'active', NULL)
                    ON CONFLICT (claim_id) DO UPDATE SET
                        summary_text = :txt, status = 'active', confidence = 0.95
                """), {"cid": new_id, "tid": tenant_id, "pred": row.predicate, "txt": correct_value})

                # Link old → new
                await db.execute(sql_text("""
                    UPDATE claims SET superseded_by_id = :new WHERE claim_id = :old
                """), {"new": new_id, "old": row.claim_id})

                corrections.append({
                    "type": "claim",
                    "old_id": row.claim_id,
                    "new_id": new_id,
                    "old": (row.summary_text or "")[:100],
                    "new": correct_value[:100],
                })

        await db.commit()

    if not corrections:
        return {
            "status": "not_found",
            "message": f"No {item_type} found matching '{search_text}'. Try a broader search.",
        }

    return {
        "status": "corrected",
        "corrections": corrections,
        "count": len(corrections),
        "message": f"Fixed {len(corrections)} {item_type}(s). Old values archived for audit.",
    }


async def _memory_feedback(args: dict) -> dict:
    """Record feedback on retrieval quality for learning.

    This helps GAMI learn which retrievals were useful, improving future recall.
    Call this after using or evaluating recalled memories.
    """
    from api.services.learning_service import get_retrieval_logger, OUTCOME_SIGNALS
    from api.services.db import AsyncSessionLocal

    session_id = args["session_id"]
    feedback_type = args["feedback_type"]
    correction_text = args.get("correction_text")

    signal_value = OUTCOME_SIGNALS.get(feedback_type, 0.0)
    retrieval_logger = get_retrieval_logger()

    async with AsyncSessionLocal() as db:
        success = await retrieval_logger.record_outcome(
            db=db,
            session_id=session_id,
            outcome_type=feedback_type,
            correction_text=correction_text,
        )

    return {
        "status": "recorded" if success else "no_matching_log",
        "session_id": session_id,
        "feedback_type": feedback_type,
        "signal_value": signal_value,
        "message": (
            f"Feedback recorded: {feedback_type} (signal={signal_value})"
            if success else
            "No recent retrieval log found for this session"
        ),
    }


async def _memory_suggest_procedure(args: dict) -> dict:
    """Suggest relevant workflow patterns for a task.

    Searches workflow memories first (primary), falls back to legacy procedures.
    Workflow memories consolidate naturally over time via dream_consolidate.
    """
    from api.llm.embeddings import embed_text
    from api.services.db import AsyncSessionLocal
    from sqlalchemy import text as sql_text

    query = args.get("query", "")
    if not query:
        return {"error": "query required"}

    context = args.get("context", "")
    tenant_id = args.get("tenant_id", "claude-opus")
    limit = args.get("limit", 5)
    min_confidence = args.get("min_confidence", 0.4)

    # Combine query and context for embedding
    search_text = f"{query} {context}".strip()
    embedding = await embed_text(search_text)
    vec_str = "[" + ",".join(str(v) for v in embedding) + "]"

    async with AsyncSessionLocal() as db:
        # Primary: Search workflow memories
        workflow_rows = await db.execute(
            sql_text("""
                SELECT memory_id, normalized_text, importance_score,
                       1 - (embedding <=> CAST(:vec AS vector)) as similarity
                FROM assistant_memories
                WHERE owner_tenant_id = :tid
                AND memory_type = 'workflow'
                AND status = 'active'
                AND importance_score >= :min_conf
                AND embedding IS NOT NULL
                ORDER BY embedding <=> CAST(:vec AS vector)
                LIMIT :lim
            """),
            {"vec": vec_str, "tid": tenant_id, "min_conf": min_confidence, "lim": limit},
        )

        workflows = []
        for r in workflow_rows.fetchall():
            workflows.append({
                "id": r.memory_id,
                "description": r.normalized_text,
                "confidence": float(r.importance_score) if r.importance_score else 0.5,
                "similarity": float(r.similarity),
                "source": "workflow_memory",
            })

        # Fallback: Check legacy procedures table if we need more results
        legacy_procedures = []
        if len(workflows) < limit:
            try:
                proc_rows = await db.execute(
                    sql_text("""
                        SELECT procedure_id, name, description, category,
                               steps, parameters, preconditions, postconditions,
                               success_rate, execution_count, confidence,
                               1 - (embedding <=> CAST(:vec AS vector)) as similarity
                        FROM procedures
                        WHERE owner_tenant_id = :tid
                        AND status = 'active'
                        AND confidence >= :min_conf
                        AND embedding IS NOT NULL
                        ORDER BY embedding <=> CAST(:vec AS vector)
                        LIMIT :lim
                    """),
                    {"vec": vec_str, "tid": tenant_id, "min_conf": min_confidence,
                     "lim": limit - len(workflows)},
                )

                for r in proc_rows.fetchall():
                    legacy_procedures.append({
                        "procedure_id": r.procedure_id,
                        "name": r.name,
                        "description": r.description,
                        "category": r.category,
                        "steps": r.steps,
                        "parameters": r.parameters,
                        "preconditions": r.preconditions,
                        "postconditions": r.postconditions,
                        "success_rate": float(r.success_rate) if r.success_rate else 0.5,
                        "executions": r.execution_count,
                        "confidence": float(r.confidence) if r.confidence else 0.5,
                        "similarity": float(r.similarity),
                        "source": "legacy_procedure",
                    })
            except Exception:
                pass  # Legacy table may not exist or have different schema

    total = len(workflows) + len(legacy_procedures)
    return {
        "query": query,
        "workflows": workflows,
        "legacy_procedures": legacy_procedures,
        "total": total,
        "message": f"Found {len(workflows)} workflow memories and {len(legacy_procedures)} legacy procedures" if total else "No matching workflows found",
        "note": "Workflow memories consolidate naturally via dream cycle" if workflows else None,
    }


# ---------------------------------------------------------------------------
# Handler map
# ---------------------------------------------------------------------------

TOOL_HANDLERS = {
    "memory_recall": _memory_recall,
    "memory_remember": _memory_remember,
    "memory_forget": _memory_forget,
    "memory_update": _memory_update,
    "memory_cite": _memory_cite,
    "memory_verify": _memory_verify,
    "memory_search": _memory_search,
    "memory_context": _memory_context,
    "ingest_source": _ingest_source,
    "graph_explore": _graph_explore,
    "admin_stats": _admin_stats,
    "get_unprocessed_segments": _get_unprocessed_segments,
    "store_extractions": _store_extractions,
    "run_haiku_extraction": _run_haiku_extraction,
    "ingest_file": _ingest_file,
    "dream_haiku": _dream_haiku,
    "memory_correct": _memory_correct,
    "memory_feedback": _memory_feedback,
    "memory_suggest_procedure": _memory_suggest_procedure,
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def run_stdio():
    """Run the MCP server using stdio transport."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


async def run_sse(port: int = 9001):
    """Run the MCP server using SSE transport."""
    from mcp.server.sse import SseServerTransport
    from starlette.applications import Starlette
    from starlette.routing import Route, Mount
    import uvicorn

    sse = SseServerTransport("/messages/")

    async def handle_sse(request):
        async with sse.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            await server.run(
                streams[0], streams[1],
                server.create_initialization_options(),
            )

    app = Starlette(
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ]
    )

    config = uvicorn.Config(app, host="0.0.0.0", port=port)
    srv = uvicorn.Server(config)
    await srv.serve()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="GAMI MCP Server")
    parser.add_argument(
        "--transport", choices=["stdio", "sse"], default="stdio",
        help="Transport mode (default: stdio)",
    )
    parser.add_argument("--port", type=int, default=9001, help="SSE port")
    args = parser.parse_args()

    if args.transport == "sse":
        asyncio.run(run_sse(args.port))
    else:
        asyncio.run(run_stdio())


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Dream Cycle Control
# ---------------------------------------------------------------------------

async def _dream_start(args: dict) -> dict:
    """Start the dream cycle (background knowledge synthesis)."""
    import subprocess
    phase = args.get("phase")
    duration = args.get("duration", 3600)
    
    cmd = ["python3", "scripts/dream_cycle.py", "--duration", str(duration), "--check-idle"]
    if phase:
        cmd.extend(["--phase", phase])
    
    proc = subprocess.Popen(
        cmd, cwd="/opt/gami",
        env={**os.environ, "PYTHONPATH": "/opt/gami"},
        stdout=open("/tmp/gami-dream.log", "a"),
        stderr=subprocess.STDOUT,
    )
    
    # Save PID for pause/stop
    with open("/tmp/gami-dream.pid", "w") as f:
        f.write(str(proc.pid))
    
    return {"status": "started", "pid": proc.pid, "phase": phase or "all", "duration": duration}


async def _dream_stop(args: dict) -> dict:
    """Stop the running dream cycle gracefully."""
    import signal as sig
    try:
        with open("/tmp/gami-dream.pid") as f:
            pid = int(f.read().strip())
        os.kill(pid, sig.SIGTERM)
        return {"status": "stopping", "pid": pid, "message": "Sent SIGTERM — dream cycle will finish current task and exit"}
    except FileNotFoundError:
        return {"status": "not_running", "message": "No dream cycle PID file found"}
    except ProcessLookupError:
        return {"status": "not_running", "message": "Dream cycle process not found (already stopped)"}


async def _dream_status(args: dict) -> dict:
    """Check if dream cycle is running and its progress."""
    try:
        with open("/tmp/gami-dream.pid") as f:
            pid = int(f.read().strip())
        # Check if process is alive
        os.kill(pid, 0)
        
        # Read last few lines of log
        log_lines = []
        try:
            with open("/tmp/gami-dream.log") as f:
                log_lines = f.readlines()[-5:]
        except:
            pass
        
        return {"status": "running", "pid": pid, "recent_log": [l.strip() for l in log_lines]}
    except (FileNotFoundError, ProcessLookupError):
        return {"status": "not_running"}


# Register dream tools
TOOL_HANDLERS["dream_start"] = _dream_start
TOOL_HANDLERS["dream_stop"] = _dream_stop
TOOL_HANDLERS["dream_status"] = _dream_status


async def _review_proposals(args: dict) -> dict:
    """Review and approve/reject pending proposals from the dream cycle."""
    from api.services.db import AsyncSessionLocal
    
    action = args.get("action", "list")  # list, approve, reject
    proposal_id = args.get("proposal_id")
    
    async with AsyncSessionLocal() as db:
        if action == "list":
            rows = await db.execute(text("""
                SELECT proposal_id, change_type, target_type, reason, confidence,
                       proposed_state_json, created_at
                FROM proposed_changes 
                WHERE status = 'pending'
                ORDER BY 
                    CASE WHEN proposed_state_json::text ILIKE '%credential%' THEN 0 ELSE 1 END,
                    confidence DESC
                LIMIT 20
            """))
            proposals = []
            for r in rows:
                state = json.loads(r.proposed_state_json) if r.proposed_state_json else {}
                proposals.append({
                    "id": r.proposal_id,
                    "type": r.change_type,
                    "target": r.target_type,
                    "reason": r.reason,
                    "confidence": round(float(r.confidence), 2) if r.confidence else 0,
                    "priority": state.get("priority", "normal"),
                    "old": state.get("old_text", state.get("a_name", ""))[:100],
                    "new": state.get("new_text", state.get("b_name", ""))[:100],
                })
            
            count = await db.execute(text("SELECT count(*) FROM proposed_changes WHERE status = 'pending'"))
            total = count.scalar()
            
            return {"pending_count": total, "proposals": proposals}
        
        elif action == "approve" and proposal_id:
            # Get the proposal
            row = await db.execute(text(
                "SELECT change_type, target_type, target_id, proposed_state_json FROM proposed_changes WHERE proposal_id = :pid AND status = 'pending'"
            ), {"pid": proposal_id})
            prop = row.fetchone()
            if not prop:
                return {"error": "Proposal not found or already reviewed"}
            
            state = json.loads(prop.proposed_state_json) if prop.proposed_state_json else {}
            
            # Execute the change
            if prop.change_type == "merge_entities":
                merge_into = state.get("merge_into")
                if merge_into:
                    await db.execute(text("UPDATE entities SET status = 'merged', merged_into_id = :t WHERE entity_id = :e"),
                                   {"e": prop.target_id, "t": merge_into})
            elif prop.change_type == "supersede_claim":
                new_id = state.get("superseded_by")
                if new_id:
                    await db.execute(text("UPDATE claims SET status = 'superseded', superseded_by_id = :n WHERE claim_id = :o"),
                                   {"o": prop.target_id, "n": new_id})
            elif prop.change_type == "update_memory":
                # Flag memory as needing update (don't auto-change credentials)
                await db.execute(text("UPDATE assistant_memories SET status = 'stale' WHERE memory_id = :m"),
                               {"m": prop.target_id})
            
            await db.execute(text("""
                UPDATE proposed_changes SET status = 'approved', reviewed_by = 'claude-opus', 
                    reviewed_at = NOW(), review_notes = 'Approved by Claude'
                WHERE proposal_id = :pid
            """), {"pid": proposal_id})
            await db.commit()
            return {"status": "approved", "proposal_id": proposal_id}
        
        elif action == "reject" and proposal_id:
            notes = args.get("reason", "Rejected by Claude")
            await db.execute(text("""
                UPDATE proposed_changes SET status = 'rejected', reviewed_by = 'claude-opus',
                    reviewed_at = NOW(), review_notes = :notes
                WHERE proposal_id = :pid
            """), {"pid": proposal_id, "notes": notes})
            await db.commit()
            return {"status": "rejected", "proposal_id": proposal_id}
    
    return {"error": "Invalid action. Use: list, approve, reject"}


TOOL_HANDLERS["review_proposals"] = _review_proposals


# ---------------------------------------------------------------------------
# Tenant & Ingest Pipeline Tools
# ---------------------------------------------------------------------------

async def _create_tenant(args: dict) -> dict:
    """Create a new tenant for organizing content."""
    from api.services.db import AsyncSessionLocal
    from sqlalchemy import text as sql_text

    tenant_id = args["tenant_id"]
    display_name = args.get("display_name", tenant_id)
    description = args.get("description", "")

    async with AsyncSessionLocal() as db:
        # Check if exists
        row = await db.execute(
            sql_text("SELECT tenant_id FROM tenants WHERE tenant_id = :tid"),
            {"tid": tenant_id},
        )
        if row.fetchone():
            return {"status": "exists", "tenant_id": tenant_id, "message": "Tenant already exists"}

        await db.execute(
            sql_text(
                "INSERT INTO tenants (tenant_id, name, description, tenant_type, status) "
                "VALUES (:tid, :name, :desc, 'content', 'active')"
            ),
            {"tid": tenant_id, "name": display_name, "desc": description},
        )
        await db.commit()

    return {"status": "created", "tenant_id": tenant_id, "display_name": display_name}


async def _bulk_ingest(args: dict) -> dict:
    """Launch bulk ingest of a directory into a tenant."""
    import subprocess
    import glob

    path = args["path"]
    tenant_id = args["tenant_id"]
    file_type = args.get("file_type", "pdf")
    recursive = args.get("recursive", True)

    if not os.path.exists(path):
        return {"status": "error", "message": f"Path does not exist: {path}"}

    # Count files
    ext_map = {"pdf": "*.pdf", "markdown": "*.md", "plaintext": "*.txt"}
    pattern = ext_map.get(file_type, f"*.{file_type}")
    if os.path.isdir(path):
        if recursive:
            files = glob.glob(os.path.join(path, "**", pattern), recursive=True)
        else:
            files = glob.glob(os.path.join(path, pattern))
    else:
        files = [path]

    file_count = len(files)
    if file_count == 0:
        return {"status": "error", "message": f"No {file_type} files found in {path}"}

    # Estimate time (~30s per PDF for parse+store)
    est_minutes = file_count * 0.5 / 60  # rough estimate

    # Launch bulk_ingest.py in background
    cmd = [
        sys.executable, "/opt/gami/scripts/bulk_ingest.py",
        "--path", path,
        "--tenant", tenant_id,
        "--type", file_type,
        "--workers", "2",
    ]
    if not recursive:
        cmd.append("--no-recursive")

    log_path = f"/tmp/gami-bulk-ingest-{tenant_id}.log"
    proc = subprocess.Popen(
        cmd,
        cwd="/opt/gami",
        env={**os.environ, "PYTHONPATH": "/opt/gami"},
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    return {
        "status": "started",
        "pid": proc.pid,
        "file_count": file_count,
        "tenant_id": tenant_id,
        "file_type": file_type,
        "estimated_minutes": round(est_minutes, 1),
        "log_path": log_path,
        "message": (
            f"Bulk ingest started (PID {proc.pid}). "
            f"{file_count} {file_type} files into tenant '{tenant_id}'. "
            f"Monitor: tail -f {log_path}"
        ),
    }


async def _tenant_stats(args: dict) -> dict:
    """Get stats for a specific tenant."""
    from api.services.db import AsyncSessionLocal
    from sqlalchemy import text as sql_text

    tenant_id = args["tenant_id"]

    async with AsyncSessionLocal() as db:
        # Check tenant exists
        row = await db.execute(
            sql_text("SELECT tenant_id, name, status FROM tenants WHERE tenant_id = :tid"),
            {"tid": tenant_id},
        )
        tenant = row.fetchone()
        if not tenant:
            return {"error": f"Tenant '{tenant_id}' not found"}

        # Source count
        src = await db.execute(
            sql_text("SELECT COUNT(*) FROM sources WHERE owner_tenant_id = :tid"),
            {"tid": tenant_id},
        )
        source_count = src.scalar()

        # Segment counts
        seg = await db.execute(
            sql_text("SELECT COUNT(*) FROM segments WHERE owner_tenant_id = :tid"),
            {"tid": tenant_id},
        )
        segment_count = seg.scalar()

        # Embedded count
        emb = await db.execute(
            sql_text(
                "SELECT COUNT(*) FROM segments "
                "WHERE owner_tenant_id = :tid AND embedding IS NOT NULL"
            ),
            {"tid": tenant_id},
        )
        embedded_count = emb.scalar()

        # Entity count
        ent = await db.execute(
            sql_text("SELECT COUNT(*) FROM entities WHERE owner_tenant_id = :tid"),
            {"tid": tenant_id},
        )
        entity_count = ent.scalar()

        # Total text size
        txt_size = await db.execute(
            sql_text(
                "SELECT COALESCE(SUM(LENGTH(text)), 0) FROM segments "
                "WHERE owner_tenant_id = :tid"
            ),
            {"tid": tenant_id},
        )
        total_text_chars = txt_size.scalar()

    return {
        "tenant_id": tenant_id,
        "display_name": tenant[1],
        "status": tenant[2],
        "sources": source_count,
        "segments": segment_count,
        "embedded": embedded_count,
        "unembedded": segment_count - embedded_count,
        "entities": entity_count,
        "total_text_chars": total_text_chars,
        "total_text_mb": round(total_text_chars / (1024 * 1024), 2),
    }


async def _tenant_search(args: dict) -> dict:
    """Search within a specific tenant using hybrid vector+keyword search."""
    from api.llm.embeddings import embed_text
    from api.services.db import AsyncSessionLocal
    from sqlalchemy import text as sql_text

    query = args["query"]
    tenant_id = args["tenant_id"]
    max_results = min(args.get("max_results", 10), 50)

    # Get query embedding — use the right method for this tenant
    # GPU tenants (books, manuals, whitepapers) need GPU GGUF embeddings
    # Ollama tenants use the standard embed_text
    embed_method = "ollama"
    try:
        tier_row = await AsyncSessionLocal().execute(
            sql_text("SELECT config_json->>'embedding_method' FROM tenants WHERE tenant_id = :tid"),
            {"tid": tenant_id},
        )
        embed_method = tier_row.scalar() or "ollama"
    except Exception:
        pass

    try:
        if embed_method == "gpu_gguf":
            # Use GGUF model for GPU tenants — must match their stored embeddings
            import subprocess
            result = subprocess.run(
                ["python3", "-c",
                 f"import sys; sys.path.insert(0,'/opt/gami'); "
                 f"from scripts.gpu_embed_gguf import embed_query; "
                 f"emb = embed_query({query!r}); "
                 f"print(','.join(str(v) for v in emb))"],
                capture_output=True, text=True, timeout=30,
                env={**__import__("os").environ, "LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"},
            )
            if result.returncode == 0:
                query_embedding = [float(v) for v in result.stdout.strip().split(",")]
                vec_str = "[" + result.stdout.strip() + "]"
            else:
                raise RuntimeError(f"GPU embed failed: {result.stderr[:200]}")
        else:
            query_embedding = await embed_text(query, is_query=True)
            vec_str = "[" + ",".join(str(v) for v in query_embedding) + "]"
    except Exception as e:
        logger.warning("Embedding failed, falling back to lexical-only: %s", e)
        query_embedding = None
        vec_str = None

    async with AsyncSessionLocal() as db:
        results = []

        if vec_str:
            # Vector search (primary), augmented with lexical boost
            rows = await db.execute(
                sql_text("""
                    SELECT segment_id, text, source_id, segment_type,
                           page_start, page_end, title_or_heading,
                           (1 - (embedding <=> CAST(:vec AS vector))) AS score
                    FROM segments
                    WHERE owner_tenant_id = :tid
                      AND embedding IS NOT NULL
                    ORDER BY embedding <=> CAST(:vec AS vector)
                    LIMIT :lim
                """),
                {"vec": vec_str, "tid": tenant_id, "lim": max_results},
            )
        else:
            # Lexical only
            rows = await db.execute(
                sql_text("""
                    SELECT segment_id, text, source_id, segment_type,
                           page_start, page_end, title_or_heading,
                           ts_rank(lexical_tsv, plainto_tsquery('english', :q)) AS score
                    FROM segments
                    WHERE owner_tenant_id = :tid
                      AND lexical_tsv @@ plainto_tsquery('english', :q)
                    ORDER BY score DESC
                    LIMIT :lim
                """),
                {"tid": tenant_id, "q": query, "lim": max_results},
            )

        for r in rows.fetchall():
            # Get source title for citation
            src_row = await db.execute(
                sql_text("SELECT title, author_or_origin FROM sources WHERE source_id = :sid"),
                {"sid": r[2]},
            )
            src = src_row.fetchone()

            results.append({
                "segment_id": r[0],
                "text": r[1][:800],
                "source_id": r[2],
                "segment_type": r[3],
                "page_start": r[4],
                "page_end": r[5],
                "heading": r[6],
                "score": round(float(r[7]), 4) if r[7] else 0,
                "source_title": src[0] if src else None,
                "source_author": src[1] if src else None,
                "citation": f"{src[0] or 'Unknown'}" + (f", p.{r[4]}" if r[4] else ""),
            })

    return {
        "query": query,
        "tenant_id": tenant_id,
        "results": results,
        "total": len(results),
    }


TOOL_HANDLERS["create_tenant"] = _create_tenant
TOOL_HANDLERS["bulk_ingest"] = _bulk_ingest
TOOL_HANDLERS["tenant_stats"] = _tenant_stats
TOOL_HANDLERS["tenant_search"] = _tenant_search
