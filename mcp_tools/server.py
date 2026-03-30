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
    try:
        handler = TOOL_HANDLERS.get(name)
        if not handler:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

        result = await handler(arguments)
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

    except Exception as exc:
        logger.error("Tool %s failed: %s", name, exc, exc_info=True)
        return [TextContent(
            type="text",
            text=json.dumps({"error": str(exc), "tool": name}),
        )]


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

    result = await recall(
        query=query,
        tenant_id=tenant_id,
        tenant_ids=tenant_ids,
        max_tokens=max_tokens,
        mode=None,  # auto-classify
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

    return {
        "context": result.context_text,
        "citations": citations,
        "total_tokens": result.total_tokens_used,
        "results_used": len(result.evidence),
        "mode": result.mode,
        "search_ms": result.search_ms,
    }


async def _memory_remember(args: dict) -> dict:
    """Store a new assistant memory."""
    from api.llm.embeddings import embed_text
    from api.services.db import AsyncSessionLocal

    text_content = args["text"]
    memory_type = args.get("memory_type", "fact")
    subject_id = args.get("subject_id", "general")
    tenant_id = args.get("tenant_id", "claude-opus")
    importance = args.get("importance", 0.5)

    # Embed the memory
    embedding = await embed_text(text_content)
    vec_str = "[" + ",".join(str(v) for v in embedding) + "]"

    memory_id = f"MEM_{uuid.uuid4().hex[:16]}"
    now = datetime.now(timezone.utc)

    async with AsyncSessionLocal() as db:
        from sqlalchemy import text as sql_text

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
