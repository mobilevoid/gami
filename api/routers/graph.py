"""Graph exploration API for GAMI.

Provides endpoints for exploring the knowledge graph built on Apache AGE.
"""
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.services.graph_service import (
    get_neighborhood,
    find_path,
    get_cluster,
    get_entity_by_name,
)
from api.services.graph_sync import sync_all_from_db
from api.services.db import AsyncSessionLocal

logger = logging.getLogger("gami.routers.graph")

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ExploreRequest(BaseModel):
    entity_id: Optional[str] = Field(default=None, description="Entity ID to explore")
    entity_name: Optional[str] = Field(default=None, description="Entity name to look up")
    tenant_id: str = Field(default="default")
    depth: int = Field(default=2, ge=1, le=3)
    limit: int = Field(default=20, ge=1, le=100)


class PathRequest(BaseModel):
    from_entity_id: str
    to_entity_id: str
    max_depth: int = Field(default=5, ge=1, le=6)


class ExploreResponse(BaseModel):
    entity_id: Optional[str] = None
    center: Optional[dict] = None
    neighbors: list = []
    edges: list = []
    node_count: int = 0
    edge_count: int = 0


class PathResponse(BaseModel):
    found: bool
    from_entity_id: str
    to_entity_id: str
    nodes: list = []
    edges: list = []
    path_length: int = -1


class ClusterResponse(BaseModel):
    found: bool
    cluster_id: str
    cluster_type: Optional[str] = None
    member_count: int = 0
    members: list = []
    summary: Optional[str] = None
    confidence: Optional[float] = None


class SyncResponse(BaseModel):
    entities_synced: int = 0
    relations_synced: int = 0
    claims_synced: int = 0
    errors: int = 0


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/explore", response_model=ExploreResponse)
async def explore_graph(req: ExploreRequest):
    """Explore the neighborhood of an entity in the knowledge graph.

    Provide either entity_id directly, or entity_name to look up first.
    """
    entity_id = req.entity_id

    if not entity_id and req.entity_name:
        async with AsyncSessionLocal() as db:
            entity = await get_entity_by_name(
                req.entity_name, req.tenant_id, db=db
            )
            if entity:
                entity_id = entity["entity_id"]
            else:
                return ExploreResponse(entity_id=None)

    if not entity_id:
        raise HTTPException(status_code=400, detail="Provide entity_id or entity_name")

    try:
        result = await get_neighborhood(
            entity_id=entity_id,
            depth=req.depth,
            limit=req.limit,
        )
        return ExploreResponse(
            entity_id=entity_id,
            center=result.get("center"),
            neighbors=result.get("neighbors", []),
            edges=result.get("edges", []),
            node_count=result.get("node_count", 0),
            edge_count=result.get("edge_count", 0),
        )
    except Exception as exc:
        logger.error("Graph explore failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Graph explore failed: {exc}")


@router.post("/path", response_model=PathResponse)
async def find_graph_path(req: PathRequest):
    """Find the shortest path between two entities."""
    try:
        result = await find_path(
            from_entity_id=req.from_entity_id,
            to_entity_id=req.to_entity_id,
            max_depth=req.max_depth,
        )
        return PathResponse(**result)
    except Exception as exc:
        logger.error("Graph path failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Graph path failed: {exc}")


@router.get("/cluster/{cluster_id}", response_model=ClusterResponse)
async def get_graph_cluster(cluster_id: str):
    """Get a topic cluster with its members."""
    try:
        result = await get_cluster(cluster_id)
        if not result.get("found"):
            raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} not found")
        return ClusterResponse(
            found=True,
            cluster_id=result["cluster_id"],
            cluster_type=result.get("cluster_type"),
            member_count=result.get("member_count", 0),
            members=result.get("members", []),
            summary=result.get("summary"),
            confidence=result.get("confidence"),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Cluster fetch failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Cluster fetch failed: {exc}")


@router.post("/sync", response_model=SyncResponse)
async def sync_graph(tenant_id: Optional[str] = None, batch_size: int = 100):
    """Sync entities, relations, and claims from relational DB to the AGE graph."""
    try:
        stats = await sync_all_from_db(tenant_id=tenant_id, batch_size=batch_size)
        return SyncResponse(**stats)
    except Exception as exc:
        logger.error("Graph sync failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Graph sync failed: {exc}")
