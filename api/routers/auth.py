"""Auth / tenant management router for GAMI.

Provides CRUD endpoints for tenants and permission grants.
"""
import json
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from api.services.db import get_db

logger = logging.getLogger("gami.routers.auth")

router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class TenantCreate(BaseModel):
    tenant_id: str
    name: str
    description: Optional[str] = None
    tenant_type: str = "agent"
    daily_write_budget: int = 10000
    daily_delete_budget: int = 100
    config_json: dict = {}


class PermissionGrant(BaseModel):
    target_tenant_id: str
    permission: str = "read"
    scope: str = "all"
    granted_by: str = "admin"
    expires_at: Optional[str] = None


# ---------------------------------------------------------------------------
# Tenant endpoints
# ---------------------------------------------------------------------------

@router.get("/tenants")
async def list_tenants(
    db: AsyncSession = Depends(get_db),
):
    """List all tenants."""
    result = await db.execute(
        text(
            "SELECT tenant_id, name, description, tenant_type, status, "
            "daily_write_budget, daily_delete_budget, created_at "
            "FROM tenants ORDER BY created_at"
        )
    )
    tenants = [dict(r._mapping) for r in result.fetchall()]
    return {"count": len(tenants), "tenants": tenants}


@router.get("/tenants/{tenant_id}")
async def get_tenant(
    tenant_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get details of a single tenant, including permissions and stats."""
    result = await db.execute(
        text(
            "SELECT tenant_id, name, description, tenant_type, status, "
            "daily_write_budget, daily_delete_budget, config_json, created_at "
            "FROM tenants WHERE tenant_id = :tid"
        ),
        {"tid": tenant_id},
    )
    row = result.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"Tenant '{tenant_id}' not found.")

    tenant = dict(row._mapping)

    # Fetch permissions
    perm_result = await db.execute(
        text(
            "SELECT target_tenant_id, permission, scope, granted_at, "
            "granted_by, expires_at "
            "FROM tenant_permissions WHERE tenant_id = :tid"
        ),
        {"tid": tenant_id},
    )
    tenant["permissions"] = [dict(r._mapping) for r in perm_result.fetchall()]

    # Fetch stats
    src_result = await db.execute(
        text("SELECT COUNT(*) as cnt FROM sources WHERE owner_tenant_id = :tid"),
        {"tid": tenant_id},
    )
    seg_result = await db.execute(
        text("SELECT COUNT(*) as cnt FROM segments WHERE owner_tenant_id = :tid"),
        {"tid": tenant_id},
    )
    tenant["stats"] = {
        "sources": src_result.scalar(),
        "segments": seg_result.scalar(),
    }

    return tenant


@router.post("/tenants")
async def create_tenant(
    body: TenantCreate,
    db: AsyncSession = Depends(get_db),
):
    """Create a new tenant."""
    # Check for existing
    result = await db.execute(
        text("SELECT tenant_id FROM tenants WHERE tenant_id = :tid"),
        {"tid": body.tenant_id},
    )
    if result.fetchone():
        raise HTTPException(
            status_code=409,
            detail=f"Tenant '{body.tenant_id}' already exists.",
        )

    now = datetime.now(timezone.utc)
    await db.execute(
        text("""
            INSERT INTO tenants (
                tenant_id, name, description, tenant_type, status,
                daily_write_budget, daily_delete_budget, config_json, created_at
            ) VALUES (
                :tid, :name, :desc, :ttype, 'active',
                :wb, :db, CAST(:cfg AS jsonb), :now
            )
        """),
        {
            "tid": body.tenant_id,
            "name": body.name,
            "desc": body.description,
            "ttype": body.tenant_type,
            "wb": body.daily_write_budget,
            "db": body.daily_delete_budget,
            "cfg": json.dumps(body.config_json),
            "now": now,
        },
    )
    await db.commit()

    logger.info("Created tenant: %s (%s)", body.tenant_id, body.tenant_type)
    return {
        "status": "created",
        "tenant_id": body.tenant_id,
        "name": body.name,
        "tenant_type": body.tenant_type,
    }


@router.post("/tenants/{tenant_id}/permissions")
async def grant_permission(
    tenant_id: str,
    body: PermissionGrant,
    db: AsyncSession = Depends(get_db),
):
    """Grant a permission from one tenant to another."""
    # Validate both tenants exist
    for tid in [tenant_id, body.target_tenant_id]:
        result = await db.execute(
            text("SELECT tenant_id FROM tenants WHERE tenant_id = :tid"),
            {"tid": tid},
        )
        if not result.fetchone():
            raise HTTPException(
                status_code=404, detail=f"Tenant '{tid}' not found."
            )

    # Parse expiry
    expires = None
    if body.expires_at:
        try:
            expires = datetime.fromisoformat(body.expires_at)
        except ValueError:
            raise HTTPException(
                status_code=400, detail="Invalid expires_at format (use ISO 8601)."
            )

    now = datetime.now(timezone.utc)
    try:
        await db.execute(
            text("""
                INSERT INTO tenant_permissions (
                    tenant_id, target_tenant_id, permission, scope,
                    granted_at, granted_by, expires_at
                ) VALUES (
                    :tid, :target, :perm, :scope, :now, :by, :expires
                )
            """),
            {
                "tid": tenant_id,
                "target": body.target_tenant_id,
                "perm": body.permission,
                "scope": body.scope,
                "now": now,
                "by": body.granted_by,
                "expires": expires,
            },
        )
        await db.commit()
    except Exception as exc:
        if "duplicate key" in str(exc).lower() or "unique" in str(exc).lower():
            raise HTTPException(
                status_code=409,
                detail="Permission already granted.",
            )
        raise

    logger.info(
        "Granted %s permission: %s -> %s (scope: %s)",
        body.permission, tenant_id, body.target_tenant_id, body.scope,
    )
    return {
        "status": "granted",
        "tenant_id": tenant_id,
        "target_tenant_id": body.target_tenant_id,
        "permission": body.permission,
        "scope": body.scope,
    }
