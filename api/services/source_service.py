"""Source registration and management service for GAMI."""
import hashlib
import os
import shutil
import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from api.config import settings


def _short_uuid() -> str:
    """Return the first 8 hex chars of a uuid4."""
    return uuid.uuid4().hex[:8]


def _generate_source_id(source_type: str) -> str:
    """Generate a deterministic-format source ID: SRC_{TYPE}_{uuid_short}."""
    return f"SRC_{source_type.upper()}_{_short_uuid()}"


def _compute_checksum(file_path: str) -> str:
    """Compute SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _detect_mime_type(file_path: str, source_type: str) -> str:
    """Best-effort MIME type from extension and source_type."""
    ext = os.path.splitext(file_path)[1].lower()
    mime_map = {
        ".md": "text/markdown",
        ".txt": "text/plain",
        ".html": "text/html",
        ".htm": "text/html",
        ".json": "application/json",
        ".jsonl": "application/x-jsonl",
        ".pdf": "application/pdf",
        ".sqlite": "application/x-sqlite3",
        ".db": "application/x-sqlite3",
    }
    return mime_map.get(ext, "application/octet-stream")


async def register_source(
    db: AsyncSession,
    tenant_id: str,
    file_path: str,
    source_type: str,
    title: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> dict:
    """
    Register a new source file in the database.

    - Computes file checksum
    - Checks for duplicate (same checksum + tenant = skip)
    - Copies raw file to object store
    - Inserts into sources table
    - Returns dict with source_id, is_duplicate flag
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Source file not found: {file_path}")

    checksum = _compute_checksum(file_path)
    file_size = os.path.getsize(file_path)

    # Check for duplicate
    result = await db.execute(
        text(
            "SELECT source_id FROM sources "
            "WHERE checksum = :checksum AND owner_tenant_id = :tenant_id"
        ),
        {"checksum": checksum, "tenant_id": tenant_id},
    )
    existing = result.fetchone()
    if existing:
        return {
            "source_id": existing[0],
            "is_duplicate": True,
            "message": f"Duplicate detected — existing source {existing[0]}",
        }

    source_id = _generate_source_id(source_type)
    mime_type = _detect_mime_type(file_path, source_type)

    # Copy raw file to object store
    raw_dir = os.path.join(settings.OBJECT_STORE, "raw", source_id)
    os.makedirs(raw_dir, exist_ok=True)
    dest_path = os.path.join(raw_dir, os.path.basename(file_path))
    shutil.copy2(file_path, dest_path)

    if title is None:
        title = os.path.basename(file_path)

    now = datetime.now(timezone.utc)

    await db.execute(
        text("""
            INSERT INTO sources (
                source_id, owner_tenant_id, source_type, title,
                source_uri, raw_file_path, checksum, file_size_bytes,
                mime_type, parse_status, metadata_json, ingested_at
            ) VALUES (
                :source_id, :tenant_id, :source_type, :title,
                :source_uri, :raw_file_path, :checksum, :file_size,
                :mime_type, 'pending', CAST(:metadata AS jsonb), :now
            )
        """),
        {
            "source_id": source_id,
            "tenant_id": tenant_id,
            "source_type": source_type,
            "title": title,
            "source_uri": f"file://{os.path.abspath(file_path)}",
            "raw_file_path": dest_path,
            "checksum": checksum,
            "file_size": file_size,
            "mime_type": mime_type,
            "metadata": __import__("json").dumps(metadata or {}),
            "now": now,
        },
    )
    await db.commit()

    return {
        "source_id": source_id,
        "is_duplicate": False,
        "file_path": dest_path,
        "checksum": checksum,
        "file_size_bytes": file_size,
    }


async def get_source(db: AsyncSession, source_id: str) -> Optional[dict]:
    """Retrieve a single source record by ID."""
    result = await db.execute(
        text(
            "SELECT source_id, owner_tenant_id, source_type, title, "
            "source_uri, raw_file_path, checksum, file_size_bytes, "
            "mime_type, parse_status, parser_version, metadata_json, "
            "ingested_at, storage_tier "
            "FROM sources WHERE source_id = :sid"
        ),
        {"sid": source_id},
    )
    row = result.fetchone()
    if not row:
        return None
    return dict(row._mapping)


async def list_sources(
    db: AsyncSession,
    tenant_id: str,
    source_type: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict]:
    """List sources for a tenant, optionally filtered by type."""
    query = (
        "SELECT source_id, source_type, title, parse_status, "
        "file_size_bytes, ingested_at "
        "FROM sources WHERE owner_tenant_id = :tid"
    )
    params: dict = {"tid": tenant_id, "lim": limit, "off": offset}

    if source_type:
        query += " AND source_type = :stype"
        params["stype"] = source_type

    query += " ORDER BY ingested_at DESC LIMIT :lim OFFSET :off"

    result = await db.execute(text(query), params)
    return [dict(r._mapping) for r in result.fetchall()]


async def update_parse_status(
    db: AsyncSession,
    source_id: str,
    status: str,
    parser_version: Optional[str] = None,
) -> None:
    """Update the parse_status (and optionally parser_version) of a source."""
    if parser_version:
        await db.execute(
            text(
                "UPDATE sources SET parse_status = :status, "
                "parser_version = :pv WHERE source_id = :sid"
            ),
            {"status": status, "pv": parser_version, "sid": source_id},
        )
    else:
        await db.execute(
            text(
                "UPDATE sources SET parse_status = :status "
                "WHERE source_id = :sid"
            ),
            {"status": status, "sid": source_id},
        )
    await db.commit()
