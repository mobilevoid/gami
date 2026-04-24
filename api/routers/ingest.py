"""Ingestion API router for GAMI.

Provides endpoints to register sources, trigger parsing, and monitor
ingestion job status.
"""
import json
import logging
import os
import tempfile
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from api.config import settings
from api.services.db import get_db
from api.services.source_service import (
    get_source,
    list_sources,
    register_source,
    update_parse_status,
)
from api.services.segment_service import get_segment, get_segments, store_segments

logger = logging.getLogger("gami.routers.ingest")


async def post_ingest_embed_and_manifold(
    db: AsyncSession,
    segment_ids: list[str],
    batch_size: int = 50,
) -> dict:
    """Post-ingest hook: embed segments and compute manifold coordinates.

    This enables immediate searchability via product manifold search
    without waiting for the dream cycle.

    Args:
        db: Database session
        segment_ids: List of newly created segment IDs
        batch_size: Number of segments to process at once

    Returns:
        Dict with counts of embedded and manifold-computed segments
    """
    if not segment_ids:
        return {"embedded": 0, "manifold_computed": 0}

    try:
        import numpy as np
        import torch
        from api.llm.embeddings import embed_text
        from api.llm.manifold_embeddings import get_manifold_encoder

        embedded = 0
        manifold_computed = 0
        encoder = get_manifold_encoder()

        # Process in batches
        for i in range(0, len(segment_ids), batch_size):
            batch_ids = segment_ids[i:i + batch_size]

            # Fetch segment texts
            result = await db.execute(
                text("""
                    SELECT segment_id, text
                    FROM segments
                    WHERE segment_id = ANY(:ids)
                    AND embedding IS NULL
                """),
                {"ids": batch_ids},
            )
            rows = result.fetchall()

            for row in rows:
                # Handle both tuple and named access
                if hasattr(row, 'segment_id'):
                    seg_id, seg_text = row.segment_id, row.text
                else:
                    seg_id, seg_text = row[0], row[1]
                if not seg_text or len(seg_text.strip()) < 10:
                    continue

                try:
                    # Generate 768d embedding
                    embedding = await embed_text(seg_text[:8000])  # Truncate long texts
                    vec_str = "[" + ",".join(str(v) for v in embedding) + "]"

                    # Update segment with embedding
                    await db.execute(
                        text("UPDATE segments SET embedding = CAST(:v AS vector) WHERE segment_id = :sid"),
                        {"v": vec_str, "sid": seg_id},
                    )
                    embedded += 1

                    # Compute manifold coordinates
                    emb_array = np.array(embedding, dtype=np.float32).reshape(1, -1)
                    emb_tensor = torch.tensor(emb_array, dtype=torch.float32)

                    with torch.no_grad():
                        coords = encoder.encode_batch_from_embeddings(emb_tensor)[0]

                    h_list = coords.hyperbolic.tolist()
                    s_list = coords.spherical.tolist()
                    e_vec = "[" + ",".join(str(v) for v in coords.euclidean) + "]"

                    await db.execute(
                        text("""
                            INSERT INTO product_manifold_coords
                                (target_id, target_type, hyperbolic_coords, spherical_coords, euclidean_coords)
                            VALUES (:tid, 'segment', :h, :s, CAST(:e AS vector))
                            ON CONFLICT (target_id, target_type) DO UPDATE SET
                                hyperbolic_coords = EXCLUDED.hyperbolic_coords,
                                spherical_coords = EXCLUDED.spherical_coords,
                                euclidean_coords = EXCLUDED.euclidean_coords,
                                computed_at = NOW()
                        """),
                        {"tid": seg_id, "h": h_list, "s": s_list, "e": e_vec},
                    )
                    manifold_computed += 1

                except Exception as e:
                    logger.warning(f"Failed to embed/manifold segment {seg_id}: {e}")
                    continue

            await db.commit()

        logger.info(f"Post-ingest: embedded {embedded}, manifold {manifold_computed} segments")
        return {"embedded": embedded, "manifold_computed": manifold_computed}

    except Exception as e:
        logger.error(f"Post-ingest hook failed: {e}")
        return {"embedded": 0, "manifold_computed": 0, "error": str(e)}

router = APIRouter()


@router.post("/source")
async def ingest_source(
    file_path: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    source_type: str = Form("markdown"),
    title: Optional[str] = Form(None),
    tenant_id: str = Form("shared"),
    metadata_json: str = Form("{}"),
    db: AsyncSession = Depends(get_db),
):
    """
    Register and parse a source file.

    Provide either `file_path` (server-side path) or `file` (uploaded file).
    The source is registered, parsed synchronously, and segments stored.
    For large files, use the async Celery path via /source/async.
    """
    # Validate inputs
    if not file_path and not file:
        raise HTTPException(
            status_code=400,
            detail="Provide either 'file_path' or 'file' upload.",
        )

    try:
        metadata = json.loads(metadata_json)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid metadata_json.")

    # Validate tenant exists
    result = await db.execute(
        text("SELECT tenant_id FROM tenants WHERE tenant_id = :tid"),
        {"tid": tenant_id},
    )
    if not result.fetchone():
        raise HTTPException(status_code=404, detail=f"Tenant '{tenant_id}' not found.")

    # Handle file upload — save to temp location
    actual_path = file_path
    temp_path = None
    if file and not file_path:
        temp_dir = os.path.join(settings.OBJECT_STORE, "uploads")
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, f"{uuid.uuid4().hex}_{file.filename}")
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        actual_path = temp_path

    if not os.path.isfile(actual_path):
        raise HTTPException(status_code=404, detail=f"File not found: {actual_path}")

    # Register source
    try:
        reg_result = await register_source(
            db=db,
            tenant_id=tenant_id,
            file_path=actual_path,
            source_type=source_type,
            title=title,
            metadata=metadata,
        )
    except Exception as exc:
        logger.error("Source registration failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    if reg_result.get("is_duplicate"):
        return {
            "status": "duplicate",
            "source_id": reg_result["source_id"],
            "message": reg_result["message"],
        }

    source_id = reg_result["source_id"]

    # Parse synchronously
    try:
        from parsers import get_parser, get_parser_for_file

        try:
            parser = get_parser(source_type)
        except ValueError:
            parser = get_parser_for_file(actual_path)

        parse_result = parser.parse(actual_path, metadata)

        # Store segments
        segment_ids = await store_segments(
            db=db,
            source_id=source_id,
            tenant_id=tenant_id,
            parsed_segments=parse_result.segments,
        )

        await update_parse_status(db, source_id, "parsed", parser_version="1.0.0")

        # Post-ingest hook: embed segments and compute manifold coords for immediate search
        embed_result = await post_ingest_embed_and_manifold(db, segment_ids)

        # Create job record
        job_id = f"JOB_INGEST_{uuid.uuid4().hex[:8]}"
        now = datetime.now(timezone.utc)
        await db.execute(
            text("""
                INSERT INTO jobs (
                    job_id, owner_tenant_id, job_type, target_type,
                    target_id, status, priority, attempt_count,
                    max_attempts, scheduled_at, started_at, completed_at,
                    result_json, created_at
                ) VALUES (
                    :jid, :tid, 'ingest', 'source', :sid, 'completed',
                    0, 1, 1, :now, :now, :now,
                    CAST(:result AS jsonb), :now
                )
            """),
            {
                "jid": job_id,
                "tid": tenant_id,
                "sid": source_id,
                "now": now,
                "result": json.dumps({
                    "segments_created": len(segment_ids),
                    "source_type": source_type,
                    "parser": type(parser).__name__,
                }),
            },
        )
        await db.commit()

        return {
            "status": "completed",
            "source_id": source_id,
            "job_id": job_id,
            "segments_created": len(segment_ids),
            "segments_embedded": embed_result.get("embedded", 0),
            "manifold_computed": embed_result.get("manifold_computed", 0),
            "file_size_bytes": reg_result.get("file_size_bytes"),
            "checksum": reg_result.get("checksum"),
        }

    except Exception as exc:
        logger.error("Parse failed for %s: %s", source_id, exc, exc_info=True)
        await update_parse_status(db, source_id, "failed")
        raise HTTPException(status_code=500, detail=f"Parse failed: {exc}")

    finally:
        # Clean up temp upload
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


@router.post("/source/async")
async def ingest_source_async(
    file_path: str = Form(...),
    source_type: str = Form("markdown"),
    title: Optional[str] = Form(None),
    tenant_id: str = Form("shared"),
    metadata_json: str = Form("{}"),
    db: AsyncSession = Depends(get_db),
):
    """
    Register a source and queue it for async parsing via Celery.

    Returns immediately with a job_id for status polling.
    """
    try:
        metadata = json.loads(metadata_json)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid metadata_json.")

    # Validate tenant
    result = await db.execute(
        text("SELECT tenant_id FROM tenants WHERE tenant_id = :tid"),
        {"tid": tenant_id},
    )
    if not result.fetchone():
        raise HTTPException(status_code=404, detail=f"Tenant '{tenant_id}' not found.")

    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    # Register source
    reg_result = await register_source(
        db=db,
        tenant_id=tenant_id,
        file_path=file_path,
        source_type=source_type,
        title=title,
        metadata=metadata,
    )

    if reg_result.get("is_duplicate"):
        return {
            "status": "duplicate",
            "source_id": reg_result["source_id"],
            "message": reg_result["message"],
        }

    source_id = reg_result["source_id"]

    # Create pending job
    job_id = f"JOB_INGEST_{uuid.uuid4().hex[:8]}"
    now = datetime.now(timezone.utc)
    await db.execute(
        text("""
            INSERT INTO jobs (
                job_id, owner_tenant_id, job_type, target_type,
                target_id, status, priority, attempt_count,
                max_attempts, scheduled_at, created_at
            ) VALUES (
                :jid, :tid, 'ingest', 'source', :sid, 'pending',
                0, 0, 3, :now, :now
            )
        """),
        {"jid": job_id, "tid": tenant_id, "sid": source_id, "now": now},
    )
    await db.commit()

    # Queue Celery task
    try:
        from workers.celery_app import celery_app

        celery_app.send_task(
            "gami.parse_source",
            args=[source_id, reg_result["file_path"], source_type, tenant_id],
            kwargs={"job_id": job_id},
        )
    except Exception as exc:
        logger.error("Failed to queue Celery task: %s", exc)
        # Job stays pending — can be retried

    return {
        "status": "queued",
        "source_id": source_id,
        "job_id": job_id,
    }


@router.get("/status/{job_id}")
async def ingestion_status(
    job_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Return the current status of an ingestion job."""
    result = await db.execute(
        text(
            "SELECT job_id, job_type, target_id, status, "
            "attempt_count, max_attempts, started_at, completed_at, "
            "result_json, error_json, created_at "
            "FROM jobs WHERE job_id = :jid"
        ),
        {"jid": job_id},
    )
    row = result.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return dict(row._mapping)


@router.get("/sources")
async def list_sources_endpoint(
    tenant_id: str = "shared",
    source_type: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    """List all sources for a tenant."""
    sources = await list_sources(db, tenant_id, source_type, limit, offset)
    return {"tenant_id": tenant_id, "count": len(sources), "sources": sources}


@router.get("/sources/{source_id}")
async def get_source_endpoint(
    source_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get details of a single source."""
    source = await get_source(db, source_id)
    if not source:
        raise HTTPException(status_code=404, detail=f"Source '{source_id}' not found.")
    return source


@router.get("/sources/{source_id}/segments")
async def get_segments_endpoint(
    source_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get all segments for a source."""
    segs = await get_segments(db, source_id)
    return {"source_id": source_id, "count": len(segs), "segments": segs}


@router.get("/segments/{segment_id}")
async def get_segment_endpoint(
    segment_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get a single segment by ID."""
    seg = await get_segment(db, segment_id)
    if not seg:
        raise HTTPException(
            status_code=404, detail=f"Segment '{segment_id}' not found."
        )
    return seg
