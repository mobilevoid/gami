"""GAMI — Graph-Augmented Memory Interface.

FastAPI application with ingestion, auth, and memory retrieval routers.
"""
import logging
import traceback
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.config import settings

logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL))
logger = logging.getLogger("gami")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: initialize and tear down resources."""
    logger.info("GAMI starting up...")

    # Initialize async database engine
    from api.services.db import async_engine

    # Verify DB connectivity
    try:
        async with async_engine.connect() as conn:
            result = await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
            result.fetchone()
        logger.info("Database connection verified (async engine).")
    except Exception as exc:
        logger.error("Database connection failed: %s", exc)

    # Verify Redis connectivity
    try:
        import redis

        r = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
        )
        r.ping()
        r.close()
        logger.info("Redis connection verified.")
    except Exception as exc:
        logger.warning("Redis not available: %s", exc)

    # Ensure storage directories exist
    import os

    for subdir in ("raw", "uploads", "parsed"):
        os.makedirs(os.path.join(settings.OBJECT_STORE, subdir), exist_ok=True)

    yield

    logger.info("GAMI shutting down...")
    from api.services.db import async_engine

    await async_engine.dispose()


app = FastAPI(
    title="GAMI — Graph-Augmented Memory Interface",
    version="0.1.0",
    description="Polyglot memory store with graph, vector, and lexical retrieval",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Global error handler
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(
        "Unhandled exception on %s %s: %s",
        request.method, request.url.path, exc,
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error": str(exc),
            "path": str(request.url.path),
        },
    )


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Basic health check."""
    return {
        "status": "ok",
        "service": "gami",
        "version": "0.1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/health/deep")
async def health_deep():
    """Deep health check — verifies DB and Redis."""
    checks = {"db": "unknown", "redis": "unknown"}

    # Check DB
    try:
        from api.services.db import async_engine

        async with async_engine.connect() as conn:
            result = await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
            result.fetchone()
        checks["db"] = "ok"
    except Exception as exc:
        checks["db"] = f"error: {exc}"

    # Check Redis
    try:
        import redis

        r = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
        )
        r.ping()
        r.close()
        checks["redis"] = "ok"
    except Exception as exc:
        checks["redis"] = f"error: {exc}"

    overall = "ok" if all(v == "ok" for v in checks.values()) else "degraded"
    return {
        "status": overall,
        "service": "gami",
        "version": "0.1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
    }


# ---------------------------------------------------------------------------
# Include routers
# ---------------------------------------------------------------------------

from api.routers.ingest import router as ingest_router
from api.routers.auth import router as auth_router
from api.routers.memory import router as memory_router
from api.routers.graph import router as graph_router
from api.routers.admin import router as admin_router

app.include_router(ingest_router, prefix="/api/v1/ingest", tags=["ingest"])
app.include_router(auth_router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(memory_router, prefix="/api/v1/memory", tags=["memory"])
app.include_router(graph_router, prefix="/api/v1/graph", tags=["graph"])
app.include_router(admin_router, prefix="/api/v1/admin", tags=["admin"])


# ---------------------------------------------------------------------------
# Prometheus metrics endpoint
# ---------------------------------------------------------------------------

import time as _time
import os as _os

from fastapi.responses import PlainTextResponse
from sqlalchemy import text as _sql_text

_start_time = _time.time()


@app.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics():
    """Prometheus-compatible metrics endpoint."""
    from api.services.db import AsyncSessionLocal

    lines: list[str] = []

    def _g(name: str, value, help_text: str = "", labels: str = ""):
        """Emit a gauge metric."""
        if help_text:
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} gauge")
        lbl = f"{{{labels}}}" if labels else ""
        lines.append(f"{name}{lbl} {value}")

    try:
        async with AsyncSessionLocal() as db:
            # Core counts
            for tbl, mname in [
                ("segments", "gami_segments_total"),
                ("entities", "gami_entities_total"),
                ("claims", "gami_claims_total"),
                ("relations", "gami_relations_total"),
                ("events", "gami_events_total"),
                ("sources", "gami_sources_total"),
                ("assistant_memories", "gami_memories_total"),
            ]:
                try:
                    row = await db.execute(_sql_text(f"SELECT COUNT(*) FROM {tbl}"))
                    _g(mname, row.scalar() or 0, f"Total rows in {tbl}")
                except Exception:
                    _g(mname, 0, f"Total rows in {tbl}")

            # Storage tier breakdown
            for tbl in ("segments", "entities", "claims"):
                try:
                    rows = await db.execute(
                        _sql_text(
                            f"SELECT storage_tier, COUNT(*) FROM {tbl} "
                            f"GROUP BY storage_tier"
                        )
                    )
                    for r in rows.fetchall():
                        _g(f"gami_{tbl}_by_tier", r[1], "", f'tier="{r[0]}"')
                except Exception:
                    pass

            # Embedding progress
            try:
                total = (await db.execute(_sql_text(
                    "SELECT COUNT(*) FROM segments"
                ))).scalar() or 0
                embedded = (await db.execute(_sql_text(
                    "SELECT COUNT(*) FROM segments WHERE embedding IS NOT NULL"
                ))).scalar() or 0
                progress = embedded / total if total > 0 else 1.0
                _g("gami_embedding_progress", f"{progress:.4f}",
                   "Fraction of segments with embeddings")
            except Exception:
                _g("gami_embedding_progress", 0,
                   "Fraction of segments with embeddings")

            # Recall latency (from recent completed recall jobs)
            try:
                row = await db.execute(_sql_text(
                    "SELECT COUNT(*) FROM jobs WHERE job_type = 'ingest' "
                    "AND status = 'completed'"
                ))
                lines.append("# HELP gami_recall_requests_total Total completed ingest jobs")
                lines.append("# TYPE gami_recall_requests_total counter")
                lines.append(f"gami_recall_requests_total {row.scalar() or 0}")
            except Exception:
                pass

            # Pending proposals
            try:
                row = await db.execute(_sql_text(
                    "SELECT COUNT(*) FROM proposed_changes WHERE status = 'pending'"
                ))
                _g("gami_pending_proposals", row.scalar() or 0,
                   "Pending proposed changes")
            except Exception:
                pass

            # Job status distribution
            try:
                rows = await db.execute(_sql_text(
                    "SELECT status, COUNT(*) FROM jobs GROUP BY status"
                ))
                lines.append("# HELP gami_jobs Jobs by status")
                lines.append("# TYPE gami_jobs gauge")
                for r in rows.fetchall():
                    lines.append(f'gami_jobs{{status="{r[0]}"}} {r[1]}')
            except Exception:
                pass

            # Tenant count
            try:
                row = await db.execute(_sql_text("SELECT COUNT(*) FROM tenants"))
                _g("gami_tenants_total", row.scalar() or 0, "Total tenants")
            except Exception:
                pass

    except Exception as exc:
        lines.append(f"# ERROR database_query_failed {exc}")

    # Cold storage bytes
    try:
        cold_bytes = 0
        cold_store = settings.COLD_STORE
        if _os.path.isdir(cold_store):
            for root, dirs, files in _os.walk(cold_store):
                for f in files:
                    fp = _os.path.join(root, f)
                    if _os.path.isfile(fp):
                        cold_bytes += _os.path.getsize(fp)
        _g("gami_cold_storage_bytes", cold_bytes, "Total bytes in cold storage")
    except Exception:
        _g("gami_cold_storage_bytes", 0, "Total bytes in cold storage")

    # Uptime
    _g("gami_uptime_seconds", f"{_time.time() - _start_time:.0f}",
       "API server uptime in seconds")

    return "\n".join(lines) + "\n"
