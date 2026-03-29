import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

from api.config import settings

logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL))
logger = logging.getLogger("gami")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("GAMI starting up...")
    # TODO: Initialize database connections, Redis, etc.
    yield
    logger.info("GAMI shutting down...")

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

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "gami",
        "version": "0.1.0",
        "timestamp": datetime.utcnow().isoformat(),
    }

# TODO: Import and include routers in Phase 0 completion
# from api.routers import memory, ingest, graph, admin, auth
# app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
# app.include_router(memory.router, prefix="/api/v1/memory", tags=["memory"])
# app.include_router(ingest.router, prefix="/api/v1/ingest", tags=["ingest"])
# app.include_router(graph.router, prefix="/api/v1/graph", tags=["graph"])
# app.include_router(admin.router, prefix="/api/v1/admin", tags=["admin"])
