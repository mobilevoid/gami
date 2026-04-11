"""API Metrics Middleware for GAMI.

Provides request timing, counting, and error tracking middleware
that integrates with the manifold metrics system.
"""
import time
import logging
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from manifold.metrics import (
    track_api_request,
    track_error,
    ACTIVE_REQUESTS,
)

logger = logging.getLogger("gami.api.middleware.metrics")


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to track API request metrics.

    Tracks:
    - Request count by method, path, status
    - Request latency by method, path
    - Active request count
    - Errors by type
    """

    # Paths to skip metrics for (health checks, metrics endpoint itself)
    SKIP_PATHS = {"/health", "/metrics", "/health/", "/metrics/"}

    # Normalize path patterns to avoid cardinality explosion
    PATH_NORMALIZATIONS = [
        # Replace UUIDs with placeholder
        (r"/[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", "/{uuid}"),
        # Replace segment IDs (SEG_xxx)
        (r"/SEG_[a-zA-Z0-9]+", "/{segment_id}"),
        # Replace entity IDs (ENT_xxx)
        (r"/ENT_[a-zA-Z0-9]+", "/{entity_id}"),
        # Replace tenant IDs in paths
        (r"/tenant/[^/]+", "/tenant/{tenant_id}"),
    ]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Track request metrics."""
        # Skip metrics for certain paths
        if request.url.path in self.SKIP_PATHS:
            return await call_next(request)

        # Normalize path for metrics
        path = self._normalize_path(request.url.path)
        method = request.method

        # Track active requests
        ACTIVE_REQUESTS.inc()
        start_time = time.perf_counter()

        try:
            response = await call_next(request)
            latency = time.perf_counter() - start_time

            # Track request metrics
            track_api_request(
                method=method,
                path=path,
                status=response.status_code,
                latency=latency,
            )

            # Log slow requests
            if latency > 2.0:
                logger.warning(
                    f"Slow request: {method} {path} took {latency:.2f}s "
                    f"(status={response.status_code})"
                )

            return response

        except Exception as e:
            latency = time.perf_counter() - start_time

            # Track error
            track_api_request(
                method=method,
                path=path,
                status=500,
                latency=latency,
            )
            track_error(
                error_type=type(e).__name__,
                component="api",
            )

            logger.exception(f"Request failed: {method} {path}")
            raise

        finally:
            ACTIVE_REQUESTS.dec()

    def _normalize_path(self, path: str) -> str:
        """Normalize path to avoid high cardinality.

        Replaces dynamic segments (UUIDs, IDs) with placeholders.
        """
        import re

        normalized = path
        for pattern, replacement in self.PATH_NORMALIZATIONS:
            normalized = re.sub(pattern, replacement, normalized)

        return normalized


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware to add request ID header for tracing."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add request ID to response headers."""
        import uuid

        # Use existing request ID or generate new one
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

        # Store in request state for access in handlers
        request.state.request_id = request_id

        response = await call_next(request)

        # Add to response headers
        response.headers["X-Request-ID"] = request_id

        return response
