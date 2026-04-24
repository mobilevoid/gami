"""Prometheus metrics for multi-manifold retrieval.

Exports metrics for monitoring retrieval quality, latency,
and system health.

Metrics are available at /metrics endpoint when integrated
with FastAPI via prometheus_fastapi_instrumentator or manual
exposure.
"""
import time
import logging
from contextlib import contextmanager
from typing import Optional
from functools import wraps

logger = logging.getLogger("gami.manifold.metrics")

# Try to import prometheus_client, fall back to stubs if not available
try:
    from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, REGISTRY
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
    logger.info("prometheus_client available - using real metrics")
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not available - using stub metrics")

    # Stub implementations for when prometheus_client is not available
    class Counter:
        """Stub counter metric."""
        def __init__(self, name: str, description: str, labelnames: list = None, **kwargs):
            self.name = name
            self._value = 0
            self._labels = {}

        def labels(self, **kwargs):
            key = tuple(sorted(kwargs.items()))
            if key not in self._labels:
                self._labels[key] = Counter(self.name, "", labelnames=[])
            return self._labels[key]

        def inc(self, amount: float = 1):
            self._value += amount

    class Histogram:
        """Stub histogram metric."""
        def __init__(self, name: str, description: str, labelnames: list = None, buckets: list = None, **kwargs):
            self.name = name
            self._observations = []
            self._labels = {}

        def labels(self, **kwargs):
            key = tuple(sorted(kwargs.items()))
            if key not in self._labels:
                self._labels[key] = Histogram(self.name, "", labelnames=[])
            return self._labels[key]

        def observe(self, value: float):
            self._observations.append(value)

        @contextmanager
        def time(self):
            start = time.time()
            yield
            self.observe(time.time() - start)

    class Gauge:
        """Stub gauge metric."""
        def __init__(self, name: str, description: str, labelnames: list = None, **kwargs):
            self.name = name
            self._value = 0
            self._labels = {}

        def labels(self, **kwargs):
            key = tuple(sorted(kwargs.items()))
            if key not in self._labels:
                self._labels[key] = Gauge(self.name, "", labelnames=[])
            return self._labels[key]

        def set(self, value: float):
            self._value = value

        def inc(self, amount: float = 1):
            self._value += amount

        def dec(self, amount: float = 1):
            self._value -= amount

    REGISTRY = None

    def generate_latest(registry=None):
        return b"# prometheus_client not installed\n"

    CONTENT_TYPE_LATEST = "text/plain; charset=utf-8"


# === Retrieval Metrics ===

RECALL_REQUESTS = Counter(
    "manifold_recall_requests_total",
    "Total number of recall requests",
    labelnames=["tenant_id", "mode"],
)

RECALL_LATENCY = Histogram(
    "manifold_recall_latency_seconds",
    "Recall request latency",
    labelnames=["tenant_id", "mode", "cached"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)

RECALL_RESULTS = Histogram(
    "manifold_recall_results_count",
    "Number of results returned per recall",
    labelnames=["tenant_id", "mode"],
    buckets=[1, 5, 10, 20, 50, 100],
)

CACHE_HITS = Counter(
    "manifold_cache_hits_total",
    "Cache hit count",
    labelnames=["cache_type"],
)

CACHE_MISSES = Counter(
    "manifold_cache_misses_total",
    "Cache miss count",
    labelnames=["cache_type"],
)


# === Score Metrics ===

FUSED_SCORE_DISTRIBUTION = Histogram(
    "manifold_fused_score",
    "Distribution of fused scores",
    labelnames=["mode"],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

MANIFOLD_SCORE_DISTRIBUTION = Histogram(
    "manifold_manifold_score",
    "Distribution of individual manifold scores",
    labelnames=["manifold"],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)


# === Classification Metrics ===

QUERY_CLASSIFICATION = Counter(
    "manifold_query_classifications_total",
    "Query classifications by mode",
    labelnames=["mode"],
)

CLASSIFICATION_CONFIDENCE = Histogram(
    "manifold_classification_confidence",
    "Classification confidence distribution",
    labelnames=["mode"],
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)


# === Shadow Mode Metrics ===

SHADOW_COMPARISONS = Counter(
    "manifold_shadow_comparisons_total",
    "Shadow mode comparison count",
    labelnames=["result"],
)

SHADOW_OVERLAP_RATIO = Histogram(
    "manifold_shadow_overlap_ratio",
    "Overlap ratio between old and new systems",
    labelnames=[],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)


# === Promotion Metrics ===

PROMOTION_SCORE_DISTRIBUTION = Histogram(
    "manifold_promotion_score",
    "Distribution of promotion scores",
    labelnames=["object_type"],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

PROMOTED_OBJECTS = Gauge(
    "manifold_promoted_objects_total",
    "Number of promoted objects by type",
    labelnames=["object_type"],
)


# === Embedding Metrics ===

EMBEDDING_LATENCY = Histogram(
    "manifold_embedding_latency_seconds",
    "Embedding generation latency",
    labelnames=["manifold"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
)

EMBEDDING_BATCH_SIZE = Histogram(
    "manifold_embedding_batch_size",
    "Embedding batch sizes",
    labelnames=[],
    buckets=[1, 10, 25, 50, 100, 200],
)


# === Error Metrics ===

ERRORS = Counter(
    "manifold_errors_total",
    "Error count by type",
    labelnames=["error_type", "component"],
)


# === MCP Tool Metrics ===

MCP_TOOL_REQUESTS = Counter(
    "mcp_tool_requests_total",
    "Total MCP tool requests",
    labelnames=["tool", "status"],
)

MCP_TOOL_LATENCY = Histogram(
    "mcp_tool_latency_seconds",
    "MCP tool call latency",
    labelnames=["tool"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
)

MCP_TOOL_ERRORS = Counter(
    "mcp_tool_errors_total",
    "MCP tool errors by type",
    labelnames=["tool", "error_type"],
)


# === API Metrics ===

API_REQUEST_COUNT = Counter(
    "gami_api_requests_total",
    "Total API requests",
    labelnames=["method", "path", "status"],
)

API_REQUEST_LATENCY = Histogram(
    "gami_api_request_latency_seconds",
    "API request latency",
    labelnames=["method", "path"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)

ACTIVE_REQUESTS = Gauge(
    "gami_active_requests",
    "Number of active API requests",
    labelnames=[],
)


# === Manifold Embeddings Metrics ===

MANIFOLD_EMBEDDING_COUNT = Gauge(
    "manifold_embeddings_total",
    "Total manifold embeddings by type",
    labelnames=["manifold_type"],
)

MANIFOLD_SEARCH_LATENCY = Histogram(
    "manifold_search_latency_seconds",
    "Manifold search latency by type",
    labelnames=["manifold_type"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
)

MANIFOLD_FUSION_LATENCY = Histogram(
    "manifold_fusion_latency_seconds",
    "Multi-manifold result fusion latency",
    labelnames=[],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1],
)


# === Helper Functions ===

def track_recall(tenant_id: str, mode: str, cached: bool = False):
    """Decorator to track recall metrics."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            RECALL_REQUESTS.labels(tenant_id=tenant_id, mode=mode).inc()

            with RECALL_LATENCY.labels(
                tenant_id=tenant_id,
                mode=mode,
                cached=str(cached).lower(),
            ).time():
                result = await func(*args, **kwargs)

            # Track result count
            if hasattr(result, 'candidates'):
                RECALL_RESULTS.labels(
                    tenant_id=tenant_id,
                    mode=mode,
                ).observe(len(result.candidates))

            return result
        return wrapper
    return decorator


def track_classification(mode: str, confidence: float):
    """Track query classification."""
    QUERY_CLASSIFICATION.labels(mode=mode).inc()
    CLASSIFICATION_CONFIDENCE.labels(mode=mode).observe(confidence)


def track_shadow_comparison(result: str, overlap_ratio: float):
    """Track shadow mode comparison."""
    SHADOW_COMPARISONS.labels(result=result).inc()
    SHADOW_OVERLAP_RATIO.observe(overlap_ratio)


def track_error(error_type: str, component: str):
    """Track an error."""
    ERRORS.labels(error_type=error_type, component=component).inc()


def track_mcp_tool(tool: str, latency: float, success: bool, error_type: str = None):
    """Track MCP tool call metrics."""
    status = "success" if success else "error"
    MCP_TOOL_REQUESTS.labels(tool=tool, status=status).inc()
    MCP_TOOL_LATENCY.labels(tool=tool).observe(latency)
    if not success and error_type:
        MCP_TOOL_ERRORS.labels(tool=tool, error_type=error_type).inc()


def track_api_request(method: str, path: str, status: int, latency: float):
    """Track API request metrics."""
    API_REQUEST_COUNT.labels(method=method, path=path, status=str(status)).inc()
    API_REQUEST_LATENCY.labels(method=method, path=path).observe(latency)


def track_manifold_search(manifold_type: str, latency: float):
    """Track manifold search latency."""
    MANIFOLD_SEARCH_LATENCY.labels(manifold_type=manifold_type).observe(latency)


def track_manifold_fusion(latency: float):
    """Track manifold fusion latency."""
    MANIFOLD_FUSION_LATENCY.observe(latency)


def update_manifold_counts(counts: dict):
    """Update manifold embedding count gauges."""
    for manifold_type, count in counts.items():
        MANIFOLD_EMBEDDING_COUNT.labels(manifold_type=manifold_type).set(count)


def get_metrics():
    """Get Prometheus metrics output."""
    if PROMETHEUS_AVAILABLE:
        return generate_latest(REGISTRY)
    return b"# prometheus_client not installed\n"


def get_content_type():
    """Get Prometheus content type header."""
    return CONTENT_TYPE_LATEST
