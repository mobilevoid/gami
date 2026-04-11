"""Prometheus metrics for multi-manifold retrieval.

Exports metrics for monitoring retrieval quality, latency,
and system health.
"""
import time
from contextlib import contextmanager
from typing import Optional
from functools import wraps

# Stub metrics for isolated module
# In production, would use prometheus_client


class Counter:
    """Stub counter metric."""
    def __init__(self, name: str, description: str, labels: list = None):
        self.name = name
        self._value = 0
        self._labels = {}

    def labels(self, **kwargs):
        key = tuple(sorted(kwargs.items()))
        if key not in self._labels:
            self._labels[key] = Counter(self.name, "", [])
        return self._labels[key]

    def inc(self, amount: float = 1):
        self._value += amount


class Histogram:
    """Stub histogram metric."""
    def __init__(self, name: str, description: str, labels: list = None, buckets: list = None):
        self.name = name
        self._observations = []
        self._labels = {}

    def labels(self, **kwargs):
        key = tuple(sorted(kwargs.items()))
        if key not in self._labels:
            self._labels[key] = Histogram(self.name, "", [])
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
    def __init__(self, name: str, description: str, labels: list = None):
        self.name = name
        self._value = 0
        self._labels = {}

    def labels(self, **kwargs):
        key = tuple(sorted(kwargs.items()))
        if key not in self._labels:
            self._labels[key] = Gauge(self.name, "", [])
        return self._labels[key]

    def set(self, value: float):
        self._value = value

    def inc(self, amount: float = 1):
        self._value += amount

    def dec(self, amount: float = 1):
        self._value -= amount


# === Retrieval Metrics ===

RECALL_REQUESTS = Counter(
    "manifold_recall_requests_total",
    "Total number of recall requests",
    ["tenant_id", "mode"],
)

RECALL_LATENCY = Histogram(
    "manifold_recall_latency_seconds",
    "Recall request latency",
    ["tenant_id", "mode", "cached"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)

RECALL_RESULTS = Histogram(
    "manifold_recall_results_count",
    "Number of results returned per recall",
    ["tenant_id", "mode"],
    buckets=[1, 5, 10, 20, 50, 100],
)

CACHE_HITS = Counter(
    "manifold_cache_hits_total",
    "Cache hit count",
    ["cache_type"],
)

CACHE_MISSES = Counter(
    "manifold_cache_misses_total",
    "Cache miss count",
    ["cache_type"],
)


# === Score Metrics ===

FUSED_SCORE_DISTRIBUTION = Histogram(
    "manifold_fused_score",
    "Distribution of fused scores",
    ["mode"],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

MANIFOLD_SCORE_DISTRIBUTION = Histogram(
    "manifold_manifold_score",
    "Distribution of individual manifold scores",
    ["manifold"],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)


# === Classification Metrics ===

QUERY_CLASSIFICATION = Counter(
    "manifold_query_classifications_total",
    "Query classifications by mode",
    ["mode"],
)

CLASSIFICATION_CONFIDENCE = Histogram(
    "manifold_classification_confidence",
    "Classification confidence distribution",
    ["mode"],
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)


# === Shadow Mode Metrics ===

SHADOW_COMPARISONS = Counter(
    "manifold_shadow_comparisons_total",
    "Shadow mode comparison count",
    ["result"],
)

SHADOW_OVERLAP_RATIO = Histogram(
    "manifold_shadow_overlap_ratio",
    "Overlap ratio between old and new systems",
    [],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)


# === Promotion Metrics ===

PROMOTION_SCORE_DISTRIBUTION = Histogram(
    "manifold_promotion_score",
    "Distribution of promotion scores",
    ["object_type"],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

PROMOTED_OBJECTS = Gauge(
    "manifold_promoted_objects_total",
    "Number of promoted objects by type",
    ["object_type"],
)


# === Embedding Metrics ===

EMBEDDING_LATENCY = Histogram(
    "manifold_embedding_latency_seconds",
    "Embedding generation latency",
    ["manifold"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
)

EMBEDDING_BATCH_SIZE = Histogram(
    "manifold_embedding_batch_size",
    "Embedding batch sizes",
    [],
    buckets=[1, 10, 25, 50, 100, 200],
)


# === Error Metrics ===

ERRORS = Counter(
    "manifold_errors_total",
    "Error count by type",
    ["error_type", "component"],
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
