"""Input validation for manifold system.

Provides validation functions for all inputs to ensure
data integrity and security.
"""
import re
from typing import List, Optional, Any, Dict
from dataclasses import dataclass

from .exceptions import (
    QueryError,
    ConfigError,
    TenantError,
    ErrorCode,
    query_empty,
    query_too_long,
)
from .models.schemas import QueryModeV2, ManifoldWeights


# Constants
MAX_QUERY_LENGTH = 2000
MAX_TOP_K = 100
MIN_TOP_K = 1
MAX_BATCH_SIZE = 500
MAX_TEXT_LENGTH = 50000
VALID_MANIFOLDS = {"topic", "claim", "procedure", "relation", "time", "evidence"}
TENANT_ID_PATTERN = re.compile(r'^[a-z0-9][a-z0-9_-]{0,62}[a-z0-9]$|^[a-z0-9]$')
OBJECT_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_-]{1,128}$')


@dataclass
class ValidationResult:
    """Result of validation."""
    valid: bool
    errors: List[str]
    warnings: List[str]

    @classmethod
    def success(cls) -> "ValidationResult":
        return cls(valid=True, errors=[], warnings=[])

    @classmethod
    def failure(cls, *errors: str) -> "ValidationResult":
        return cls(valid=False, errors=list(errors), warnings=[])

    def add_warning(self, warning: str) -> "ValidationResult":
        self.warnings.append(warning)
        return self


def validate_query(query: Any) -> str:
    """Validate and normalize a query string.

    Args:
        query: The query to validate.

    Returns:
        Normalized query string.

    Raises:
        QueryError: If query is invalid.
    """
    if query is None:
        raise query_empty()

    if not isinstance(query, str):
        raise QueryError(
            f"Query must be a string, got {type(query).__name__}",
            ErrorCode.INVALID_QUERY,
        )

    query = query.strip()

    if not query:
        raise query_empty()

    if len(query) > MAX_QUERY_LENGTH:
        raise query_too_long(len(query), MAX_QUERY_LENGTH)

    return query


def validate_top_k(top_k: Any) -> int:
    """Validate and clamp top_k parameter.

    Args:
        top_k: The top_k value to validate.

    Returns:
        Clamped integer value.
    """
    if top_k is None:
        return 20  # Default

    try:
        top_k = int(top_k)
    except (ValueError, TypeError):
        return 20

    return max(MIN_TOP_K, min(MAX_TOP_K, top_k))


def validate_tenant_id(tenant_id: Any) -> str:
    """Validate tenant ID format.

    Args:
        tenant_id: The tenant ID to validate.

    Returns:
        Validated tenant ID.

    Raises:
        TenantError: If tenant ID is invalid.
    """
    if tenant_id is None:
        return "shared"

    if not isinstance(tenant_id, str):
        raise TenantError(
            f"Tenant ID must be a string, got {type(tenant_id).__name__}",
            ErrorCode.TENANT_NOT_FOUND,
        )

    tenant_id = tenant_id.strip().lower()

    if not tenant_id:
        return "shared"

    if not TENANT_ID_PATTERN.match(tenant_id):
        raise TenantError(
            f"Invalid tenant ID format: {tenant_id}",
            ErrorCode.TENANT_NOT_FOUND,
            tenant_id=tenant_id,
        )

    return tenant_id


def validate_object_id(object_id: Any) -> str:
    """Validate object ID format.

    Args:
        object_id: The object ID to validate.

    Returns:
        Validated object ID.

    Raises:
        QueryError: If object ID is invalid.
    """
    if object_id is None or not isinstance(object_id, str):
        raise QueryError(
            "Object ID is required",
            ErrorCode.INVALID_QUERY,
        )

    object_id = object_id.strip()

    if not object_id:
        raise QueryError(
            "Object ID cannot be empty",
            ErrorCode.INVALID_QUERY,
        )

    if not OBJECT_ID_PATTERN.match(object_id):
        raise QueryError(
            f"Invalid object ID format: {object_id}",
            ErrorCode.INVALID_QUERY,
        )

    return object_id


def validate_manifold_type(manifold: Any) -> str:
    """Validate manifold type.

    Args:
        manifold: The manifold type to validate.

    Returns:
        Validated manifold type.

    Raises:
        QueryError: If manifold type is invalid.
    """
    if manifold is None:
        return "topic"

    if not isinstance(manifold, str):
        return "topic"

    manifold = manifold.strip().lower()

    if manifold not in VALID_MANIFOLDS:
        return "topic"

    return manifold


def validate_query_mode(mode: Any) -> Optional[QueryModeV2]:
    """Validate query mode override.

    Args:
        mode: The mode to validate.

    Returns:
        QueryModeV2 if valid, None otherwise.
    """
    if mode is None:
        return None

    if isinstance(mode, QueryModeV2):
        return mode

    if not isinstance(mode, str):
        return None

    try:
        return QueryModeV2(mode.lower())
    except ValueError:
        return None


def validate_manifold_weights(weights: Any) -> Optional[ManifoldWeights]:
    """Validate manifold weight override.

    Args:
        weights: The weights to validate (dict or ManifoldWeights).

    Returns:
        ManifoldWeights if valid, None otherwise.
    """
    if weights is None:
        return None

    if isinstance(weights, ManifoldWeights):
        return weights

    if not isinstance(weights, dict):
        return None

    try:
        # Extract weights from dict
        topic = float(weights.get("topic", 0))
        claim = float(weights.get("claim", 0))
        procedure = float(weights.get("procedure", 0))
        relation = float(weights.get("relation", 0))
        time = float(weights.get("time", 0))
        evidence = float(weights.get("evidence", 0))

        # Clamp to [0, 1]
        topic = max(0, min(1, topic))
        claim = max(0, min(1, claim))
        procedure = max(0, min(1, procedure))
        relation = max(0, min(1, relation))
        time = max(0, min(1, time))
        evidence = max(0, min(1, evidence))

        return ManifoldWeights(
            topic=topic,
            claim=claim,
            procedure=procedure,
            relation=relation,
            time=time,
            evidence=evidence,
        )
    except (ValueError, TypeError):
        return None


def validate_text_for_embedding(text: Any) -> str:
    """Validate text for embedding generation.

    Args:
        text: The text to validate.

    Returns:
        Validated text.

    Raises:
        QueryError: If text is invalid.
    """
    if text is None:
        raise QueryError(
            "Text is required for embedding",
            ErrorCode.INVALID_QUERY,
        )

    if not isinstance(text, str):
        raise QueryError(
            f"Text must be a string, got {type(text).__name__}",
            ErrorCode.INVALID_QUERY,
        )

    text = text.strip()

    if not text:
        raise QueryError(
            "Text cannot be empty",
            ErrorCode.INVALID_QUERY,
        )

    if len(text) > MAX_TEXT_LENGTH:
        # Truncate rather than reject
        text = text[:MAX_TEXT_LENGTH]

    return text


def validate_batch_size(size: Any) -> int:
    """Validate batch size parameter.

    Args:
        size: The batch size to validate.

    Returns:
        Validated batch size.
    """
    if size is None:
        return 100  # Default

    try:
        size = int(size)
    except (ValueError, TypeError):
        return 100

    return max(1, min(MAX_BATCH_SIZE, size))


def validate_threshold(threshold: Any, default: float = 0.3) -> float:
    """Validate similarity threshold.

    Args:
        threshold: The threshold to validate.
        default: Default value if invalid.

    Returns:
        Validated threshold in [0, 1].
    """
    if threshold is None:
        return default

    try:
        threshold = float(threshold)
    except (ValueError, TypeError):
        return default

    return max(0.0, min(1.0, threshold))


def validate_config(config: Dict[str, Any]) -> ValidationResult:
    """Validate configuration dictionary.

    Args:
        config: Configuration to validate.

    Returns:
        ValidationResult with any errors/warnings.
    """
    errors = []
    warnings = []

    # Check weight sums
    alpha_sum = (
        config.get("alpha_topic_default", 0)
        + config.get("alpha_claim_default", 0)
        + config.get("alpha_procedure_default", 0)
        + config.get("alpha_relation_default", 0)
        + config.get("alpha_time_default", 0)
        + config.get("alpha_evidence_default", 0)
    )
    if abs(alpha_sum - 1.0) > 0.01:
        errors.append(f"Alpha weights sum to {alpha_sum}, should be 1.0")

    # Check thresholds
    promo = config.get("promotion_threshold", 0.65)
    demo = config.get("demotion_threshold", 0.35)
    if promo <= demo:
        errors.append("Promotion threshold must be > demotion threshold")

    # Check bounds
    if config.get("similarity_threshold", 0.3) < 0:
        errors.append("Similarity threshold must be >= 0")

    if config.get("max_top_k", 100) < 1:
        errors.append("max_top_k must be >= 1")

    # Warnings
    if config.get("shadow_mode_enabled", False):
        warnings.append("Shadow mode is enabled - will increase latency")

    if errors:
        return ValidationResult(valid=False, errors=errors, warnings=warnings)

    return ValidationResult(valid=True, errors=[], warnings=warnings)
