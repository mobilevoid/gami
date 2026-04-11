"""Custom exceptions for the multi-manifold system.

Provides structured error handling with error codes for
debugging and monitoring.
"""
from enum import Enum
from typing import Optional, Dict, Any


class ErrorCode(Enum):
    """Error codes for manifold system errors."""

    # Query errors (1xxx)
    INVALID_QUERY = 1001
    EMPTY_QUERY = 1002
    QUERY_TOO_LONG = 1003
    UNSUPPORTED_MODE = 1004

    # Classification errors (2xxx)
    CLASSIFICATION_FAILED = 2001
    INVALID_WEIGHTS = 2002
    WEIGHT_SUM_ERROR = 2003

    # Retrieval errors (3xxx)
    RETRIEVAL_FAILED = 3001
    INDEX_NOT_AVAILABLE = 3002
    GRAPH_QUERY_FAILED = 3003
    TIMEOUT = 3004
    NO_RESULTS = 3005

    # Embedding errors (4xxx)
    EMBEDDING_FAILED = 4001
    EMBEDDING_SERVICE_UNAVAILABLE = 4002
    EMBEDDING_DIMENSION_MISMATCH = 4003
    BATCH_TOO_LARGE = 4004

    # Extraction errors (5xxx)
    EXTRACTION_FAILED = 5001
    LLM_SERVICE_UNAVAILABLE = 5002
    INVALID_EXTRACTION_RESULT = 5003
    CANONICAL_FORM_ERROR = 5004

    # Storage errors (6xxx)
    DATABASE_ERROR = 6001
    CONNECTION_FAILED = 6002
    TRANSACTION_FAILED = 6003
    CONSTRAINT_VIOLATION = 6004

    # Cache errors (7xxx)
    CACHE_ERROR = 7001
    CACHE_MISS = 7002
    CACHE_SERIALIZATION_ERROR = 7003

    # Config errors (8xxx)
    CONFIG_ERROR = 8001
    INVALID_CONFIG = 8002
    MISSING_CONFIG = 8003

    # Tenant errors (9xxx)
    TENANT_NOT_FOUND = 9001
    TENANT_ACCESS_DENIED = 9002
    TENANT_QUOTA_EXCEEDED = 9003


class ManifoldError(Exception):
    """Base exception for manifold system errors."""

    def __init__(
        self,
        message: str,
        code: ErrorCode,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}
        self.cause = cause

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "error": self.message,
            "code": self.code.value,
            "code_name": self.code.name,
            "details": self.details,
        }

    def __str__(self) -> str:
        return f"[{self.code.name}] {self.message}"


class QueryError(ManifoldError):
    """Errors related to query processing."""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.INVALID_QUERY,
        query: Optional[str] = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        if query:
            details["query"] = query[:100]  # Truncate for logging
        super().__init__(message, code, details, **kwargs)


class ClassificationError(ManifoldError):
    """Errors during query classification."""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.CLASSIFICATION_FAILED,
        **kwargs,
    ):
        super().__init__(message, code, **kwargs)


class RetrievalError(ManifoldError):
    """Errors during retrieval operations."""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.RETRIEVAL_FAILED,
        manifold: Optional[str] = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        if manifold:
            details["manifold"] = manifold
        super().__init__(message, code, details, **kwargs)


class EmbeddingError(ManifoldError):
    """Errors during embedding generation."""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.EMBEDDING_FAILED,
        model: Optional[str] = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        if model:
            details["model"] = model
        super().__init__(message, code, details, **kwargs)


class ExtractionError(ManifoldError):
    """Errors during structured extraction."""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.EXTRACTION_FAILED,
        extraction_type: Optional[str] = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        if extraction_type:
            details["extraction_type"] = extraction_type
        super().__init__(message, code, details, **kwargs)


class StorageError(ManifoldError):
    """Errors during database operations."""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.DATABASE_ERROR,
        table: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        if table:
            details["table"] = table
        if operation:
            details["operation"] = operation
        super().__init__(message, code, details, **kwargs)


class CacheError(ManifoldError):
    """Errors during cache operations."""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.CACHE_ERROR,
        key: Optional[str] = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        if key:
            details["key"] = key[:50]  # Truncate
        super().__init__(message, code, details, **kwargs)


class ConfigError(ManifoldError):
    """Errors in configuration."""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.CONFIG_ERROR,
        config_key: Optional[str] = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        if config_key:
            details["config_key"] = config_key
        super().__init__(message, code, details, **kwargs)


class TenantError(ManifoldError):
    """Errors related to tenant operations."""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.TENANT_NOT_FOUND,
        tenant_id: Optional[str] = None,
        **kwargs,
    ):
        details = kwargs.pop("details", {})
        if tenant_id:
            details["tenant_id"] = tenant_id
        super().__init__(message, code, details, **kwargs)


# Convenience functions for common errors

def query_empty() -> QueryError:
    """Create error for empty query."""
    return QueryError(
        "Query cannot be empty",
        ErrorCode.EMPTY_QUERY,
    )


def query_too_long(length: int, max_length: int) -> QueryError:
    """Create error for query exceeding max length."""
    return QueryError(
        f"Query length {length} exceeds maximum {max_length}",
        ErrorCode.QUERY_TOO_LONG,
        details={"length": length, "max_length": max_length},
    )


def embedding_service_unavailable(model: str, cause: Exception) -> EmbeddingError:
    """Create error for unavailable embedding service."""
    return EmbeddingError(
        f"Embedding service unavailable for model {model}",
        ErrorCode.EMBEDDING_SERVICE_UNAVAILABLE,
        model=model,
        cause=cause,
    )


def tenant_not_found(tenant_id: str) -> TenantError:
    """Create error for missing tenant."""
    return TenantError(
        f"Tenant '{tenant_id}' not found",
        ErrorCode.TENANT_NOT_FOUND,
        tenant_id=tenant_id,
    )


def tenant_access_denied(tenant_id: str, resource: str) -> TenantError:
    """Create error for access denied."""
    return TenantError(
        f"Tenant '{tenant_id}' does not have access to {resource}",
        ErrorCode.TENANT_ACCESS_DENIED,
        tenant_id=tenant_id,
        details={"resource": resource},
    )
