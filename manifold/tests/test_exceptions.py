"""Unit tests for custom exceptions."""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from manifold.exceptions import (
    ManifoldError,
    ErrorCode,
    QueryError,
    RetrievalError,
    EmbeddingError,
    StorageError,
    TenantError,
    query_empty,
    query_too_long,
    embedding_service_unavailable,
    tenant_not_found,
    tenant_access_denied,
)


class TestErrorCode:
    """Tests for error codes."""

    def test_error_codes_unique(self):
        """All error codes should have unique values."""
        values = [e.value for e in ErrorCode]
        assert len(values) == len(set(values))

    def test_error_code_categories(self):
        """Error codes should be organized by category."""
        # Query errors: 1xxx
        assert 1000 <= ErrorCode.INVALID_QUERY.value < 2000
        # Classification errors: 2xxx
        assert 2000 <= ErrorCode.CLASSIFICATION_FAILED.value < 3000
        # Retrieval errors: 3xxx
        assert 3000 <= ErrorCode.RETRIEVAL_FAILED.value < 4000


class TestManifoldError:
    """Tests for base ManifoldError."""

    def test_basic_error(self):
        """Should create basic error."""
        error = ManifoldError(
            "Something went wrong",
            ErrorCode.RETRIEVAL_FAILED,
        )

        assert str(error) == "[RETRIEVAL_FAILED] Something went wrong"
        assert error.code == ErrorCode.RETRIEVAL_FAILED

    def test_error_with_details(self):
        """Should include details."""
        error = ManifoldError(
            "Failed",
            ErrorCode.DATABASE_ERROR,
            details={"table": "segments", "operation": "insert"},
        )

        assert error.details["table"] == "segments"

    def test_error_with_cause(self):
        """Should chain exceptions."""
        cause = ValueError("Original error")
        error = ManifoldError(
            "Wrapper error",
            ErrorCode.EXTRACTION_FAILED,
            cause=cause,
        )

        assert error.cause is cause

    def test_to_dict(self):
        """Should convert to dictionary."""
        error = ManifoldError(
            "Test error",
            ErrorCode.INVALID_QUERY,
            details={"key": "value"},
        )

        d = error.to_dict()
        assert d["error"] == "Test error"
        assert d["code"] == ErrorCode.INVALID_QUERY.value
        assert d["code_name"] == "INVALID_QUERY"
        assert d["details"]["key"] == "value"


class TestQueryError:
    """Tests for QueryError."""

    def test_with_query(self):
        """Should include truncated query in details."""
        error = QueryError(
            "Invalid query",
            query="a" * 200,  # Long query
        )

        # Should be truncated to 100 chars
        assert len(error.details["query"]) == 100


class TestRetrievalError:
    """Tests for RetrievalError."""

    def test_with_manifold(self):
        """Should include manifold in details."""
        error = RetrievalError(
            "Search failed",
            manifold="topic",
        )

        assert error.details["manifold"] == "topic"


class TestConvenienceFunctions:
    """Tests for convenience error functions."""

    def test_query_empty(self):
        """query_empty should return correct error."""
        error = query_empty()

        assert isinstance(error, QueryError)
        assert error.code == ErrorCode.EMPTY_QUERY

    def test_query_too_long(self):
        """query_too_long should include lengths."""
        error = query_too_long(5000, 2000)

        assert isinstance(error, QueryError)
        assert error.code == ErrorCode.QUERY_TOO_LONG
        assert error.details["length"] == 5000
        assert error.details["max_length"] == 2000

    def test_embedding_service_unavailable(self):
        """embedding_service_unavailable should include cause."""
        cause = ConnectionError("Connection refused")
        error = embedding_service_unavailable("nomic-embed-text", cause)

        assert isinstance(error, EmbeddingError)
        assert error.code == ErrorCode.EMBEDDING_SERVICE_UNAVAILABLE
        assert error.cause is cause

    def test_tenant_not_found(self):
        """tenant_not_found should include tenant_id."""
        error = tenant_not_found("unknown-tenant")

        assert isinstance(error, TenantError)
        assert error.code == ErrorCode.TENANT_NOT_FOUND
        assert error.details["tenant_id"] == "unknown-tenant"

    def test_tenant_access_denied(self):
        """tenant_access_denied should include resource."""
        error = tenant_access_denied("tenant-1", "segment-123")

        assert isinstance(error, TenantError)
        assert error.code == ErrorCode.TENANT_ACCESS_DENIED
        assert error.details["resource"] == "segment-123"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
