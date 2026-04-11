"""Unit tests for input validation."""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from manifold.validation import (
    validate_query,
    validate_top_k,
    validate_tenant_id,
    validate_object_id,
    validate_manifold_type,
    validate_query_mode,
    validate_manifold_weights,
    validate_text_for_embedding,
    validate_batch_size,
    validate_threshold,
    validate_config,
    ValidationResult,
    MAX_QUERY_LENGTH,
    MAX_TOP_K,
    MIN_TOP_K,
)
from manifold.exceptions import QueryError, TenantError
from manifold.models.schemas import QueryModeV2, ManifoldWeights


class TestValidateQuery:
    """Tests for query validation."""

    def test_valid_query(self):
        """Should accept valid query."""
        result = validate_query("What is PostgreSQL?")
        assert result == "What is PostgreSQL?"

    def test_strips_whitespace(self):
        """Should strip leading/trailing whitespace."""
        result = validate_query("  test query  ")
        assert result == "test query"

    def test_rejects_none(self):
        """Should reject None."""
        with pytest.raises(QueryError):
            validate_query(None)

    def test_rejects_empty(self):
        """Should reject empty string."""
        with pytest.raises(QueryError):
            validate_query("")

    def test_rejects_whitespace_only(self):
        """Should reject whitespace-only string."""
        with pytest.raises(QueryError):
            validate_query("   ")

    def test_rejects_non_string(self):
        """Should reject non-string types."""
        with pytest.raises(QueryError):
            validate_query(123)

    def test_rejects_too_long(self):
        """Should reject query exceeding max length."""
        long_query = "x" * (MAX_QUERY_LENGTH + 1)
        with pytest.raises(QueryError):
            validate_query(long_query)

    def test_accepts_max_length(self):
        """Should accept query at exactly max length."""
        max_query = "x" * MAX_QUERY_LENGTH
        result = validate_query(max_query)
        assert len(result) == MAX_QUERY_LENGTH


class TestValidateTopK:
    """Tests for top_k validation."""

    def test_valid_value(self):
        """Should accept valid value."""
        assert validate_top_k(20) == 20

    def test_none_returns_default(self):
        """Should return default for None."""
        assert validate_top_k(None) == 20

    def test_clamps_high_value(self):
        """Should clamp values above max."""
        assert validate_top_k(1000) == MAX_TOP_K

    def test_clamps_low_value(self):
        """Should clamp values below min."""
        assert validate_top_k(0) == MIN_TOP_K
        assert validate_top_k(-5) == MIN_TOP_K

    def test_converts_string(self):
        """Should convert string to int."""
        assert validate_top_k("50") == 50

    def test_invalid_string_returns_default(self):
        """Should return default for invalid string."""
        assert validate_top_k("invalid") == 20


class TestValidateTenantId:
    """Tests for tenant ID validation."""

    def test_valid_tenant(self):
        """Should accept valid tenant ID."""
        assert validate_tenant_id("my-tenant") == "my-tenant"

    def test_none_returns_shared(self):
        """Should return 'shared' for None."""
        assert validate_tenant_id(None) == "shared"

    def test_empty_returns_shared(self):
        """Should return 'shared' for empty."""
        assert validate_tenant_id("") == "shared"

    def test_lowercases(self):
        """Should lowercase tenant ID."""
        assert validate_tenant_id("MyTenant") == "mytenant"

    def test_rejects_invalid_format(self):
        """Should reject invalid formats."""
        with pytest.raises(TenantError):
            validate_tenant_id("invalid tenant!")  # Has space and !

    def test_rejects_non_string(self):
        """Should reject non-string types."""
        with pytest.raises(TenantError):
            validate_tenant_id(123)

    def test_accepts_single_char(self):
        """Should accept single character."""
        assert validate_tenant_id("a") == "a"

    def test_accepts_underscores(self):
        """Should accept underscores."""
        assert validate_tenant_id("my_tenant") == "my_tenant"


class TestValidateObjectId:
    """Tests for object ID validation."""

    def test_valid_id(self):
        """Should accept valid object ID."""
        assert validate_object_id("segment-123") == "segment-123"

    def test_rejects_none(self):
        """Should reject None."""
        with pytest.raises(QueryError):
            validate_object_id(None)

    def test_rejects_empty(self):
        """Should reject empty string."""
        with pytest.raises(QueryError):
            validate_object_id("")

    def test_accepts_uuid_like(self):
        """Should accept UUID-like strings."""
        assert validate_object_id("a1b2c3d4-e5f6-7890")


class TestValidateManifoldType:
    """Tests for manifold type validation."""

    def test_valid_types(self):
        """Should accept all valid manifold types."""
        for manifold in ["topic", "claim", "procedure", "relation", "time", "evidence"]:
            assert validate_manifold_type(manifold) == manifold

    def test_none_returns_topic(self):
        """Should return 'topic' for None."""
        assert validate_manifold_type(None) == "topic"

    def test_invalid_returns_topic(self):
        """Should return 'topic' for invalid type."""
        assert validate_manifold_type("invalid") == "topic"

    def test_lowercases(self):
        """Should lowercase input."""
        assert validate_manifold_type("TOPIC") == "topic"


class TestValidateQueryMode:
    """Tests for query mode validation."""

    def test_valid_mode_string(self):
        """Should accept valid mode string."""
        result = validate_query_mode("timeline")
        assert result == QueryModeV2.TIMELINE

    def test_valid_mode_enum(self):
        """Should pass through QueryModeV2."""
        result = validate_query_mode(QueryModeV2.PROCEDURE)
        assert result == QueryModeV2.PROCEDURE

    def test_none_returns_none(self):
        """Should return None for None."""
        assert validate_query_mode(None) is None

    def test_invalid_returns_none(self):
        """Should return None for invalid mode."""
        assert validate_query_mode("invalid_mode") is None


class TestValidateManifoldWeights:
    """Tests for manifold weights validation."""

    def test_valid_dict(self):
        """Should accept valid weight dict."""
        result = validate_manifold_weights({
            "topic": 0.5,
            "claim": 0.3,
            "procedure": 0.2,
        })
        assert isinstance(result, ManifoldWeights)
        assert result.topic == 0.5

    def test_clamps_values(self):
        """Should clamp values to [0, 1]."""
        result = validate_manifold_weights({
            "topic": 1.5,  # Over
            "claim": -0.5,  # Under
        })
        assert result.topic == 1.0
        assert result.claim == 0.0

    def test_none_returns_none(self):
        """Should return None for None."""
        assert validate_manifold_weights(None) is None

    def test_passes_through_object(self):
        """Should pass through ManifoldWeights object."""
        weights = ManifoldWeights(topic=0.8)
        result = validate_manifold_weights(weights)
        assert result is weights


class TestValidateTextForEmbedding:
    """Tests for embedding text validation."""

    def test_valid_text(self):
        """Should accept valid text."""
        result = validate_text_for_embedding("Test text")
        assert result == "Test text"

    def test_rejects_none(self):
        """Should reject None."""
        with pytest.raises(QueryError):
            validate_text_for_embedding(None)

    def test_rejects_empty(self):
        """Should reject empty string."""
        with pytest.raises(QueryError):
            validate_text_for_embedding("")

    def test_truncates_long_text(self):
        """Should truncate very long text."""
        long_text = "x" * 100000
        result = validate_text_for_embedding(long_text)
        assert len(result) == 50000


class TestValidateBatchSize:
    """Tests for batch size validation."""

    def test_valid_value(self):
        """Should accept valid value."""
        assert validate_batch_size(100) == 100

    def test_clamps_high(self):
        """Should clamp high values."""
        assert validate_batch_size(10000) == 500

    def test_clamps_low(self):
        """Should clamp low values."""
        assert validate_batch_size(0) == 1


class TestValidateThreshold:
    """Tests for threshold validation."""

    def test_valid_value(self):
        """Should accept valid value."""
        assert validate_threshold(0.5) == 0.5

    def test_clamps_to_range(self):
        """Should clamp to [0, 1]."""
        assert validate_threshold(1.5) == 1.0
        assert validate_threshold(-0.5) == 0.0

    def test_none_returns_default(self):
        """Should return default for None."""
        assert validate_threshold(None) == 0.3
        assert validate_threshold(None, default=0.5) == 0.5


class TestValidateConfig:
    """Tests for configuration validation."""

    def test_valid_config(self):
        """Should validate good config."""
        config = {
            "alpha_topic_default": 0.25,
            "alpha_claim_default": 0.20,
            "alpha_procedure_default": 0.15,
            "alpha_relation_default": 0.15,
            "alpha_time_default": 0.10,
            "alpha_evidence_default": 0.15,
            "promotion_threshold": 0.65,
            "demotion_threshold": 0.35,
        }
        result = validate_config(config)
        assert result.valid

    def test_invalid_weight_sum(self):
        """Should detect invalid weight sum."""
        config = {
            "alpha_topic_default": 0.5,
            "alpha_claim_default": 0.5,
            "alpha_procedure_default": 0.5,  # Sum > 1
        }
        result = validate_config(config)
        assert not result.valid
        assert any("sum" in e.lower() for e in result.errors)

    def test_invalid_thresholds(self):
        """Should detect invalid thresholds."""
        config = {
            "promotion_threshold": 0.3,
            "demotion_threshold": 0.5,  # Demotion > promotion
        }
        result = validate_config(config)
        assert not result.valid


class TestValidationResult:
    """Tests for ValidationResult."""

    def test_success(self):
        """Should create success result."""
        result = ValidationResult.success()
        assert result.valid
        assert result.errors == []

    def test_failure(self):
        """Should create failure result."""
        result = ValidationResult.failure("Error 1", "Error 2")
        assert not result.valid
        assert len(result.errors) == 2

    def test_add_warning(self):
        """Should add warnings."""
        result = ValidationResult.success()
        result.add_warning("Warning 1")
        assert result.valid  # Still valid
        assert len(result.warnings) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
