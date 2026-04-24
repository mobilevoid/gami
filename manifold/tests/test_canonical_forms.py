"""Unit tests for canonical form generators.

Tests the claim normalizer, procedure normalizer, and temporal extractor.
These tests can run without database connection.
"""
import pytest
from datetime import date

# Import from manifold package
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from manifold.canonical.claim_normalizer import ClaimNormalizer, normalize_claim
from manifold.canonical.procedure_normalizer import ProcedureNormalizer, normalize_procedure
from manifold.canonical.temporal_extractor import TemporalExtractor, extract_temporal_features
from manifold.canonical.forms import (
    CanonicalClaimForm,
    CanonicalProcedureForm,
    create_claim_form,
    create_procedure_form,
)
from manifold.models.schemas import Modality


class TestClaimNormalizer:
    """Tests for claim normalization."""

    def test_simple_is_claim(self):
        """Test normalization of 'X is Y' claims."""
        claim = normalize_claim("PostgreSQL is a relational database")
        assert claim is not None
        assert claim.subject.lower() == "postgresql"
        assert claim.predicate == "is"
        assert "relational database" in claim.object.lower()

    def test_has_claim(self):
        """Test normalization of 'X has Y' claims."""
        claim = normalize_claim("The system has three components")
        assert claim is not None
        assert claim.predicate == "has"

    def test_past_tense_claim(self):
        """Test normalization of past tense claims."""
        claim = normalize_claim("The Vietnamese Communist Party exercised control over the CPK")
        assert claim is not None
        assert claim.predicate == "exercised"

    def test_modality_detection_negated(self):
        """Test detection of negated modality."""
        normalizer = ClaimNormalizer(use_llm=False)
        claim = normalizer.normalize("The system does not support Windows")
        assert claim is not None
        assert claim.modality == Modality.NEGATED

    def test_modality_detection_possible(self):
        """Test detection of possible modality."""
        normalizer = ClaimNormalizer(use_llm=False)
        claim = normalizer.normalize("The feature might be available in version 3")
        assert claim is not None
        assert claim.modality == Modality.POSSIBLE

    def test_qualifier_extraction(self):
        """Test extraction of qualifier words."""
        claim = normalize_claim("The significant majority strongly supported the measure")
        assert claim is not None
        assert "significant" in claim.qualifiers or "strong" in claim.qualifiers

    def test_temporal_extraction(self):
        """Test extraction of temporal scope."""
        claim = normalize_claim("The policy was enacted in 2020")
        assert claim is not None
        assert claim.temporal_scope is not None
        assert "2020" in claim.temporal_scope

    def test_canonical_text_generation(self):
        """Test canonical text format."""
        claim = normalize_claim("Manifold is a memory system")
        assert claim is not None
        canonical = claim.to_canonical_text()
        assert "[" in canonical
        assert "|" in canonical
        assert "modality=" in canonical


class TestProcedureNormalizer:
    """Tests for procedure normalization."""

    def test_numbered_list_extraction(self):
        """Test extraction from numbered list."""
        text = """How to deploy the application:
        1. Run the migrations
        2. Start the API server
        3. Verify the health endpoint
        """
        proc = normalize_procedure(text)
        assert proc is not None
        assert len(proc.steps) == 3
        assert proc.steps[0].order == 1

    def test_bulleted_list_extraction(self):
        """Test extraction from bulleted list."""
        text = """Installation steps:
        - Install PostgreSQL
        - Install Redis
        - Configure environment
        """
        proc = normalize_procedure(text, use_llm=False)
        assert proc is not None
        assert len(proc.steps) >= 2

    def test_sequence_word_extraction(self):
        """Test extraction using sequence words."""
        text = "First install the dependencies, then configure the settings, finally run the tests."
        proc = normalize_procedure(text, use_llm=False)
        # May or may not extract - depends on pattern matching
        # This is a weaker pattern

    def test_title_extraction(self):
        """Test title extraction."""
        text = "How to configure SSL:\n1. Generate certificate\n2. Install certificate"
        proc = normalize_procedure(text, use_llm=False)
        assert proc is not None
        assert "ssl" in proc.title.lower() or "configure" in proc.title.lower()

    def test_canonical_text_generation(self):
        """Test canonical text format."""
        proc = CanonicalProcedureForm.from_steps(
            title="Deploy Application",
            steps=["Run migrations", "Start API"],
            prerequisites=["PostgreSQL"],
            outcome="Application running",
        )
        canonical = proc.to_text()
        assert "title=" in canonical
        assert "steps=" in canonical


class TestTemporalExtractor:
    """Tests for temporal feature extraction."""

    def test_full_date_extraction(self):
        """Test full date extraction."""
        result = extract_temporal_features("The event occurred on March 15, 2026")
        assert result is not None
        assert result["start_date"].year == 2026
        assert result["start_date"].month == 3
        assert result["start_date"].day == 15
        assert result["precision"] == "day"

    def test_iso_date_extraction(self):
        """Test ISO date extraction."""
        result = extract_temporal_features("Deployed on 2026-04-10")
        assert result is not None
        assert result["start_date"] == date(2026, 4, 10)

    def test_year_extraction(self):
        """Test single year extraction."""
        result = extract_temporal_features("This happened in 2020")
        assert result is not None
        assert result["start_date"].year == 2020
        assert result["precision"] == "year"

    def test_date_range_extraction(self):
        """Test date range extraction."""
        result = extract_temporal_features("The project ran from 2020 to 2023")
        assert result is not None
        assert result["is_range"] == True
        assert result["start_date"].year == 2020
        assert result["end_date"].year == 2023

    def test_feature_vector_length(self):
        """Test that feature vector has correct length."""
        result = extract_temporal_features("In January 2025")
        assert result is not None
        assert len(result["features"]) == 12

    def test_feature_normalization(self):
        """Test that features are normalized to reasonable ranges."""
        result = extract_temporal_features("March 15, 2020")
        assert result is not None
        features = result["features"]
        # Check year normalization is in 0-1 range
        assert 0.0 <= features[0] <= 1.0
        # Check cyclical encodings are in -1 to 1 range
        assert -1.0 <= features[1] <= 1.0  # month_sin
        assert -1.0 <= features[2] <= 1.0  # month_cos

    def test_no_temporal_info(self):
        """Test handling of text without temporal information."""
        result = extract_temporal_features("This is just some text")
        assert result is None


class TestCanonicalForms:
    """Tests for canonical form classes."""

    def test_claim_form_creation(self):
        """Test creating a claim form."""
        form = create_claim_form(
            subject="PostgreSQL",
            predicate="is",
            object="a database",
            modality="factual",
        )
        assert form.subject == "PostgreSQL"
        assert "[PostgreSQL]" in str(form)

    def test_procedure_form_creation(self):
        """Test creating a procedure form."""
        form = create_procedure_form(
            title="Install Application",
            steps=["Download", "Configure", "Run"],
        )
        assert len(form.steps) == 3
        assert "title=" in str(form)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
