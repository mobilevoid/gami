"""Unit tests for evidence scoring.

Tests the 5-factor evidence score computation.
"""
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from manifold.scoring.evidence import (
    compute_evidence_score,
    EvidenceFactors,
    EvidenceVector,
    compute_source_authority,
    compute_specificity,
)


class TestEvidenceScore:
    """Tests for evidence score computation."""

    def test_all_factors_contribute(self):
        """All 5 factors should contribute to final score."""
        factors = EvidenceFactors(
            source_authority=0.8,
            corroboration_count=3,
            days_since_observed=10.0,
            specificity=0.7,
            contradiction_count=0,
            total_mentions=5,
        )

        score = compute_evidence_score(factors)
        assert 0.0 <= score <= 1.0

    def test_zero_factors(self):
        """All-zero factors should yield low score."""
        factors = EvidenceFactors(
            source_authority=0.0,
            corroboration_count=0,
            days_since_observed=365.0,  # Very old
            specificity=0.0,
            contradiction_count=5,
            total_mentions=5,
        )

        score = compute_evidence_score(factors)
        assert score < 0.3

    def test_high_authority_boosts_score(self):
        """High source authority should increase score."""
        low_auth = EvidenceFactors(source_authority=0.2)
        high_auth = EvidenceFactors(source_authority=0.9)

        low_score = compute_evidence_score(low_auth)
        high_score = compute_evidence_score(high_auth)

        assert high_score > low_score

    def test_corroboration_boosts_score(self):
        """More corroborations should increase score."""
        no_corr = EvidenceFactors(corroboration_count=0)
        high_corr = EvidenceFactors(corroboration_count=5)

        no_score = compute_evidence_score(no_corr)
        high_score = compute_evidence_score(high_corr)

        assert high_score > no_score

    def test_contradictions_reduce_score(self):
        """Contradictions should reduce score."""
        no_contra = EvidenceFactors(contradiction_count=0, total_mentions=5)
        high_contra = EvidenceFactors(contradiction_count=4, total_mentions=5)

        no_score = compute_evidence_score(no_contra)
        high_score = compute_evidence_score(high_contra)

        assert no_score > high_score

    def test_recency_decay(self):
        """Old evidence should score lower than recent."""
        recent = EvidenceFactors(days_since_observed=1.0)
        old = EvidenceFactors(days_since_observed=180.0)

        recent_score = compute_evidence_score(recent)
        old_score = compute_evidence_score(old)

        assert recent_score > old_score


class TestSourceAuthority:
    """Tests for source authority computation."""

    def test_documentation_highest(self):
        """Documentation should have high authority."""
        auth = compute_source_authority("documentation")
        assert auth >= 0.8

    def test_log_lowest(self):
        """Logs should have lower authority."""
        auth = compute_source_authority("log")
        assert auth < 0.5

    def test_primary_source_boost(self):
        """Primary sources should get a boost."""
        secondary = compute_source_authority("conversation", is_primary=False)
        primary = compute_source_authority("conversation", is_primary=True)

        assert primary > secondary

    def test_citations_boost(self):
        """More citations should boost authority."""
        no_cite = compute_source_authority("documentation", citation_count=0)
        high_cite = compute_source_authority("documentation", citation_count=20)

        assert high_cite >= no_cite


class TestSpecificity:
    """Tests for specificity scoring."""

    def test_specific_text_high_score(self):
        """Specific text should score high."""
        text = "PostgreSQL runs on port 5432 at db.example.com"
        score = compute_specificity(text)
        assert score >= 0.5

    def test_vague_text_low_score(self):
        """Vague text should score lower."""
        text = "Some services might sometimes run on various ports"
        score = compute_specificity(text)
        assert score < 0.5

    def test_numbers_boost_score(self):
        """Numbers should boost specificity."""
        without = "The service runs on the default port"
        with_num = "The service runs on port 8080"

        assert compute_specificity(with_num) > compute_specificity(without)

    def test_dates_boost_score(self):
        """Dates should boost specificity."""
        without = "The backup runs regularly"
        with_date = "The backup runs at 2:00 AM on 2026-04-10"

        assert compute_specificity(with_date) > compute_specificity(without)


class TestEvidenceVector:
    """Tests for evidence vector conversion."""

    def test_vector_length(self):
        """Vector should have 5 elements."""
        factors = EvidenceFactors(
            source_authority=0.8,
            corroboration_count=3,
            days_since_observed=10.0,
            specificity=0.7,
        )

        vector = EvidenceVector.from_factors(factors)
        assert len(vector.to_list()) == 5

    def test_vector_bounds(self):
        """All vector elements should be in [0, 1]."""
        factors = EvidenceFactors(
            source_authority=1.5,  # Over 1
            corroboration_count=100,
            days_since_observed=0.0,
            specificity=0.7,
        )

        vector = EvidenceVector.from_factors(factors)
        for val in vector.to_list():
            assert 0.0 <= val <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
