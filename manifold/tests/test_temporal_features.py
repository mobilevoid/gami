"""Unit tests for temporal feature extraction.

Tests the 12-dimensional temporal feature vector.
"""
import pytest
import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from manifold.temporal.feature_extractor import (
    TemporalExtractor,
    TemporalFeatures,
    TemporalGranularity,
    compute_temporal_similarity,
)


class TestTemporalFeatures:
    """Tests for temporal feature vector."""

    def test_feature_count(self):
        """Should produce 12 features."""
        features = TemporalFeatures()
        assert len(features.to_list()) == 12

    def test_from_list_roundtrip(self):
        """Should roundtrip through list conversion."""
        original = TemporalFeatures(
            absolute_timestamp=0.5,
            relative_recency=0.8,
            hour_sin=0.3,
            hour_cos=0.7,
            dow_sin=0.1,
            dow_cos=0.9,
            is_business_hours=1.0,
            temporal_specificity=0.6,
            duration_seconds=0.2,
            sequence_position=0.4,
            has_explicit_date=1.0,
            temporal_distance=0.75,
        )

        as_list = original.to_list()
        restored = TemporalFeatures.from_list(as_list)

        assert restored.absolute_timestamp == original.absolute_timestamp
        assert restored.relative_recency == original.relative_recency
        assert restored.temporal_specificity == original.temporal_specificity

    def test_invalid_list_length(self):
        """Should reject wrong-length lists."""
        with pytest.raises(ValueError):
            TemporalFeatures.from_list([0.1, 0.2, 0.3])  # Too few


class TestTemporalExtractor:
    """Tests for temporal extraction."""

    @pytest.fixture
    def extractor(self):
        return TemporalExtractor(
            corpus_start=datetime(2020, 1, 1),
            corpus_end=datetime(2026, 1, 1),
        )

    def test_with_timestamp(self, extractor):
        """Should use provided timestamp."""
        ts = datetime(2023, 6, 15, 14, 30, 0)
        features = extractor.extract("Some text", timestamp=ts)

        # Absolute timestamp should be normalized
        assert 0.0 <= features.absolute_timestamp <= 1.0
        # Mid-2023 should be roughly in middle of 2020-2026
        assert 0.4 < features.absolute_timestamp < 0.7

    def test_recency_decay(self, extractor):
        """Recent timestamps should have higher recency."""
        now = datetime.now()
        recent = extractor.extract("Text", timestamp=now - timedelta(days=1))
        old = extractor.extract("Text", timestamp=now - timedelta(days=365))

        assert recent.relative_recency > old.relative_recency

    def test_hour_cyclical_encoding(self, extractor):
        """Hour encoding should be cyclical."""
        morning = extractor.extract("Text", timestamp=datetime(2023, 1, 1, 9, 0))
        afternoon = extractor.extract("Text", timestamp=datetime(2023, 1, 1, 15, 0))
        midnight = extractor.extract("Text", timestamp=datetime(2023, 1, 1, 0, 0))

        # Sin/cos should differ for different times
        assert morning.hour_sin != afternoon.hour_sin
        # Midnight should have hour_sin near 0
        assert abs(midnight.hour_sin) < 0.1

    def test_day_of_week_encoding(self, extractor):
        """Day of week encoding should be cyclical."""
        # Monday
        monday = extractor.extract("Text", timestamp=datetime(2023, 1, 2, 12, 0))
        # Friday
        friday = extractor.extract("Text", timestamp=datetime(2023, 1, 6, 12, 0))

        assert monday.dow_sin != friday.dow_sin

    def test_business_hours_detection(self, extractor):
        """Should detect business hours."""
        # Tuesday 10 AM
        business = extractor.extract("Text", timestamp=datetime(2023, 1, 3, 10, 0))
        # Saturday 10 AM
        weekend = extractor.extract("Text", timestamp=datetime(2023, 1, 7, 10, 0))
        # Tuesday 10 PM
        night = extractor.extract("Text", timestamp=datetime(2023, 1, 3, 22, 0))

        assert business.is_business_hours == 1.0
        assert weekend.is_business_hours == 0.0
        assert night.is_business_hours == 0.0

    def test_explicit_date_detection(self, extractor):
        """Should detect explicit dates in text."""
        with_date = extractor.extract("Meeting on 2023-04-15 at noon")
        without_date = extractor.extract("Meeting sometime next week")

        assert with_date.has_explicit_date == 1.0
        assert without_date.has_explicit_date == 0.0

    def test_duration_extraction(self, extractor):
        """Should extract durations."""
        with_duration = extractor.extract("The process takes 30 minutes")
        without_duration = extractor.extract("The process completes quickly")

        assert with_duration.duration_seconds > 0
        assert without_duration.duration_seconds == 0

    def test_sequence_position(self, extractor):
        """Should normalize sequence position."""
        first = extractor.extract("Text", sequence_position=0, sequence_length=10)
        middle = extractor.extract("Text", sequence_position=5, sequence_length=10)
        last = extractor.extract("Text", sequence_position=9, sequence_length=10)

        assert first.sequence_position == 0.0
        assert middle.sequence_position == pytest.approx(0.555, rel=0.1)
        assert last.sequence_position == 1.0


class TestTemporalSpecificity:
    """Tests for temporal specificity scoring."""

    @pytest.fixture
    def extractor(self):
        return TemporalExtractor()

    def test_exact_time_high_specificity(self, extractor):
        """Exact datetime should have high specificity."""
        features = extractor.extract("Event at 2023-04-15T14:30:00")
        assert features.temporal_specificity >= 0.8

    def test_vague_low_specificity(self, extractor):
        """Vague references should have low specificity."""
        features = extractor.extract("Sometime in the future")
        assert features.temporal_specificity < 0.3

    def test_relative_medium_specificity(self, extractor):
        """Relative references should have medium specificity."""
        features = extractor.extract("Yesterday afternoon")
        assert 0.3 <= features.temporal_specificity <= 0.6


class TestTemporalSimilarity:
    """Tests for temporal similarity computation."""

    def test_identical_features(self):
        """Identical features should have similarity 1.0."""
        features = TemporalFeatures(
            absolute_timestamp=0.5,
            relative_recency=0.8,
            temporal_specificity=0.7,
        )
        sim = compute_temporal_similarity(features, features)
        assert sim == pytest.approx(1.0)

    def test_opposite_features(self):
        """Opposite features should have low similarity."""
        f1 = TemporalFeatures(
            absolute_timestamp=0.0,
            relative_recency=1.0,
            is_business_hours=1.0,
        )
        f2 = TemporalFeatures(
            absolute_timestamp=1.0,
            relative_recency=0.0,
            is_business_hours=0.0,
        )
        sim = compute_temporal_similarity(f1, f2)
        assert sim < 0.5

    def test_similar_recency_high_similarity(self):
        """Similar recency should contribute to similarity."""
        f1 = TemporalFeatures(relative_recency=0.8)
        f2 = TemporalFeatures(relative_recency=0.75)

        f3 = TemporalFeatures(relative_recency=0.1)

        sim_close = compute_temporal_similarity(f1, f2)
        sim_far = compute_temporal_similarity(f1, f3)

        assert sim_close > sim_far


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
