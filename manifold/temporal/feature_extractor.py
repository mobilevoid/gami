"""Temporal feature extraction for the time manifold.

Extracts 12-dimensional temporal features from text and metadata:
1. absolute_timestamp — Unix timestamp normalized to [0,1] over corpus range
2. relative_recency — Exponential decay from now
3. hour_of_day — Cyclical encoding (sin component)
4. hour_of_day_cos — Cyclical encoding (cos component)
5. day_of_week — Cyclical encoding (sin component)
6. day_of_week_cos — Cyclical encoding (cos component)
7. is_business_hours — Binary: 9am-5pm weekdays
8. temporal_specificity — How specific the time reference is
9. duration_seconds — Event duration if mentioned
10. sequence_position — Position in a sequence of events
11. has_explicit_date — Binary: contains explicit date
12. temporal_distance — Distance from query time context
"""
import re
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, List, Tuple
from enum import Enum


class TemporalGranularity(Enum):
    """Granularity of temporal reference."""
    EXACT = "exact"      # "at 3:42:15 PM"
    MINUTE = "minute"    # "at 3:42 PM"
    HOUR = "hour"        # "around 3 PM"
    DAY = "day"          # "on Tuesday"
    WEEK = "week"        # "last week"
    MONTH = "month"      # "in March"
    YEAR = "year"        # "in 2024"
    RELATIVE = "relative"  # "yesterday", "recently"
    VAGUE = "vague"      # "sometime", "eventually"


@dataclass
class TemporalFeatures:
    """12-dimensional temporal feature vector."""
    absolute_timestamp: float = 0.0
    relative_recency: float = 0.0
    hour_sin: float = 0.0
    hour_cos: float = 0.0
    dow_sin: float = 0.0
    dow_cos: float = 0.0
    is_business_hours: float = 0.0
    temporal_specificity: float = 0.0
    duration_seconds: float = 0.0
    sequence_position: float = 0.0
    has_explicit_date: float = 0.0
    temporal_distance: float = 0.0

    def to_list(self) -> List[float]:
        """Convert to 12-element list."""
        return [
            self.absolute_timestamp,
            self.relative_recency,
            self.hour_sin,
            self.hour_cos,
            self.dow_sin,
            self.dow_cos,
            self.is_business_hours,
            self.temporal_specificity,
            self.duration_seconds,
            self.sequence_position,
            self.has_explicit_date,
            self.temporal_distance,
        ]

    @classmethod
    def from_list(cls, values: List[float]) -> "TemporalFeatures":
        """Create from 12-element list."""
        if len(values) != 12:
            raise ValueError(f"Expected 12 values, got {len(values)}")
        return cls(
            absolute_timestamp=values[0],
            relative_recency=values[1],
            hour_sin=values[2],
            hour_cos=values[3],
            dow_sin=values[4],
            dow_cos=values[5],
            is_business_hours=values[6],
            temporal_specificity=values[7],
            duration_seconds=values[8],
            sequence_position=values[9],
            has_explicit_date=values[10],
            temporal_distance=values[11],
        )


class TemporalExtractor:
    """Extract temporal features from text and metadata."""

    # Regex patterns for temporal expressions
    PATTERNS = {
        "iso_datetime": re.compile(
            r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2})?'
        ),
        "iso_date": re.compile(
            r'\b\d{4}-\d{2}-\d{2}\b'
        ),
        "date_mdy": re.compile(
            r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}',
            re.I
        ),
        "date_dmy": re.compile(
            r'\d{1,2}(?:st|nd|rd|th)? (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}',
            re.I
        ),
        "time_12h": re.compile(
            r'\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)'
        ),
        "time_24h": re.compile(
            r'(?<!\d)\d{2}:\d{2}(?::\d{2})?(?!\d)'
        ),
        "relative_day": re.compile(
            r'\b(?:today|yesterday|tomorrow|tonight)\b', re.I
        ),
        "relative_week": re.compile(
            r'\b(?:last|this|next) (?:week|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
            re.I
        ),
        "relative_month": re.compile(
            r'\b(?:last|this|next) (?:month|year)\b', re.I
        ),
        "duration": re.compile(
            r'(\d+)\s*(?:seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?)',
            re.I
        ),
        "ago": re.compile(
            r'(\d+)\s*(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?)\s+ago',
            re.I
        ),
    }

    def __init__(
        self,
        corpus_start: Optional[datetime] = None,
        corpus_end: Optional[datetime] = None,
    ):
        """Initialize extractor.

        Args:
            corpus_start: Earliest timestamp in corpus (for normalization).
            corpus_end: Latest timestamp in corpus (for normalization).
        """
        self.corpus_start = corpus_start or datetime(2020, 1, 1)
        self.corpus_end = corpus_end or datetime.now()
        self.corpus_range = (self.corpus_end - self.corpus_start).total_seconds()

    def extract(
        self,
        text: str,
        timestamp: Optional[datetime] = None,
        sequence_position: Optional[int] = None,
        sequence_length: Optional[int] = None,
        query_time: Optional[datetime] = None,
    ) -> TemporalFeatures:
        """Extract temporal features from text.

        Args:
            text: The text to analyze.
            timestamp: Known timestamp of the content.
            sequence_position: Position in event sequence (0-indexed).
            sequence_length: Total events in sequence.
            query_time: Reference time for distance calculation.

        Returns:
            12-dimensional temporal features.
        """
        features = TemporalFeatures()

        # Use provided timestamp or try to extract from text
        event_time = timestamp
        if event_time is None:
            event_time = self._extract_datetime(text)

        now = datetime.now()
        query_time = query_time or now

        if event_time:
            # 1. Absolute timestamp (normalized to corpus range)
            offset = (event_time - self.corpus_start).total_seconds()
            features.absolute_timestamp = max(0, min(1, offset / self.corpus_range))

            # 2. Relative recency (half-life of 30 days)
            days_ago = (now - event_time).total_seconds() / 86400
            features.relative_recency = math.exp(-days_ago / 43.3)

            # 3-4. Hour of day (cyclical encoding)
            hour = event_time.hour + event_time.minute / 60
            features.hour_sin = math.sin(2 * math.pi * hour / 24)
            features.hour_cos = math.cos(2 * math.pi * hour / 24)

            # 5-6. Day of week (cyclical encoding)
            dow = event_time.weekday()
            features.dow_sin = math.sin(2 * math.pi * dow / 7)
            features.dow_cos = math.cos(2 * math.pi * dow / 7)

            # 7. Business hours (9-17 weekdays)
            is_weekday = dow < 5
            is_work_hours = 9 <= event_time.hour < 17
            features.is_business_hours = 1.0 if (is_weekday and is_work_hours) else 0.0

            # 12. Temporal distance from query time
            distance_days = abs((event_time - query_time).total_seconds()) / 86400
            features.temporal_distance = math.exp(-distance_days / 30)

        # 8. Temporal specificity
        features.temporal_specificity = self._compute_specificity(text)

        # 9. Duration
        features.duration_seconds = self._extract_duration(text)

        # 10. Sequence position
        if sequence_position is not None and sequence_length and sequence_length > 1:
            features.sequence_position = sequence_position / (sequence_length - 1)

        # 11. Has explicit date
        features.has_explicit_date = 1.0 if self._has_explicit_date(text) else 0.0

        return features

    def _extract_datetime(self, text: str) -> Optional[datetime]:
        """Try to extract a datetime from text."""
        # ISO datetime
        match = self.PATTERNS["iso_datetime"].search(text)
        if match:
            try:
                dt_str = match.group().replace(" ", "T")
                if len(dt_str) == 16:  # No seconds
                    dt_str += ":00"
                return datetime.fromisoformat(dt_str)
            except ValueError:
                pass

        # Could add more parsers for other formats
        return None

    def _compute_specificity(self, text: str) -> float:
        """Compute temporal specificity score."""
        granularity = self._detect_granularity(text)

        specificity_scores = {
            TemporalGranularity.EXACT: 1.0,
            TemporalGranularity.MINUTE: 0.9,
            TemporalGranularity.HOUR: 0.7,
            TemporalGranularity.DAY: 0.5,
            TemporalGranularity.WEEK: 0.3,
            TemporalGranularity.MONTH: 0.2,
            TemporalGranularity.YEAR: 0.1,
            TemporalGranularity.RELATIVE: 0.4,
            TemporalGranularity.VAGUE: 0.0,
        }

        return specificity_scores.get(granularity, 0.0)

    def _detect_granularity(self, text: str) -> TemporalGranularity:
        """Detect the granularity of temporal reference."""
        if self.PATTERNS["iso_datetime"].search(text):
            return TemporalGranularity.EXACT
        if self.PATTERNS["time_12h"].search(text) or self.PATTERNS["time_24h"].search(text):
            return TemporalGranularity.MINUTE
        if self.PATTERNS["iso_date"].search(text) or self.PATTERNS["date_mdy"].search(text) or self.PATTERNS["date_dmy"].search(text):
            return TemporalGranularity.DAY
        if self.PATTERNS["relative_day"].search(text):
            return TemporalGranularity.RELATIVE
        if self.PATTERNS["relative_week"].search(text):
            return TemporalGranularity.WEEK
        if self.PATTERNS["relative_month"].search(text):
            return TemporalGranularity.MONTH

        # Check for vague terms
        vague_terms = ["sometime", "eventually", "soon", "later", "recently"]
        if any(term in text.lower() for term in vague_terms):
            return TemporalGranularity.VAGUE

        return TemporalGranularity.VAGUE

    def _extract_duration(self, text: str) -> float:
        """Extract duration in normalized form (0-1 scale, log-scaled)."""
        match = self.PATTERNS["duration"].search(text)
        if not match:
            return 0.0

        value = int(match.group(1))
        unit = match.group().lower()

        # Convert to seconds
        if "second" in unit or "sec" in unit:
            seconds = value
        elif "minute" in unit or "min" in unit:
            seconds = value * 60
        elif "hour" in unit or "hr" in unit:
            seconds = value * 3600
        elif "day" in unit:
            seconds = value * 86400
        elif "week" in unit:
            seconds = value * 604800
        else:
            seconds = value

        # Log-scale normalize (1 second to 1 year range)
        max_seconds = 31536000  # 1 year
        normalized = math.log1p(seconds) / math.log1p(max_seconds)
        return min(1.0, normalized)

    def _has_explicit_date(self, text: str) -> bool:
        """Check if text contains an explicit date."""
        return bool(
            self.PATTERNS["iso_datetime"].search(text)
            or self.PATTERNS["iso_date"].search(text)
            or self.PATTERNS["date_mdy"].search(text)
            or self.PATTERNS["date_dmy"].search(text)
        )


def compute_temporal_similarity(
    features1: TemporalFeatures,
    features2: TemporalFeatures,
) -> float:
    """Compute similarity between two temporal feature vectors.

    Uses weighted combination focusing on relevant features.
    """
    v1 = features1.to_list()
    v2 = features2.to_list()

    # Weights for each feature (emphasize recency and timestamps)
    weights = [
        0.15,  # absolute_timestamp
        0.25,  # relative_recency
        0.05,  # hour_sin
        0.05,  # hour_cos
        0.05,  # dow_sin
        0.05,  # dow_cos
        0.08,  # is_business_hours
        0.10,  # temporal_specificity
        0.05,  # duration
        0.07,  # sequence_position
        0.05,  # has_explicit_date
        0.05,  # temporal_distance
    ]

    # Weighted squared difference for sharper discrimination
    diff = sum(w * (a - b) ** 2 for w, a, b in zip(weights, v1, v2))
    # Take sqrt to bring back to roughly [0,1] scale
    similarity = 1.0 - min(1.0, diff ** 0.5)

    return similarity
