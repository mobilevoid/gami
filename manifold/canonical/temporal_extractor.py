"""Temporal extractor — extracts structured time features from text.

Instead of embedding dates as text vectors (which poorly captures temporal
relationships), we extract structured features:

- year_normalized: 0-1 scaled year (relative to corpus range)
- month_sin, month_cos: cyclical month encoding
- day_sin, day_cos: cyclical day encoding
- is_range: whether this is a date range
- range_days: duration in days if range
- sequence_position: position within source document (0-1)
- has_explicit_date: whether date was explicitly stated
- is_relative: "last week" vs "March 15"
- temporal_precision: year/month/day/hour
- is_ongoing: "since 2020" type expressions

These 12 features form the time manifold representation.
"""
import logging
import math
import re
from datetime import datetime, date, timedelta
from typing import Optional, List, Tuple, Dict, Any

logger = logging.getLogger("manifold.canonical.temporal")


# Year range for normalization (can be adjusted based on corpus)
MIN_YEAR = 1900
MAX_YEAR = 2030


# Temporal patterns
YEAR_PATTERN = re.compile(r"\b(1[89]\d{2}|20[0-3]\d)\b")
MONTH_YEAR_PATTERN = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\s+(\d{4})\b",
    re.IGNORECASE,
)
FULL_DATE_PATTERN = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\s+(\d{1,2}),?\s*(\d{4})\b",
    re.IGNORECASE,
)
ISO_DATE_PATTERN = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
DATE_RANGE_PATTERN = re.compile(
    r"\b(\d{4})\s*[-–—to]+\s*(\d{4})\b",
    re.IGNORECASE,
)
RELATIVE_PATTERNS = re.compile(
    r"\b(yesterday|today|tomorrow|last (?:week|month|year)|"
    r"next (?:week|month|year)|(?:\d+) (?:days?|weeks?|months?|years?) ago|"
    r"in (?:\d+) (?:days?|weeks?|months?|years?))\b",
    re.IGNORECASE,
)
ONGOING_PATTERNS = re.compile(
    r"\b(since \d{4}|from \d{4}|beginning in \d{4}|starting in \d{4})\b",
    re.IGNORECASE,
)
DECADE_PATTERN = re.compile(
    r"\b(?:the |in the )?(1[89]\d0|20[0-2]0)s\b",
    re.IGNORECASE,
)
CENTURY_PATTERN = re.compile(
    r"\b(?:the )?(1[89]th|20th|21st) century\b",
    re.IGNORECASE,
)

MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


class TemporalExtractor:
    """Extracts temporal features from text for the time manifold."""

    def __init__(
        self,
        min_year: int = MIN_YEAR,
        max_year: int = MAX_YEAR,
        reference_date: Optional[date] = None,
    ):
        """Initialize the temporal extractor.

        Args:
            min_year: Minimum year for normalization.
            max_year: Maximum year for normalization.
            reference_date: Reference date for relative expressions (default: today).
        """
        self.min_year = min_year
        self.max_year = max_year
        self.reference_date = reference_date or date.today()

    def extract(
        self,
        text: str,
        sequence_position: float = 0.5,
    ) -> Optional[Dict[str, Any]]:
        """Extract temporal features from text.

        Args:
            text: Text to extract from.
            sequence_position: Position of this text within its source (0-1).

        Returns:
            Dict with features array and metadata, or None if no temporal info found.
        """
        if not text or len(text.strip()) < 5:
            return None

        # Try different extraction methods in order of precision
        result = None

        # 1. Full date (most precise)
        result = self._extract_full_date(text)
        if result:
            result["sequence_position"] = sequence_position
            return self._compute_features(result)

        # 2. Month + year
        result = self._extract_month_year(text)
        if result:
            result["sequence_position"] = sequence_position
            return self._compute_features(result)

        # 3. Date range
        result = self._extract_date_range(text)
        if result:
            result["sequence_position"] = sequence_position
            return self._compute_features(result)

        # 4. Single year
        result = self._extract_year(text)
        if result:
            result["sequence_position"] = sequence_position
            return self._compute_features(result)

        # 5. Decade
        result = self._extract_decade(text)
        if result:
            result["sequence_position"] = sequence_position
            return self._compute_features(result)

        # 6. Relative expression
        result = self._extract_relative(text)
        if result:
            result["sequence_position"] = sequence_position
            return self._compute_features(result)

        # 7. Ongoing expression
        result = self._extract_ongoing(text)
        if result:
            result["sequence_position"] = sequence_position
            return self._compute_features(result)

        return None

    def _extract_full_date(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract full date (month day, year)."""
        # Try ISO format first
        match = ISO_DATE_PATTERN.search(text)
        if match:
            year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
            try:
                d = date(year, month, day)
                return {
                    "start_date": d,
                    "end_date": d,
                    "is_range": False,
                    "precision": "day",
                    "is_relative": False,
                    "is_ongoing": False,
                    "has_explicit_date": True,
                    "raw_text": match.group(0),
                }
            except ValueError:
                pass

        # Try natural language format
        match = FULL_DATE_PATTERN.search(text)
        if match:
            month_name, day, year = match.group(1), int(match.group(2)), int(match.group(3))
            month = MONTH_MAP.get(month_name.lower(), 1)
            try:
                d = date(year, month, day)
                return {
                    "start_date": d,
                    "end_date": d,
                    "is_range": False,
                    "precision": "day",
                    "is_relative": False,
                    "is_ongoing": False,
                    "has_explicit_date": True,
                    "raw_text": match.group(0),
                }
            except ValueError:
                pass

        return None

    def _extract_month_year(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract month + year."""
        match = MONTH_YEAR_PATTERN.search(text)
        if match:
            month_name, year = match.group(1), int(match.group(2))
            month = MONTH_MAP.get(month_name.lower(), 1)
            try:
                start = date(year, month, 1)
                # End of month
                if month == 12:
                    end = date(year + 1, 1, 1) - timedelta(days=1)
                else:
                    end = date(year, month + 1, 1) - timedelta(days=1)
                return {
                    "start_date": start,
                    "end_date": end,
                    "is_range": True,
                    "precision": "month",
                    "is_relative": False,
                    "is_ongoing": False,
                    "has_explicit_date": True,
                    "raw_text": match.group(0),
                }
            except ValueError:
                pass
        return None

    def _extract_date_range(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract date range (year-year)."""
        match = DATE_RANGE_PATTERN.search(text)
        if match:
            start_year, end_year = int(match.group(1)), int(match.group(2))
            if start_year <= end_year:
                return {
                    "start_date": date(start_year, 1, 1),
                    "end_date": date(end_year, 12, 31),
                    "is_range": True,
                    "precision": "year",
                    "is_relative": False,
                    "is_ongoing": False,
                    "has_explicit_date": True,
                    "raw_text": match.group(0),
                }
        return None

    def _extract_year(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract single year."""
        match = YEAR_PATTERN.search(text)
        if match:
            year = int(match.group(1))
            return {
                "start_date": date(year, 1, 1),
                "end_date": date(year, 12, 31),
                "is_range": True,
                "precision": "year",
                "is_relative": False,
                "is_ongoing": False,
                "has_explicit_date": True,
                "raw_text": match.group(0),
            }
        return None

    def _extract_decade(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract decade reference."""
        match = DECADE_PATTERN.search(text)
        if match:
            decade_start = int(match.group(1))
            return {
                "start_date": date(decade_start, 1, 1),
                "end_date": date(decade_start + 9, 12, 31),
                "is_range": True,
                "precision": "decade",
                "is_relative": False,
                "is_ongoing": False,
                "has_explicit_date": True,
                "raw_text": match.group(0),
            }
        return None

    def _extract_relative(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract relative time expression."""
        match = RELATIVE_PATTERNS.search(text)
        if match:
            expr = match.group(0).lower()
            ref = self.reference_date

            if expr == "yesterday":
                d = ref - timedelta(days=1)
            elif expr == "today":
                d = ref
            elif expr == "tomorrow":
                d = ref + timedelta(days=1)
            elif "last week" in expr:
                d = ref - timedelta(weeks=1)
            elif "last month" in expr:
                d = ref.replace(month=ref.month - 1) if ref.month > 1 else ref.replace(year=ref.year - 1, month=12)
            elif "last year" in expr:
                d = ref.replace(year=ref.year - 1)
            else:
                # Parse "N days/weeks/months/years ago"
                d = ref  # Fallback

            return {
                "start_date": d,
                "end_date": d,
                "is_range": False,
                "precision": "day",
                "is_relative": True,
                "is_ongoing": False,
                "has_explicit_date": False,
                "raw_text": match.group(0),
            }
        return None

    def _extract_ongoing(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract ongoing time expression."""
        match = ONGOING_PATTERNS.search(text)
        if match:
            # Find the year
            year_match = YEAR_PATTERN.search(match.group(0))
            if year_match:
                start_year = int(year_match.group(1))
                return {
                    "start_date": date(start_year, 1, 1),
                    "end_date": self.reference_date,  # Ongoing until now
                    "is_range": True,
                    "precision": "year",
                    "is_relative": False,
                    "is_ongoing": True,
                    "has_explicit_date": True,
                    "raw_text": match.group(0),
                }
        return None

    def _compute_features(self, extracted: Dict[str, Any]) -> Dict[str, Any]:
        """Compute the 12-dimensional feature vector."""
        start_date = extracted["start_date"]
        end_date = extracted["end_date"]

        # Use midpoint for single value features
        if isinstance(start_date, date):
            mid_date = start_date + (end_date - start_date) / 2
        else:
            mid_date = self.reference_date

        # 1. Year normalized (0-1)
        year_norm = (mid_date.year - self.min_year) / (self.max_year - self.min_year)
        year_norm = max(0.0, min(1.0, year_norm))

        # 2-3. Month cyclical encoding
        month_rad = 2 * math.pi * (mid_date.month - 1) / 12
        month_sin = math.sin(month_rad)
        month_cos = math.cos(month_rad)

        # 4-5. Day cyclical encoding
        day_rad = 2 * math.pi * (mid_date.day - 1) / 31
        day_sin = math.sin(day_rad)
        day_cos = math.cos(day_rad)

        # 6. Is range
        is_range = 1.0 if extracted["is_range"] else 0.0

        # 7. Range days (normalized, log scale)
        if extracted["is_range"]:
            range_days = (end_date - start_date).days
            range_days_norm = math.log(range_days + 1) / math.log(3650 + 1)  # Max ~10 years
        else:
            range_days_norm = 0.0

        # 8. Sequence position
        seq_pos = extracted.get("sequence_position", 0.5)

        # 9. Has explicit date
        has_explicit = 1.0 if extracted["has_explicit_date"] else 0.0

        # 10. Is relative
        is_relative = 1.0 if extracted["is_relative"] else 0.0

        # 11. Temporal precision (encoded as 0-1)
        precision_map = {
            "hour": 1.0,
            "day": 0.8,
            "month": 0.6,
            "year": 0.4,
            "decade": 0.2,
            "century": 0.1,
        }
        precision_val = precision_map.get(extracted["precision"], 0.5)

        # 12. Is ongoing
        is_ongoing = 1.0 if extracted["is_ongoing"] else 0.0

        features = [
            year_norm,
            month_sin,
            month_cos,
            day_sin,
            day_cos,
            is_range,
            range_days_norm,
            seq_pos,
            has_explicit,
            is_relative,
            precision_val,
            is_ongoing,
        ]

        return {
            "features": features,
            "start_date": start_date,
            "end_date": end_date,
            "is_range": extracted["is_range"],
            "precision": extracted["precision"],
            "is_relative": extracted["is_relative"],
            "raw_text": extracted.get("raw_text"),
        }


def extract_temporal_features(
    text: str,
    sequence_position: float = 0.5,
) -> Optional[Dict[str, Any]]:
    """Convenience function for temporal extraction.

    Args:
        text: Text to extract from.
        sequence_position: Position within source (0-1).

    Returns:
        Dict with 'features' (12-element list) and metadata, or None.
    """
    extractor = TemporalExtractor()
    return extractor.extract(text, sequence_position)


def batch_extract_temporal(
    texts: List[str],
) -> List[Optional[Dict[str, Any]]]:
    """Extract temporal features from multiple texts.

    Args:
        texts: List of texts.

    Returns:
        List of feature dicts or None for each text.
    """
    extractor = TemporalExtractor()
    return [extractor.extract(text) for text in texts]
