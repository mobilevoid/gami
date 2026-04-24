"""Base parser class and registry for GAMI source ingestion."""
import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ParsedSegment:
    """A parsed segment from a source document."""

    text: str
    segment_type: str  # chapter, section, paragraph, message, tool_call, etc.
    ordinal: int = 0
    depth: int = 0
    title_or_heading: Optional[str] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    speaker_role: Optional[str] = None
    speaker_name: Optional[str] = None
    message_timestamp: Optional[datetime] = None
    children: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class ParseResult:
    """Result of parsing a source."""

    title: str
    source_type: str
    segments: list[ParsedSegment]
    metadata: dict = field(default_factory=dict)
    author: Optional[str] = None
    language: str = "en"


class BaseParser(ABC):
    """Base class for all GAMI parsers."""

    @abstractmethod
    def can_parse(self, file_path: str, mime_type: str = None) -> bool:
        """Check if this parser can handle the given file."""
        pass

    @abstractmethod
    def parse(self, file_path: str, metadata: dict = None) -> ParseResult:
        """Parse a file and return structured segments."""
        pass

    @staticmethod
    def compute_checksum(file_path: str) -> str:
        """Compute SHA256 checksum of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return f"sha256:{sha256.hexdigest()}"

    @staticmethod
    def compute_text_checksum(text: str) -> str:
        """Compute SHA256 checksum of a text string."""
        return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


# ---------------------------------------------------------------------------
# Parser registry
# ---------------------------------------------------------------------------
_parsers: dict[str, type[BaseParser]] = {}


def register_parser(name: str):
    """Decorator that registers a parser class under *name*."""

    def decorator(cls):
        _parsers[name] = cls
        return cls

    return decorator


def get_parser(name: str) -> BaseParser:
    """Instantiate a registered parser by name."""
    if name not in _parsers:
        raise ValueError(
            f"Unknown parser: {name}. Available: {list(_parsers.keys())}"
        )
    return _parsers[name]()


def get_parser_for_file(file_path: str, mime_type: str = None) -> BaseParser:
    """Return the first parser that claims it can handle *file_path*."""
    for name, parser_cls in _parsers.items():
        p = parser_cls()
        if p.can_parse(file_path, mime_type):
            return p
    raise ValueError(f"No parser found for {file_path}")
