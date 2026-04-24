"""Plaintext parser for GAMI — splits on paragraph boundaries.

Suitable for logs, config files, plain documentation, and any
unstructured text that doesn't match a more specific parser.
"""
import logging
import os
import re
from typing import Optional

from parsers.base import (
    BaseParser,
    ParseResult,
    ParsedSegment,
    register_parser,
)

logger = logging.getLogger(__name__)

# Extensions we explicitly claim (everything else can still be forced)
_TEXT_EXTENSIONS = {
    ".txt",
    ".log",
    ".conf",
    ".cfg",
    ".ini",
    ".env",
    ".csv",
    ".tsv",
    ".yml",
    ".yaml",
    ".toml",
    ".json",
    ".xml",
    ".html",
    ".htm",
    ".rst",
    ".tex",
    ".sh",
    ".bash",
    ".zsh",
    ".py",
    ".js",
    ".ts",
    ".go",
    ".rs",
    ".c",
    ".h",
    ".cpp",
    ".java",
    ".rb",
    ".pl",
    ".lua",
    ".sql",
    ".css",
    ".scss",
}

# Two or more consecutive newlines signal a paragraph break
_PARA_SPLIT = re.compile(r"\n{2,}")


@register_parser("plaintext")
class PlaintextParser(BaseParser):
    """Split plain text into paragraph-level segments."""

    # ------------------------------------------------------------------
    # BaseParser interface
    # ------------------------------------------------------------------

    def can_parse(self, file_path: str, mime_type: str = None) -> bool:
        if mime_type and mime_type.startswith("text/"):
            return True
        _, ext = os.path.splitext(file_path)
        return ext.lower() in _TEXT_EXTENSIONS

    def parse(self, file_path: str, metadata: dict = None) -> ParseResult:
        logger.info("Parsing plaintext file: %s", file_path)
        with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
            raw = fh.read()

        segments = self._split_paragraphs(raw)

        result_meta = dict(metadata or {})
        result_meta["checksum"] = self.compute_checksum(file_path)
        result_meta["file_path"] = file_path

        return ParseResult(
            title=os.path.basename(file_path),
            source_type="plaintext",
            segments=segments,
            metadata=result_meta,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _split_paragraphs(raw: str) -> list[ParsedSegment]:
        """Split *raw* on double-newlines, tracking line numbers."""
        segments: list[ParsedSegment] = []
        # Build a map of char-offset -> line-number
        line_starts: list[int] = [0]
        for i, ch in enumerate(raw):
            if ch == "\n":
                line_starts.append(i + 1)

        def _char_to_line(offset: int) -> int:
            """Return 1-based line number for a character offset."""
            lo, hi = 0, len(line_starts) - 1
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if line_starts[mid] <= offset:
                    lo = mid
                else:
                    hi = mid - 1
            return lo + 1  # 1-based

        paragraphs = _PARA_SPLIT.split(raw)
        ordinal = 0
        pos = 0  # running character position

        for para in paragraphs:
            text = para.strip()
            # Advance pos to find this paragraph in the raw string
            idx = raw.find(para, pos)
            if idx == -1:
                idx = pos
            if text:
                segments.append(
                    ParsedSegment(
                        text=text,
                        segment_type="paragraph",
                        ordinal=ordinal,
                        char_start=idx,
                        char_end=idx + len(para),
                        line_start=_char_to_line(idx),
                        line_end=_char_to_line(idx + len(para) - 1),
                    )
                )
                ordinal += 1
            pos = idx + len(para)

        return segments
