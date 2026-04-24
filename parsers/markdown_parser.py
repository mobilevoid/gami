"""Markdown parser for GAMI — preserves heading hierarchy and frontmatter.

Splits markdown documents into hierarchical segments based on headings,
preserving parent-child relationships and code blocks.
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


@register_parser("markdown")
class MarkdownParser(BaseParser):
    """Parse markdown files into hierarchical section segments."""

    # ------------------------------------------------------------------
    # BaseParser interface
    # ------------------------------------------------------------------

    def can_parse(self, file_path: str, mime_type: str = None) -> bool:
        if mime_type and "markdown" in mime_type:
            return True
        return file_path.lower().endswith(".md")

    def parse(self, file_path: str, metadata: dict = None) -> ParseResult:
        logger.info("Parsing markdown file: %s", file_path)
        with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
            content = fh.read()

        segments = self._parse_content(content)

        result_meta = dict(metadata or {})
        result_meta["checksum"] = self.compute_checksum(file_path)
        result_meta["file_path"] = file_path

        # Use the first heading as the title, or filename
        title = os.path.basename(file_path)
        for seg in segments:
            if seg.title_or_heading:
                title = seg.title_or_heading
                break

        return ParseResult(
            title=title,
            source_type="markdown",
            segments=segments,
            metadata=result_meta,
        )

    # ------------------------------------------------------------------
    # Internal parsing
    # ------------------------------------------------------------------

    _HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")

    def _parse_content(self, content: str) -> list[ParsedSegment]:
        lines = content.split("\n")
        segments: list[ParsedSegment] = []
        ordinal = 0
        char_offset = 0

        # heading_stack[depth] = ordinal of that heading's segment
        heading_stack: dict[int, int] = {}

        current_text_lines: list[str] = []
        current_heading: Optional[str] = None
        current_depth: int = 0
        current_parent_ordinal: Optional[int] = None
        current_line_start: int = 1
        current_char_start: int = 0

        in_code_block = False
        code_block_lines: list[str] = []
        code_block_start_line: int = 0
        code_block_char_start: int = 0

        def _flush_text():
            nonlocal ordinal
            text_body = "\n".join(current_text_lines).strip()
            if not text_body:
                return

            if current_heading:
                seg_type = "section"
                title = current_heading
            else:
                seg_type = "paragraph"
                title = None

            seg = ParsedSegment(
                text=text_body,
                segment_type=seg_type,
                ordinal=ordinal,
                depth=current_depth,
                title_or_heading=title,
                char_start=current_char_start,
                char_end=current_char_start + len(text_body),
                line_start=current_line_start,
                line_end=current_line_start + len(current_text_lines) - 1,
                metadata={"parent_ordinal": current_parent_ordinal},
            )
            segments.append(seg)
            ordinal += 1

        for line_num, line in enumerate(lines, start=1):
            line_char_start = char_offset

            # Handle fenced code blocks
            if line.strip().startswith("```"):
                if not in_code_block:
                    _flush_text()
                    current_text_lines = []
                    in_code_block = True
                    code_block_lines = [line]
                    code_block_start_line = line_num
                    code_block_char_start = line_char_start
                else:
                    code_block_lines.append(line)
                    code_text = "\n".join(code_block_lines)
                    seg = ParsedSegment(
                        text=code_text,
                        segment_type="code_block",
                        ordinal=ordinal,
                        depth=current_depth + 1,
                        char_start=code_block_char_start,
                        char_end=line_char_start + len(line),
                        line_start=code_block_start_line,
                        line_end=line_num,
                        metadata={
                            "parent_ordinal": heading_stack.get(current_depth),
                        },
                    )
                    segments.append(seg)
                    ordinal += 1
                    in_code_block = False
                    code_block_lines = []
                char_offset += len(line) + 1
                continue

            if in_code_block:
                code_block_lines.append(line)
                char_offset += len(line) + 1
                continue

            # Check for heading
            m = self._HEADING_RE.match(line)
            if m:
                _flush_text()
                current_text_lines = []

                depth = len(m.group(1))
                heading_text = m.group(2).strip()

                parent_ord = None
                for d in range(depth - 1, 0, -1):
                    if d in heading_stack:
                        parent_ord = heading_stack[d]
                        break

                heading_stack[depth] = ordinal
                for d in list(heading_stack.keys()):
                    if d > depth:
                        del heading_stack[d]

                current_heading = heading_text
                current_depth = depth
                current_parent_ordinal = parent_ord
                current_line_start = line_num
                current_char_start = line_char_start
            else:
                current_text_lines.append(line)

            char_offset += len(line) + 1

        # Flush remaining
        if in_code_block and code_block_lines:
            seg = ParsedSegment(
                text="\n".join(code_block_lines),
                segment_type="code_block",
                ordinal=ordinal,
                depth=current_depth + 1,
                char_start=code_block_char_start,
                char_end=char_offset,
                line_start=code_block_start_line,
                line_end=len(lines),
                metadata={
                    "parent_ordinal": heading_stack.get(current_depth),
                },
            )
            segments.append(seg)
        else:
            _flush_text()

        return segments
