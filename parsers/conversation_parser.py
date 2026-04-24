"""Claude Code JSONL session parser for GAMI.

Parses the JSONL files produced by Claude Code (one JSON object per line)
into structured conversation segments suitable for knowledge extraction.
"""
import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

from parsers.base import (
    BaseParser,
    ParseResult,
    ParsedSegment,
    register_parser,
)

logger = logging.getLogger(__name__)

# Message types we extract meaningful content from
_CONTENT_TYPES = {"user", "assistant"}
# Types we skip entirely (noise)
_SKIP_TYPES = {
    "progress",
    "queue-operation",
    "file-history-snapshot",
    "last-prompt",
}


@register_parser("conversation")
class ConversationParser(BaseParser):
    """Parse Claude Code JSONL session files into conversation segments."""

    # ------------------------------------------------------------------
    # BaseParser interface
    # ------------------------------------------------------------------

    def can_parse(self, file_path: str, mime_type: str = None) -> bool:
        if mime_type and "jsonl" in mime_type:
            return True
        return file_path.endswith(".jsonl")

    def parse(self, file_path: str, metadata: dict = None) -> ParseResult:
        logger.info("Parsing conversation JSONL: %s", file_path)
        raw_messages = self._load_jsonl(file_path)
        segments = self._messages_to_segments(raw_messages)

        session_id = self._extract_session_id(raw_messages)
        slug = self._extract_slug(raw_messages)
        title = slug or session_id or os.path.basename(file_path)

        result_meta = dict(metadata or {})
        result_meta["session_id"] = session_id
        result_meta["slug"] = slug
        result_meta["checksum"] = self.compute_checksum(file_path)
        result_meta["file_path"] = file_path
        result_meta["total_raw_lines"] = len(raw_messages)
        result_meta["version"] = self._extract_version(raw_messages)

        return ParseResult(
            title=title,
            source_type="conversation",
            segments=segments,
            metadata=result_meta,
        )

    # ------------------------------------------------------------------
    # JSONL loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_jsonl(file_path: str) -> list[dict]:
        messages: list[dict] = []
        with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    messages.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(
                        "Skipping malformed JSON at %s:%d", file_path, line_no
                    )
        return messages

    # ------------------------------------------------------------------
    # Metadata extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_session_id(msgs: list[dict]) -> Optional[str]:
        for m in msgs:
            sid = m.get("sessionId")
            if sid:
                return sid
        return None

    @staticmethod
    def _extract_slug(msgs: list[dict]) -> Optional[str]:
        for m in msgs:
            slug = m.get("slug")
            if slug:
                return slug
        return None

    @staticmethod
    def _extract_version(msgs: list[dict]) -> Optional[str]:
        for m in msgs:
            v = m.get("version")
            if v:
                return v
        return None

    # ------------------------------------------------------------------
    # Segment extraction
    # ------------------------------------------------------------------

    def _messages_to_segments(self, msgs: list[dict]) -> list[ParsedSegment]:
        segments: list[ParsedSegment] = []
        ordinal = 0

        for msg in msgs:
            msg_type = msg.get("type", "")

            # Skip noise
            if msg_type in _SKIP_TYPES:
                continue

            # System messages carry context (CLAUDE.md, etc.) — keep them
            if msg_type == "system":
                text = self._extract_system_text(msg)
                if not text:
                    continue
                segments.append(
                    ParsedSegment(
                        text=text,
                        segment_type="system_context",
                        ordinal=ordinal,
                        speaker_role="system",
                        message_timestamp=self._parse_ts(msg),
                        metadata=self._base_meta(msg),
                    )
                )
                ordinal += 1
                continue

            if msg_type not in _CONTENT_TYPES:
                continue

            message_obj = msg.get("message", {})
            if not isinstance(message_obj, dict):
                continue

            role = message_obj.get("role", msg_type)
            content = message_obj.get("content", "")
            timestamp = self._parse_ts(msg)
            base = self._base_meta(msg)

            # Skip meta / command caveat messages
            if msg.get("isMeta"):
                continue

            # ---- content is a plain string --------------------------------
            if isinstance(content, str):
                text = content.strip()
                if not text:
                    continue
                # Detect tool_result delivered as user message
                tool_result = msg.get("toolUseResult")
                if tool_result is not None:
                    seg_type = "tool_result"
                    base["tool_use_id"] = self._find_tool_use_id(content, msg)
                    base["is_error"] = tool_result.get("is_error", False)
                    base["is_image"] = tool_result.get("isImage", False)
                    if base["is_image"]:
                        text = "[image output omitted]"
                else:
                    seg_type = "message"

                segments.append(
                    ParsedSegment(
                        text=text,
                        segment_type=seg_type,
                        ordinal=ordinal,
                        speaker_role=role,
                        message_timestamp=timestamp,
                        metadata=base,
                    )
                )
                ordinal += 1
                continue

            # ---- content is an array of content blocks --------------------
            if isinstance(content, list):
                for block in content:
                    seg = self._parse_content_block(
                        block, role, timestamp, ordinal, base
                    )
                    if seg is not None:
                        segments.append(seg)
                        ordinal += 1

        return segments

    # ------------------------------------------------------------------
    # Content-block handling
    # ------------------------------------------------------------------

    def _parse_content_block(
        self,
        block: dict,
        role: str,
        timestamp: Optional[datetime],
        ordinal: int,
        base_meta: dict,
    ) -> Optional[ParsedSegment]:
        btype = block.get("type", "")

        # --- text block ---
        if btype == "text":
            text = block.get("text", "").strip()
            if not text:
                return None
            return ParsedSegment(
                text=text,
                segment_type="message",
                ordinal=ordinal,
                speaker_role=role,
                message_timestamp=timestamp,
                metadata=base_meta,
            )

        # --- thinking block (extended thinking) ---
        if btype == "thinking":
            thinking = block.get("thinking", "").strip()
            if not thinking:
                return None
            return ParsedSegment(
                text=thinking,
                segment_type="thinking",
                ordinal=ordinal,
                speaker_role="assistant",
                message_timestamp=timestamp,
                metadata=base_meta,
            )

        # --- tool_use block ---
        if btype == "tool_use":
            tool_name = block.get("name", "unknown")
            tool_input = block.get("input", {})
            # Build a concise textual representation
            text = self._tool_use_to_text(tool_name, tool_input)
            meta = dict(base_meta)
            meta["tool_name"] = tool_name
            meta["tool_use_id"] = block.get("id")
            meta["tool_input"] = tool_input
            return ParsedSegment(
                text=text,
                segment_type="tool_call",
                ordinal=ordinal,
                speaker_role="assistant",
                message_timestamp=timestamp,
                metadata=meta,
            )

        # --- tool_result block (inside user content array) ---
        if btype == "tool_result":
            result_content = block.get("content", "")
            is_error = block.get("is_error", False)
            # content may be string or list of {type: text/image, ...}
            text = self._flatten_tool_result_content(result_content)
            if not text:
                return None
            meta = dict(base_meta)
            meta["tool_use_id"] = block.get("tool_use_id")
            meta["is_error"] = is_error
            return ParsedSegment(
                text=text,
                segment_type="tool_result",
                ordinal=ordinal,
                speaker_role="tool",
                message_timestamp=timestamp,
                metadata=meta,
            )

        # Unknown block type — skip silently
        return None

    # ------------------------------------------------------------------
    # Tool text helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tool_use_to_text(name: str, inp: dict) -> str:
        """Create a human-readable summary of a tool invocation."""
        if name == "Bash":
            cmd = inp.get("command", "")
            desc = inp.get("description", "")
            label = desc if desc else cmd[:120]
            return f"[Bash] {label}"
        if name in ("Read", "ReadFile"):
            return f"[Read] {inp.get('file_path', '?')}"
        if name in ("Write", "WriteFile"):
            return f"[Write] {inp.get('file_path', '?')}"
        if name == "Edit":
            return f"[Edit] {inp.get('file_path', '?')}"
        if name == "Grep":
            return f"[Grep] pattern={inp.get('pattern', '?')} path={inp.get('path', '.')}"
        if name == "Glob":
            return f"[Glob] {inp.get('pattern', '?')}"
        # Generic fallback
        summary = json.dumps(inp, ensure_ascii=False)
        if len(summary) > 200:
            summary = summary[:200] + "..."
        return f"[{name}] {summary}"

    @staticmethod
    def _flatten_tool_result_content(content) -> str:
        """Flatten tool result content to plain text, skipping images."""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    if item.get("type") == "image":
                        parts.append("[image output omitted]")
                    elif item.get("type") == "text":
                        parts.append(item.get("text", ""))
            return "\n".join(parts).strip()
        return ""

    @staticmethod
    def _find_tool_use_id(content, msg: dict) -> Optional[str]:
        """Try to extract tool_use_id from various locations."""
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and "tool_use_id" in block:
                    return block["tool_use_id"]
        return msg.get("sourceToolUseId")

    # ------------------------------------------------------------------
    # Timestamp parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_ts(msg: dict) -> Optional[datetime]:
        ts = msg.get("timestamp")
        if ts is None:
            return None
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc)
        if isinstance(ts, str):
            # ISO 8601 with optional Z
            ts = ts.replace("Z", "+00:00")
            try:
                return datetime.fromisoformat(ts)
            except ValueError:
                return None
        return None

    @staticmethod
    def _extract_system_text(msg: dict) -> str:
        """Extract text from a system-type message."""
        message_obj = msg.get("message", {})
        if isinstance(message_obj, dict):
            content = message_obj.get("content", "")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, str):
                        parts.append(block)
                    elif isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text", ""))
                return "\n".join(parts).strip()
        if isinstance(message_obj, str):
            return message_obj.strip()
        # Fallback: check top-level content
        content = msg.get("content", "")
        if isinstance(content, str):
            return content.strip()
        return ""

    @staticmethod
    def _base_meta(msg: dict) -> dict:
        meta: dict = {}
        for key in ("uuid", "parentUuid", "sessionId"):
            val = msg.get(key)
            if val is not None:
                meta[key] = val
        return meta
