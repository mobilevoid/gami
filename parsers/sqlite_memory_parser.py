"""OpenClaw / Clawdbot SQLite memory database parser for GAMI.

Reads the chunks table from an OpenClaw memory SQLite database and
maps each chunk to a ParsedSegment.  Embeddings are preserved in
metadata for optional downstream use.
"""
import json
import logging
import os
import sqlite3
from typing import Optional

from parsers.base import (
    BaseParser,
    ParseResult,
    ParsedSegment,
    register_parser,
)

logger = logging.getLogger(__name__)


@register_parser("sqlite_memory")
class SQLiteMemoryParser(BaseParser):
    """Parse OpenClaw/Clawdbot SQLite memory databases."""

    # ------------------------------------------------------------------
    # BaseParser interface
    # ------------------------------------------------------------------

    def can_parse(self, file_path: str, mime_type: str = None) -> bool:
        if not os.path.isfile(file_path):
            return False
        _, ext = os.path.splitext(file_path)
        if ext.lower() not in (".sqlite", ".sqlite3", ".db"):
            return False
        # Quick probe: does it contain a chunks table?
        try:
            con = sqlite3.connect(f"file:{file_path}?mode=ro", uri=True)
            cur = con.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='chunks'"
            )
            has_chunks = cur.fetchone() is not None
            con.close()
            return has_chunks
        except Exception:
            return False

    def parse(self, file_path: str, metadata: dict = None) -> ParseResult:
        logger.info("Parsing SQLite memory DB: %s", file_path)

        con = sqlite3.connect(f"file:{file_path}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row

        # ----- meta table -------------------------------------------------
        db_meta: dict = {}
        try:
            for row in con.execute("SELECT key, value FROM meta"):
                db_meta[row["key"]] = row["value"]
        except sqlite3.OperationalError:
            logger.debug("No meta table in %s", file_path)

        # ----- files table (for reference) --------------------------------
        files_map: dict[str, dict] = {}
        try:
            for row in con.execute("SELECT * FROM files"):
                files_map[row["path"]] = {
                    "source": row["source"],
                    "hash": row["hash"],
                    "mtime": row["mtime"],
                    "size": row["size"],
                }
        except sqlite3.OperationalError:
            logger.debug("No files table in %s", file_path)

        # ----- chunks table -----------------------------------------------
        segments: list[ParsedSegment] = []
        ordinal = 0

        try:
            cursor = con.execute(
                "SELECT id, path, source, start_line, end_line, "
                "hash, model, text, embedding, updated_at "
                "FROM chunks ORDER BY path, start_line"
            )
        except sqlite3.OperationalError as exc:
            logger.error("Cannot read chunks from %s: %s", file_path, exc)
            con.close()
            return ParseResult(
                title=os.path.basename(file_path),
                source_type="sqlite_memory",
                segments=[],
                metadata={"error": str(exc)},
            )

        for row in cursor:
            text = row["text"] or ""
            if not text.strip():
                continue

            chunk_meta: dict = {
                "chunk_id": row["id"],
                "source_path": row["path"],
                "source": row["source"],
                "hash": row["hash"],
                "model": row["model"],
                "updated_at": row["updated_at"],
            }

            # Parse embedding — stored as JSON array string
            embedding = row["embedding"]
            if embedding:
                try:
                    parsed_emb = json.loads(embedding)
                    chunk_meta["embedding_dims"] = len(parsed_emb)
                    # Store embedding as compact list (downstream can use it)
                    chunk_meta["embedding"] = parsed_emb
                except (json.JSONDecodeError, TypeError):
                    chunk_meta["embedding_dims"] = 0

            # Enrich with file metadata if available
            file_info = files_map.get(row["path"])
            if file_info:
                chunk_meta["file_size"] = file_info["size"]
                chunk_meta["file_mtime"] = file_info["mtime"]

            # Derive a heading from the path
            heading = row["path"] or None

            segments.append(
                ParsedSegment(
                    text=text.strip(),
                    segment_type="memory_chunk",
                    ordinal=ordinal,
                    depth=0,
                    title_or_heading=heading,
                    line_start=row["start_line"],
                    line_end=row["end_line"],
                    metadata=chunk_meta,
                )
            )
            ordinal += 1

        con.close()

        result_meta = dict(metadata or {})
        result_meta["checksum"] = self.compute_checksum(file_path)
        result_meta["file_path"] = file_path
        result_meta["db_meta"] = db_meta
        result_meta["files_count"] = len(files_map)
        result_meta["chunks_count"] = len(segments)

        title = db_meta.get("name", os.path.basename(file_path))

        return ParseResult(
            title=title,
            source_type="sqlite_memory",
            segments=segments,
            metadata=result_meta,
        )
