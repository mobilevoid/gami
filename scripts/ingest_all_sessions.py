#!/usr/bin/env python3
"""Bulk ingest all conversation sessions and SQLite memories into GAMI.

Uses direct Python imports (sync DB path) to avoid HTTP timeout issues
with large files. The API server is NOT required for this script.

NOTE: This is an example script. Customize the DATA_SOURCES and tenant IDs
below for your specific setup.

Usage:
    cd /path/to/gami && PYTHONPATH=/path/to/gami python3 scripts/ingest_all_sessions.py
"""
import glob
import json
import logging
import os
import sys
import time
import uuid
import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path

# Add GAMI to path
GAMI_DIR = os.getenv("GAMI_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, GAMI_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(GAMI_DIR, ".env"))

import sqlalchemy
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from parsers import get_parser
from parsers.base import ParsedSegment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ingest_all")

# ── DB setup (sync) ──────────────────────────────────────────────────────
DB_URL = (
    f"postgresql+psycopg2://"
    f"{os.getenv('GAMI_DB_USER', 'gami')}:{os.getenv('GAMI_DB_PASSWORD', '')}@"
    f"{os.getenv('GAMI_DB_HOST', '127.0.0.1')}:{os.getenv('GAMI_DB_PORT', '5433')}/"
    f"{os.getenv('GAMI_DB_NAME', 'gami')}"
)
engine = create_engine(DB_URL, pool_pre_ping=True)
Session = sessionmaker(bind=engine)

OBJECT_STORE = os.getenv("GAMI_OBJECT_STORE", "/opt/gami/storage/objects")

# ── Tiktoken for token counting ──────────────────────────────────────────
import tiktoken
_encoder = tiktoken.get_encoding("cl100k_base")

def count_tokens(text_str: str) -> int:
    # Strip NUL characters and allow special tokens
    clean = text_str.replace("\x00", "")
    try:
        return len(_encoder.encode(clean))
    except Exception:
        return len(_encoder.encode(clean, disallowed_special=()))


# ── Helper functions ─────────────────────────────────────────────────────

def compute_checksum(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def check_duplicate(db, tenant_id: str, checksum: str):
    """Return existing source_id if duplicate, else None."""
    result = db.execute(
        text("SELECT source_id FROM sources WHERE checksum = :cs AND owner_tenant_id = :tid"),
        {"cs": checksum, "tid": tenant_id},
    )
    row = result.fetchone()
    return row[0] if row else None


def register_source(db, tenant_id: str, file_path: str, source_type: str, title: str, checksum: str):
    """Register source in DB and copy raw file."""
    file_size = os.path.getsize(file_path)
    source_id = f"SRC_{source_type.upper()}_{uuid.uuid4().hex[:8]}"

    ext = os.path.splitext(file_path)[1].lower()
    mime_map = {
        ".jsonl": "application/x-jsonl",
        ".sqlite": "application/x-sqlite3",
        ".db": "application/x-sqlite3",
    }
    mime_type = mime_map.get(ext, "application/octet-stream")

    # Copy raw file
    raw_dir = os.path.join(OBJECT_STORE, "raw", source_id)
    os.makedirs(raw_dir, exist_ok=True)
    dest_path = os.path.join(raw_dir, os.path.basename(file_path))
    shutil.copy2(file_path, dest_path)

    now = datetime.now(timezone.utc)
    db.execute(
        text("""
            INSERT INTO sources (
                source_id, owner_tenant_id, source_type, title,
                source_uri, raw_file_path, checksum, file_size_bytes,
                mime_type, parse_status, metadata_json, ingested_at
            ) VALUES (
                :source_id, :tenant_id, :source_type, :title,
                :source_uri, :raw_file_path, :checksum, :file_size,
                :mime_type, 'pending', CAST(:metadata AS jsonb), :now
            )
        """),
        {
            "source_id": source_id,
            "tenant_id": tenant_id,
            "source_type": source_type,
            "title": title,
            "source_uri": f"file://{os.path.abspath(file_path)}",
            "raw_file_path": dest_path,
            "checksum": checksum,
            "file_size": file_size,
            "mime_type": mime_type,
            "metadata": json.dumps({}),
            "now": now,
        },
    )
    db.commit()
    return source_id, dest_path


def store_segments(db, source_id: str, tenant_id: str, segments: list):
    """Store parsed segments in DB."""
    if not segments:
        return 0

    now = datetime.now(timezone.utc)
    count = 0

    # Pre-compute segment IDs
    ordinal_to_id = {}
    for seg in segments:
        sid = f"SEG_{source_id}_{seg.ordinal}"
        ordinal_to_id[seg.ordinal] = sid

    INSERT_SQL = text("""
        INSERT INTO segments (
            segment_id, source_id, owner_tenant_id,
            parent_segment_id, segment_type, ordinal, depth,
            title_or_heading, text, token_count,
            char_start, char_end, line_start, line_end,
            speaker_role, speaker_name, message_timestamp,
            language, quality_flags_json, storage_tier, created_at
        ) VALUES (
            :seg_id, :source_id, :tenant_id,
            :parent_id, :seg_type, :ordinal, :depth,
            :title, :text, :tokens,
            :char_start, :char_end, :line_start, :line_end,
            :speaker_role, :speaker_name, :msg_ts,
            :lang, CAST(:qf AS jsonb), 'hot', :now
        )
    """)

    for seg in segments:
        seg_id = ordinal_to_id[seg.ordinal]
        # Resolve parent
        parent_id = None
        if seg.metadata:
            parent_ord = seg.metadata.get("parent_ordinal")
            if parent_ord is not None and parent_ord in ordinal_to_id:
                parent_id = ordinal_to_id[parent_ord]

        # Clean NUL characters (PostgreSQL text columns reject \x00)
        clean_text = seg.text.replace("\x00", "") if seg.text else ""
        clean_title = seg.title_or_heading.replace("\x00", "") if seg.title_or_heading else seg.title_or_heading
        token_count = count_tokens(clean_text)
        quality_flags = seg.metadata.get("quality_flags", {}) if seg.metadata else {}

        db.execute(INSERT_SQL, {
            "seg_id": seg_id,
            "source_id": source_id,
            "tenant_id": tenant_id,
            "parent_id": parent_id,
            "seg_type": seg.segment_type,
            "ordinal": seg.ordinal,
            "depth": seg.depth,
            "title": clean_title,
            "text": clean_text,
            "tokens": token_count,
            "char_start": seg.char_start,
            "char_end": seg.char_end,
            "line_start": seg.line_start,
            "line_end": seg.line_end,
            "speaker_role": seg.speaker_role,
            "speaker_name": seg.speaker_name,
            "msg_ts": seg.message_timestamp,
            "lang": "en",
            "qf": json.dumps(quality_flags),
            "now": now,
        })
        count += 1

    db.commit()
    return count


def update_parse_status(db, source_id: str, status: str):
    db.execute(
        text("UPDATE sources SET parse_status = :status, parser_version = '1.0.0' WHERE source_id = :sid"),
        {"status": status, "sid": source_id},
    )
    db.commit()


def create_job(db, tenant_id: str, source_id: str, segments_created: int, source_type: str, parser_name: str):
    job_id = f"JOB_INGEST_{uuid.uuid4().hex[:8]}"
    now = datetime.now(timezone.utc)
    db.execute(
        text("""
            INSERT INTO jobs (
                job_id, owner_tenant_id, job_type, target_type,
                target_id, status, priority, attempt_count,
                max_attempts, scheduled_at, started_at, completed_at,
                result_json, created_at
            ) VALUES (
                :jid, :tid, 'ingest', 'source', :sid, 'completed',
                0, 1, 1, :now, :now, :now,
                CAST(:result AS jsonb), :now
            )
        """),
        {
            "jid": job_id,
            "tid": tenant_id,
            "sid": source_id,
            "now": now,
            "result": json.dumps({
                "segments_created": segments_created,
                "source_type": source_type,
                "parser": parser_name,
            }),
        },
    )
    db.commit()
    return job_id


def ingest_file(file_path: str, source_type: str, tenant_id: str, title: str = None) -> dict:
    """Ingest a single file. Returns result dict."""
    db = Session()
    try:
        file_size = os.path.getsize(file_path)
        size_mb = file_size / (1024 * 1024)

        # Compute checksum
        checksum = compute_checksum(file_path)

        # Check duplicate
        existing = check_duplicate(db, tenant_id, checksum)
        if existing:
            return {"status": "duplicate", "source_id": existing, "file": file_path}

        # Get parser
        parser = get_parser(source_type)

        # Register source
        if title is None:
            title = os.path.basename(file_path)
        source_id, dest_path = register_source(db, tenant_id, file_path, source_type, title, checksum)

        # Parse
        t0 = time.time()
        parse_result = parser.parse(file_path, {})
        parse_time = time.time() - t0

        # Store segments
        t1 = time.time()
        seg_count = store_segments(db, source_id, tenant_id, parse_result.segments)
        store_time = time.time() - t1

        # Update status
        update_parse_status(db, source_id, "parsed")

        # Create job record
        job_id = create_job(db, tenant_id, source_id, seg_count, source_type, type(parser).__name__)

        return {
            "status": "completed",
            "source_id": source_id,
            "segments": seg_count,
            "size_mb": round(size_mb, 1),
            "parse_time": round(parse_time, 1),
            "store_time": round(store_time, 1),
            "file": file_path,
        }

    except Exception as e:
        db.rollback()
        logger.error("Failed to ingest %s: %s", file_path, e)
        return {"status": "error", "file": file_path, "error": str(e)}
    finally:
        db.close()


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    total_start = time.time()
    stats = {
        "completed": 0,
        "duplicate": 0,
        "error": 0,
        "total_segments": 0,
        "total_size_mb": 0,
    }
    errors = []

    # ─── 1. Claude Code JSONL sessions ────────────────────────────────
    claude_files = sorted(glob.glob(os.path.expanduser("~/.claude/projects/*/*.jsonl")))
    logger.info("=" * 70)
    logger.info("PHASE 1: Claude Code JSONL sessions (%d files)", len(claude_files))
    logger.info("=" * 70)

    for i, f in enumerate(claude_files, 1):
        size_mb = os.path.getsize(f) / (1024 * 1024)
        logger.info("[%d/%d] %s (%.1f MB)", i, len(claude_files), os.path.basename(f), size_mb)
        result = ingest_file(f, "conversation", "default")
        if result["status"] == "completed":
            stats["completed"] += 1
            stats["total_segments"] += result["segments"]
            stats["total_size_mb"] += result.get("size_mb", 0)
            logger.info("  -> OK: %s, %d segments (parse: %.1fs, store: %.1fs)",
                        result["source_id"], result["segments"],
                        result["parse_time"], result["store_time"])
        elif result["status"] == "duplicate":
            stats["duplicate"] += 1
            logger.info("  -> DUPLICATE (existing: %s)", result["source_id"])
        else:
            stats["error"] += 1
            errors.append(result)
            logger.error("  -> ERROR: %s", result.get("error", "unknown"))

    # ─── 2. OpenClaw JSONL sessions ───────────────────────────────────
    agent-a_files = sorted(glob.glob(os.path.expanduser("~/.agent-a/**/*.jsonl"), recursive=True))
    logger.info("")
    logger.info("=" * 70)
    logger.info("PHASE 2: OpenClaw JSONL sessions (%d files)", len(agent-a_files))
    logger.info("=" * 70)

    for i, f in enumerate(agent-a_files, 1):
        size_mb = os.path.getsize(f) / (1024 * 1024)
        logger.info("[%d/%d] %s (%.1f MB)", i, len(agent-a_files), os.path.basename(f), size_mb)
        result = ingest_file(f, "conversation", "agent-a")
        if result["status"] == "completed":
            stats["completed"] += 1
            stats["total_segments"] += result["segments"]
            stats["total_size_mb"] += result.get("size_mb", 0)
            logger.info("  -> OK: %s, %d segments (parse: %.1fs, store: %.1fs)",
                        result["source_id"], result["segments"],
                        result["parse_time"], result["store_time"])
        elif result["status"] == "duplicate":
            stats["duplicate"] += 1
            logger.info("  -> DUPLICATE (existing: %s)", result["source_id"])
        else:
            stats["error"] += 1
            errors.append(result)
            logger.error("  -> ERROR: %s", result.get("error", "unknown"))

    # ─── 3. Clawdbot JSONL sessions ──────────────────────────────────
    agent-b_files = sorted(glob.glob(os.path.expanduser("~/.agent-b/**/*.jsonl"), recursive=True))
    logger.info("")
    logger.info("=" * 70)
    logger.info("PHASE 3: Clawdbot JSONL sessions (%d files)", len(agent-b_files))
    logger.info("=" * 70)

    for i, f in enumerate(agent-b_files, 1):
        size_mb = os.path.getsize(f) / (1024 * 1024)
        logger.info("[%d/%d] %s (%.1f MB)", i, len(agent-b_files), os.path.basename(f), size_mb)
        result = ingest_file(f, "conversation", "agent-b")
        if result["status"] == "completed":
            stats["completed"] += 1
            stats["total_segments"] += result["segments"]
            stats["total_size_mb"] += result.get("size_mb", 0)
            logger.info("  -> OK: %s, %d segments (parse: %.1fs, store: %.1fs)",
                        result["source_id"], result["segments"],
                        result["parse_time"], result["store_time"])
        elif result["status"] == "duplicate":
            stats["duplicate"] += 1
            logger.info("  -> DUPLICATE (existing: %s)", result["source_id"])
        else:
            stats["error"] += 1
            errors.append(result)
            logger.error("  -> ERROR: %s", result.get("error", "unknown"))

    # ─── 4. OpenClaw SQLite memory ────────────────────────────────────
    agent-a_sqlite = os.path.expanduser("~/.agent-a/memory/main.sqlite")
    logger.info("")
    logger.info("=" * 70)
    logger.info("PHASE 4: OpenClaw SQLite memory")
    logger.info("=" * 70)

    if os.path.isfile(agent-a_sqlite):
        size_mb = os.path.getsize(agent-a_sqlite) / (1024 * 1024)
        logger.info("File: %s (%.1f MB)", agent-a_sqlite, size_mb)
        result = ingest_file(agent-a_sqlite, "sqlite_memory", "agent-a", title="OpenClaw Memory DB")
        if result["status"] == "completed":
            stats["completed"] += 1
            stats["total_segments"] += result["segments"]
            stats["total_size_mb"] += result.get("size_mb", 0)
            logger.info("  -> OK: %s, %d segments (parse: %.1fs, store: %.1fs)",
                        result["source_id"], result["segments"],
                        result["parse_time"], result["store_time"])
        elif result["status"] == "duplicate":
            stats["duplicate"] += 1
            logger.info("  -> DUPLICATE (existing: %s)", result["source_id"])
        else:
            stats["error"] += 1
            errors.append(result)
            logger.error("  -> ERROR: %s", result.get("error", "unknown"))
    else:
        logger.warning("File not found: %s", agent-a_sqlite)

    # ─── 5. Clawdbot SQLite memory ───────────────────────────────────
    agent-b_sqlite = os.path.expanduser("~/.agent-b/memory/main.sqlite")
    logger.info("")
    logger.info("=" * 70)
    logger.info("PHASE 5: Clawdbot SQLite memory")
    logger.info("=" * 70)

    if os.path.isfile(agent-b_sqlite):
        size_mb = os.path.getsize(agent-b_sqlite) / (1024 * 1024)
        logger.info("File: %s (%.1f MB)", agent-b_sqlite, size_mb)
        result = ingest_file(agent-b_sqlite, "sqlite_memory", "agent-b", title="Clawdbot Memory DB")
        if result["status"] == "completed":
            stats["completed"] += 1
            stats["total_segments"] += result["segments"]
            stats["total_size_mb"] += result.get("size_mb", 0)
            logger.info("  -> OK: %s, %d segments (parse: %.1fs, store: %.1fs)",
                        result["source_id"], result["segments"],
                        result["parse_time"], result["store_time"])
        elif result["status"] == "duplicate":
            stats["duplicate"] += 1
            logger.info("  -> DUPLICATE (existing: %s)", result["source_id"])
        else:
            stats["error"] += 1
            errors.append(result)
            logger.error("  -> ERROR: %s", result.get("error", "unknown"))
    else:
        logger.warning("File not found: %s", agent-b_sqlite)

    # ─── Summary ─────────────────────────────────────────────────────
    elapsed = time.time() - total_start
    logger.info("")
    logger.info("=" * 70)
    logger.info("INGESTION COMPLETE")
    logger.info("=" * 70)
    logger.info("Completed:  %d", stats["completed"])
    logger.info("Duplicates: %d", stats["duplicate"])
    logger.info("Errors:     %d", stats["error"])
    logger.info("Total segments: %d", stats["total_segments"])
    logger.info("Total data: %.1f MB", stats["total_size_mb"])
    logger.info("Elapsed: %.0f seconds (%.1f minutes)", elapsed, elapsed / 60)

    if errors:
        logger.info("")
        logger.info("ERRORS:")
        for e in errors:
            logger.info("  %s: %s", e["file"], e.get("error", "unknown"))

    # ─── Final DB stats ──────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("DATABASE SUMMARY")
    logger.info("=" * 70)

    db = Session()
    try:
        result = db.execute(text("""
            SELECT s.owner_tenant_id,
                   count(DISTINCT s.source_id) as sources,
                   COALESCE(sum(seg.seg_count), 0) as total_segments,
                   COALESCE(sum(seg.token_count), 0) as total_tokens
            FROM sources s
            LEFT JOIN (
                SELECT source_id,
                       count(*) as seg_count,
                       sum(token_count) as token_count
                FROM segments
                GROUP BY source_id
            ) seg ON s.source_id = seg.source_id
            GROUP BY s.owner_tenant_id
            ORDER BY total_tokens DESC
        """))
        rows = result.fetchall()
        logger.info("%-20s %8s %12s %14s", "TENANT", "SOURCES", "SEGMENTS", "TOKENS")
        logger.info("-" * 60)
        total_sources = 0
        total_segs = 0
        total_toks = 0
        for row in rows:
            tenant, sources, segments, tokens = row
            logger.info("%-20s %8d %12d %14d", tenant, sources, segments, tokens)
            total_sources += sources
            total_segs += segments
            total_toks += tokens
        logger.info("-" * 60)
        logger.info("%-20s %8d %12d %14d", "TOTAL", total_sources, total_segs, total_toks)
    finally:
        db.close()


if __name__ == "__main__":
    main()
