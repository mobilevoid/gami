#!/usr/bin/env python3
"""Bulk ingest all known data sources into GAMI.

Walks through all data source paths from the implementation plan and
ingests them with appropriate source types and tenant assignments.

Usage:
    python -m cli.ingest_all [--api http://127.0.0.1:9090] [--dry-run]
"""
import argparse
import glob
import logging
import os
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("gami.cli.ingest_all")

DEFAULT_API = "http://127.0.0.1:9090"

# ---------------------------------------------------------------------------
# Data source definitions
# Each entry: (description, glob_or_finder, source_type, tenant_id)
# ---------------------------------------------------------------------------

DATA_SOURCES = [
    # 1. Claude Code conversation sessions
    {
        "name": "Claude Code sessions",
        "paths": [
            os.path.expanduser("~/.claude/projects/*/conversations/*.jsonl"),
        ],
        "source_type": "conversation",
        "tenant_id": "default",
    },
    # 2. Claude memory / project files
    {
        "name": "Claude memory files",
        "paths": [
            os.path.expanduser("~/.claude/projects/*/memory/*.md"),
            os.path.expanduser("~/.claude/projects/*/CLAUDE.md"),
        ],
        "source_type": "markdown",
        "tenant_id": "default",
    },
    # 3. OpenClaw SQLite memories
    {
        "name": "OpenClaw SQLite memories",
        "paths": [
            os.path.expanduser("~/agent-a/memory/*.sqlite"),
            os.path.expanduser("~/agent-a/memory/*.db"),
            os.path.expanduser("~/agent-a/.memory/*.sqlite"),
        ],
        "source_type": "sqlite_memory",
        "tenant_id": "agent-a",
    },
    # 4. OpenClaw sessions
    {
        "name": "OpenClaw conversation sessions",
        "paths": [
            os.path.expanduser("~/agent-a/sessions/*.jsonl"),
            os.path.expanduser("~/agent-a/conversations/*.jsonl"),
        ],
        "source_type": "conversation",
        "tenant_id": "agent-a",
    },
    # 5. Clawdbot SQLite memories
    {
        "name": "Clawdbot SQLite memories",
        "paths": [
            os.path.expanduser("~/agent-b/memory/*.sqlite"),
            os.path.expanduser("~/agent-b/memory/*.db"),
            os.path.expanduser("~/agent-b/.memory/*.sqlite"),
        ],
        "source_type": "sqlite_memory",
        "tenant_id": "agent-b",
    },
    # 6. Clawdbot sessions
    {
        "name": "Clawdbot conversation sessions",
        "paths": [
            os.path.expanduser("~/agent-b/sessions/*.jsonl"),
            os.path.expanduser("~/agent-b/conversations/*.jsonl"),
        ],
        "source_type": "conversation",
        "tenant_id": "agent-b",
    },
    # 7. Clawd memory + knowledge
    {
        "name": "Clawd knowledge docs",
        "paths": [
            os.path.expanduser("~/clawd/*.md"),
            os.path.expanduser("~/clawd/knowledge/*.md"),
            os.path.expanduser("~/clawd/memory/*.md"),
        ],
        "source_type": "markdown",
        "tenant_id": "clawd",
    },
    # 8. Agent-Zero missions
    {
        "name": "Agent-Zero missions",
        "paths": [
            os.path.expanduser("~/agent-zero/*.md"),
            os.path.expanduser("~/agent-zero/missions/*.md"),
            os.path.expanduser("~/agent-zero/missions/*.json"),
        ],
        "source_type": "markdown",
        "tenant_id": "agent-zero",
    },
    # 9. Top-level documentation
    {
        "name": "System documentation",
        "paths": [
            os.path.expanduser("~/CLAUDE.md"),
            os.path.expanduser("~/AI-STACK.md"),
            os.path.expanduser("~/.open-webui-memory/*.md"),
        ],
        "source_type": "markdown",
        "tenant_id": "shared",
    },
    # 10. Misc project docs
    {
        "name": "Project documentation (misc)",
        "paths": [
            os.path.expanduser("~/netcloud/*.md"),
            os.path.expanduser("~/stageover-layered/*.md"),
            os.path.expanduser("~/stageover-build/*.md"),
            os.path.expanduser("~/Downloads/*.md"),
        ],
        "source_type": "markdown",
        "tenant_id": "shared",
    },
]


def collect_files(source_def: dict) -> list[str]:
    """Expand all path globs for a source definition, return unique file list."""
    files = set()
    for pattern in source_def["paths"]:
        for f in glob.glob(pattern, recursive=True):
            if os.path.isfile(f):
                files.add(os.path.abspath(f))
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(
        description="Bulk ingest all known data sources into GAMI",
    )
    parser.add_argument(
        "--api",
        default=DEFAULT_API,
        help=f"GAMI API URL (default: {DEFAULT_API})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files without ingesting",
    )
    parser.add_argument(
        "--source",
        type=int,
        default=None,
        help="Only ingest source group N (1-based index)",
    )
    args = parser.parse_args()

    # Import ingest function from sibling module
    from cli.ingest_cmd import ingest_file

    total_files = 0
    total_success = 0
    total_skipped = 0
    total_failed = 0
    total_segments = 0
    start = time.time()

    sources = DATA_SOURCES
    if args.source is not None:
        idx = args.source - 1
        if 0 <= idx < len(DATA_SOURCES):
            sources = [DATA_SOURCES[idx]]
        else:
            logger.error("Invalid --source %d (range: 1-%d)", args.source, len(DATA_SOURCES))
            sys.exit(1)

    for i, src_def in enumerate(sources, 1):
        files = collect_files(src_def)
        logger.info(
            "[Group %d/%d] %s: %d files (type=%s, tenant=%s)",
            i, len(sources), src_def["name"], len(files),
            src_def["source_type"], src_def["tenant_id"],
        )

        if not files:
            continue

        total_files += len(files)

        if args.dry_run:
            for f in files:
                logger.info("  [DRY RUN] %s (%d bytes)", f, os.path.getsize(f))
            continue

        for j, file_path in enumerate(files, 1):
            try:
                result = ingest_file(
                    api_url=args.api,
                    file_path=file_path,
                    source_type=src_def["source_type"],
                    tenant_id=src_def["tenant_id"],
                )
                status = result.get("status", "unknown")
                if status == "duplicate":
                    total_skipped += 1
                    logger.info(
                        "  [%d/%d] SKIP (dup) %s",
                        j, len(files), os.path.basename(file_path),
                    )
                elif status == "completed":
                    total_success += 1
                    segs = result.get("segments_created", 0)
                    total_segments += segs
                    logger.info(
                        "  [%d/%d] OK %s -> %s (%d segs)",
                        j, len(files), os.path.basename(file_path),
                        result.get("source_id"), segs,
                    )
                else:
                    logger.warning("  [%d/%d] %s: %s", j, len(files), file_path, result)
            except Exception as exc:
                total_failed += 1
                logger.error(
                    "  [%d/%d] FAIL %s: %s",
                    j, len(files), os.path.basename(file_path), exc,
                )

    elapsed = time.time() - start
    logger.info("=" * 60)
    logger.info("Bulk ingestion complete in %.1fs", elapsed)
    logger.info(
        "  Files: %d total, %d success, %d skipped, %d failed",
        total_files, total_success, total_skipped, total_failed,
    )
    logger.info("  Segments created: %d", total_segments)


if __name__ == "__main__":
    main()
