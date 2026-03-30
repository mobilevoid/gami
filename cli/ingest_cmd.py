#!/usr/bin/env python3
"""CLI for batch ingestion of files into GAMI.

Usage:
    python -m cli.ingest_cmd --path '*.md' --type markdown --tenant shared
    python -m cli.ingest_cmd --path /path/to/file.md --type markdown
    python -m cli.ingest_cmd --path '/home/ai/clawd/*.md' --type markdown --tenant clawd
"""
import argparse
import glob
import json
import logging
import os
import sys
import time

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("gami.cli.ingest")

DEFAULT_API = "http://127.0.0.1:9090"


def ingest_file(
    api_url: str,
    file_path: str,
    source_type: str,
    tenant_id: str,
    title: str = None,
    metadata: dict = None,
    timeout: float = 120.0,
) -> dict:
    """Ingest a single file via the GAMI API."""
    abs_path = os.path.abspath(file_path)
    data = {
        "file_path": abs_path,
        "source_type": source_type,
        "tenant_id": tenant_id,
        "metadata_json": json.dumps(metadata or {}),
    }
    if title:
        data["title"] = title

    resp = httpx.post(
        f"{api_url}/api/v1/ingest/source",
        data=data,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser(
        description="Ingest files into GAMI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--path",
        required=True,
        help="File path or glob pattern (e.g., '*.md', '/path/to/dir/**/*.jsonl')",
    )
    parser.add_argument(
        "--type",
        dest="source_type",
        default="markdown",
        help="Source type (markdown, conversation, sqlite_memory, plaintext)",
    )
    parser.add_argument(
        "--tenant",
        default="shared",
        help="Tenant ID (default: shared)",
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
        "--recursive", "-r",
        action="store_true",
        help="Enable recursive glob matching",
    )
    args = parser.parse_args()

    # Expand glob
    if "*" in args.path or "?" in args.path:
        files = sorted(glob.glob(args.path, recursive=args.recursive))
    elif os.path.isdir(args.path):
        files = sorted(glob.glob(os.path.join(args.path, "**", "*"), recursive=True))
        files = [f for f in files if os.path.isfile(f)]
    elif os.path.isfile(args.path):
        files = [args.path]
    else:
        logger.error("No files found matching: %s", args.path)
        sys.exit(1)

    if not files:
        logger.error("No files found matching: %s", args.path)
        sys.exit(1)

    logger.info("Found %d file(s) to ingest", len(files))

    if args.dry_run:
        for f in files:
            size = os.path.getsize(f)
            logger.info("  [DRY RUN] %s (%s bytes)", f, size)
        return

    # Ingest each file
    success = 0
    failed = 0
    skipped = 0
    total_segments = 0
    start = time.time()

    for i, file_path in enumerate(files, 1):
        try:
            result = ingest_file(
                api_url=args.api,
                file_path=file_path,
                source_type=args.source_type,
                tenant_id=args.tenant,
            )
            status = result.get("status", "unknown")
            if status == "duplicate":
                skipped += 1
                logger.info(
                    "[%d/%d] SKIP (dup) %s -> %s",
                    i, len(files), os.path.basename(file_path),
                    result.get("source_id"),
                )
            elif status == "completed":
                success += 1
                segs = result.get("segments_created", 0)
                total_segments += segs
                logger.info(
                    "[%d/%d] OK %s -> %s (%d segments)",
                    i, len(files), os.path.basename(file_path),
                    result.get("source_id"), segs,
                )
            else:
                logger.warning(
                    "[%d/%d] UNKNOWN STATUS %s: %s",
                    i, len(files), os.path.basename(file_path), result,
                )
        except Exception as exc:
            failed += 1
            logger.error(
                "[%d/%d] FAIL %s: %s",
                i, len(files), os.path.basename(file_path), exc,
            )

    elapsed = time.time() - start
    logger.info(
        "Done in %.1fs: %d success, %d skipped (dup), %d failed, %d total segments",
        elapsed, success, skipped, failed, total_segments,
    )


if __name__ == "__main__":
    main()
