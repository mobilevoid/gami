#!/usr/bin/env python3
"""Bulk ingest files into a GAMI tenant.

Usage:
    python bulk_ingest.py --path /mnt/16tb/books/ --tenant books --type pdf
    python bulk_ingest.py --path /path/to/docs/ --tenant whitepapers --type markdown
    python bulk_ingest.py --path /path/to/manuals/ --tenant manuals --type pdf --workers 4

Features:
    - Auto-creates tenant if it doesn't exist
    - SHA256 checksum dedup (skips already-ingested files)
    - Parallel file processing with configurable workers
    - Progress reporting with ETA
    - Resume capability (skip files already in sources table by checksum)
    - Queues embeddings for background processing via embed_tenant.py
"""
import argparse
import hashlib
import logging
import os
import sys
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Optional

# Ensure GAMI root is in path
GAMI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if GAMI_ROOT not in sys.path:
    sys.path.insert(0, GAMI_ROOT)

from api.config import settings
from parsers.chunker import chunk_segments

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bulk_ingest")


def compute_file_checksum(file_path: str) -> str:
    """Compute SHA256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return f"sha256:{sha256.hexdigest()}"


def get_sync_engine():
    """Create a sync SQLAlchemy engine."""
    from sqlalchemy import create_engine
    return create_engine(
        settings.DATABASE_URL_SYNC,
        pool_size=5,
        max_overflow=5,
        pool_pre_ping=True,
    )


def ensure_tenant(engine, tenant_id: str, display_name: str):
    """Create tenant if it doesn't exist."""
    from sqlalchemy import text

    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT tenant_id FROM tenants WHERE tenant_id = :tid"),
            {"tid": tenant_id},
        ).fetchone()

        if not row:
            conn.execute(
                text(
                    "INSERT INTO tenants (tenant_id, name, description, tenant_type, status) "
                    "VALUES (:tid, :name, :desc, 'content', 'active')"
                ),
                {
                    "tid": tenant_id,
                    "name": display_name,
                    "desc": f"Auto-created by bulk_ingest on {datetime.now(timezone.utc).isoformat()}",
                },
            )
            conn.commit()
            logger.info("Created tenant: %s (%s)", tenant_id, display_name)
        else:
            logger.info("Tenant already exists: %s", tenant_id)


def get_existing_checksums(engine, tenant_id: str) -> set[str]:
    """Get all checksums already ingested for this tenant."""
    from sqlalchemy import text

    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT checksum FROM sources WHERE owner_tenant_id = :tid"),
            {"tid": tenant_id},
        ).fetchall()
    return {r[0] for r in rows}


def parse_single_file(file_path: str, file_type: str) -> Optional[dict]:
    """Parse a single file in a worker process. Returns serializable dict.

    This runs in a separate process, so we import everything locally.
    """
    try:
        gami_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if gami_root not in sys.path:
            sys.path.insert(0, gami_root)

        from parsers.chunker import chunk_segments

        if file_type == "pdf":
            from parsers.pdf_parser import PdfParser
            parser = PdfParser()
        elif file_type == "markdown":
            from parsers.markdown_parser import MarkdownParser
            parser = MarkdownParser()
        elif file_type == "plaintext":
            from parsers.plaintext_parser import PlaintextParser
            parser = PlaintextParser()
        else:
            return {"error": f"Unknown file type: {file_type}", "file_path": file_path}

        if not parser.can_parse(file_path):
            return {"error": f"Parser cannot handle: {file_path}", "file_path": file_path}

        result = parser.parse(file_path)

        if not result.segments:
            return {
                "file_path": file_path,
                "title": result.title,
                "segments": [],
                "metadata": result.metadata,
                "author": result.author,
                "source_type": result.source_type,
                "skipped": True,
                "reason": "no_segments",
            }

        # Apply chunking (PDF parser already chunks internally)
        if file_type == "pdf":
            chunked = result.segments
        else:
            chunked = chunk_segments(result.segments)

        # Serialize segments
        seg_dicts = []
        for seg in chunked:
            seg_dicts.append({
                "text": seg.text,
                "segment_type": seg.segment_type,
                "ordinal": seg.ordinal,
                "depth": seg.depth,
                "title_or_heading": seg.title_or_heading,
                "char_start": seg.char_start,
                "char_end": seg.char_end,
                "page_start": seg.page_start,
                "page_end": seg.page_end,
                "line_start": seg.line_start,
                "line_end": seg.line_end,
                "token_count": len(seg.text) // 4,  # rough estimate
                "metadata": seg.metadata or {},
            })

        return {
            "file_path": file_path,
            "title": result.title,
            "segments": seg_dicts,
            "metadata": result.metadata,
            "author": result.author,
            "source_type": result.source_type,
        }

    except Exception as e:
        return {"error": str(e), "file_path": file_path}


def store_parsed_result(engine, result: dict, tenant_id: str, checksum: str, file_path: str):
    """Store a parsed result into the database."""
    from sqlalchemy import text

    source_id = f"SRC_PDF_{hashlib.md5(file_path.encode()).hexdigest()[:16]}"
    file_size = os.path.getsize(file_path)

    with engine.connect() as conn:
        # Insert source
        conn.execute(
            text(
                "INSERT INTO sources (source_id, owner_tenant_id, source_type, title, "
                "author_or_origin, raw_file_path, checksum, file_size_bytes, mime_type, "
                "parse_status, parser_version, metadata_json) "
                "VALUES (:sid, :tid, :stype, :title, :author, :path, :cksum, :fsize, "
                ":mime, 'parsed', 'pdf_parser_v1', :meta) "
                "ON CONFLICT (source_id) DO NOTHING"
            ),
            {
                "sid": source_id,
                "tid": tenant_id,
                "stype": result.get("source_type", "pdf"),
                "title": result.get("title", os.path.basename(file_path)),
                "author": result.get("author"),
                "path": file_path,
                "cksum": checksum,
                "fsize": file_size,
                "mime": "application/pdf",
                "meta": __import__("json").dumps(result.get("metadata", {})),
            },
        )

        # Insert segments
        seg_count = 0
        for seg in result.get("segments", []):
            seg_id = f"SEG_{source_id}_{seg['ordinal']}"
            try:
                conn.execute(
                    text(
                        "INSERT INTO segments (segment_id, source_id, owner_tenant_id, "
                        "segment_type, ordinal, depth, title_or_heading, text, token_count, "
                        "char_start, char_end, page_start, page_end, line_start, line_end) "
                        "VALUES (:sid, :src, :tid, :stype, :ord, :depth, :heading, :txt, "
                        ":tc, :cs, :ce, :ps, :pe, :ls, :le) "
                        "ON CONFLICT (segment_id) DO NOTHING"
                    ),
                    {
                        "sid": seg_id,
                        "src": source_id,
                        "tid": tenant_id,
                        "stype": seg["segment_type"],
                        "ord": seg["ordinal"],
                        "depth": seg["depth"],
                        "heading": seg.get("title_or_heading"),
                        "txt": seg["text"],
                        "tc": seg.get("token_count", len(seg["text"]) // 4),
                        "cs": seg.get("char_start"),
                        "ce": seg.get("char_end"),
                        "ps": seg.get("page_start"),
                        "pe": seg.get("page_end"),
                        "ls": seg.get("line_start"),
                        "le": seg.get("line_end"),
                    },
                )
                seg_count += 1
            except Exception as e:
                logger.warning("Failed to insert segment %s: %s", seg_id, e)

        conn.commit()

    return source_id, seg_count


def collect_files(path: str, file_type: str, recursive: bool = True) -> list[str]:
    """Collect all files of the given type from a directory."""
    extensions = {
        "pdf": [".pdf"],
        "markdown": [".md", ".markdown"],
        "plaintext": [".txt", ".text", ".log"],
    }
    exts = extensions.get(file_type, [f".{file_type}"])

    files = []
    if os.path.isfile(path):
        files.append(path)
    elif os.path.isdir(path):
        if recursive:
            for root, _dirs, filenames in os.walk(path):
                for fname in sorted(filenames):
                    if any(fname.lower().endswith(ext) for ext in exts):
                        files.append(os.path.join(root, fname))
        else:
            for fname in sorted(os.listdir(path)):
                if any(fname.lower().endswith(ext) for ext in exts):
                    files.append(os.path.join(path, fname))
    else:
        logger.error("Path does not exist: %s", path)

    return files


def main():
    parser = argparse.ArgumentParser(description="Bulk ingest files into a GAMI tenant")
    parser.add_argument("--path", required=True, help="Directory or file to ingest")
    parser.add_argument("--tenant", required=True, help="Tenant ID to ingest into")
    parser.add_argument("--tenant-name", default=None, help="Display name for new tenant")
    parser.add_argument("--type", dest="file_type", default="pdf", choices=["pdf", "markdown", "plaintext"],
                        help="File type to process")
    parser.add_argument("--workers", type=int, default=2, help="Parallel workers for parsing")
    parser.add_argument("--no-recursive", action="store_true", help="Don't recurse into subdirectories")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be ingested without doing it")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of files to process (0 = all)")
    args = parser.parse_args()

    tenant_id = args.tenant
    display_name = args.tenant_name or f"{tenant_id.replace('-', ' ').title()} Library"

    # Collect files
    logger.info("Scanning %s for %s files...", args.path, args.file_type)
    files = collect_files(args.path, args.file_type, recursive=not args.no_recursive)
    logger.info("Found %d %s files", len(files), args.file_type)

    if not files:
        logger.error("No files found to ingest")
        return

    if args.limit > 0:
        files = files[:args.limit]
        logger.info("Limited to %d files", len(files))

    # Setup database
    engine = get_sync_engine()

    # Ensure tenant exists
    ensure_tenant(engine, tenant_id, display_name)

    # Get existing checksums for dedup
    logger.info("Loading existing checksums for tenant '%s'...", tenant_id)
    existing_checksums = get_existing_checksums(engine, tenant_id)
    logger.info("Found %d already-ingested sources", len(existing_checksums))

    # Compute checksums and filter already-ingested files
    logger.info("Computing checksums...")
    files_to_ingest = []
    skipped_dedup = 0
    for fpath in files:
        checksum = compute_file_checksum(fpath)
        if checksum in existing_checksums:
            skipped_dedup += 1
            continue
        files_to_ingest.append((fpath, checksum))

    logger.info(
        "%d files to ingest (%d skipped as already ingested)",
        len(files_to_ingest), skipped_dedup,
    )

    if not files_to_ingest:
        logger.info("Nothing to ingest — all files already present")
        engine.dispose()
        return

    if args.dry_run:
        for fpath, cksum in files_to_ingest:
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            print(f"  {fpath} ({size_mb:.1f} MB) [{cksum[:20]}...]")
        logger.info("Dry run — %d files would be ingested", len(files_to_ingest))
        engine.dispose()
        return

    # Process files
    total = len(files_to_ingest)
    completed = 0
    total_segments = 0
    errors = 0
    skipped_empty = 0
    start_time = time.time()

    logger.info("Starting ingestion with %d workers...", args.workers)

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Submit parse jobs
        future_map = {}
        for fpath, checksum in files_to_ingest:
            future = executor.submit(parse_single_file, fpath, args.file_type)
            future_map[future] = (fpath, checksum)

        # Collect results and store
        for future in as_completed(future_map):
            fpath, checksum = future_map[future]
            completed += 1

            try:
                result = future.result()
            except Exception as e:
                errors += 1
                logger.error("[%d/%d] PARSE ERROR %s: %s", completed, total, os.path.basename(fpath), e)
                continue

            if "error" in result:
                errors += 1
                logger.error("[%d/%d] ERROR %s: %s", completed, total, os.path.basename(fpath), result["error"])
                continue

            if result.get("skipped"):
                skipped_empty += 1
                logger.warning("[%d/%d] SKIP (empty) %s", completed, total, os.path.basename(fpath))
                continue

            # Store in DB
            try:
                source_id, seg_count = store_parsed_result(engine, result, tenant_id, checksum, fpath)
                total_segments += seg_count

                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                remaining = (total - completed) / rate if rate > 0 else 0

                logger.info(
                    "[%d/%d] OK %s — %d segments (%.1f files/min, ETA %.0fs)",
                    completed, total, os.path.basename(fpath), seg_count,
                    rate * 60, remaining,
                )
            except Exception as e:
                errors += 1
                logger.error("[%d/%d] DB ERROR %s: %s", completed, total, os.path.basename(fpath), e)

    elapsed = time.time() - start_time
    engine.dispose()

    # Summary
    logger.info("=" * 60)
    logger.info("INGESTION COMPLETE")
    logger.info("  Tenant:         %s", tenant_id)
    logger.info("  Files processed: %d / %d", completed, total)
    logger.info("  Segments stored: %d", total_segments)
    logger.info("  Skipped (dedup): %d", skipped_dedup)
    logger.info("  Skipped (empty): %d", skipped_empty)
    logger.info("  Errors:          %d", errors)
    logger.info("  Time:            %.1fs (%.1f files/min)", elapsed, completed / elapsed * 60 if elapsed > 0 else 0)
    logger.info("=" * 60)
    logger.info(
        "Next step: embed segments with:\n"
        "  python /opt/gami/scripts/embed_tenant.py --tenant %s --batch-size 50",
        tenant_id,
    )


if __name__ == "__main__":
    main()
