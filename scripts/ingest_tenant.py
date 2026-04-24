#!/usr/bin/env python3
"""Generic tenant data ingestion script for GAMI.

This script ingests files into a GAMI tenant with full support for:
- Product manifold embeddings (H^32 × S^16 × E^64)
- Multiple file formats (markdown, text, JSON, PDF, etc.)
- Batch processing with progress tracking
- Real-time embedding and manifold coordinate generation

Usage:
    # Ingest a single file
    python scripts/ingest_tenant.py --tenant my-tenant --file /path/to/file.md

    # Ingest a directory
    python scripts/ingest_tenant.py --tenant my-tenant --directory /path/to/docs/

    # Ingest with specific file types
    python scripts/ingest_tenant.py --tenant my-tenant --directory /path/to/docs/ --extensions .md,.txt

    # Dry run (show what would be ingested)
    python scripts/ingest_tenant.py --tenant my-tenant --directory /path/to/docs/ --dry-run

Environment Variables:
    GAMI_DATABASE_URL: PostgreSQL connection string (required)
    GAMI_API_URL: GAMI API URL (default: http://127.0.0.1:9090)
    OLLAMA_URL: Ollama URL for embeddings (default: http://127.0.0.1:11434)

Examples:
    # Using environment variables
    export GAMI_DATABASE_URL="postgresql://user:pass@localhost:5432/gami"
    python scripts/ingest_tenant.py --tenant research --directory ./papers/

    # Create tenant first if needed
    python scripts/ingest_tenant.py --tenant new-project --create-tenant --file ./README.md
"""
import argparse
import asyncio
import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Add GAMI root to path
GAMI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if GAMI_ROOT not in sys.path:
    sys.path.insert(0, GAMI_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
log = logging.getLogger(__name__)

# Supported file extensions and their source types
FILE_TYPE_MAP = {
    '.md': 'markdown',
    '.markdown': 'markdown',
    '.txt': 'text',
    '.text': 'text',
    '.json': 'json',
    '.jsonl': 'jsonl',
    '.pdf': 'pdf',
    '.html': 'html',
    '.htm': 'html',
    '.rst': 'rst',
    '.org': 'org',
    '.xml': 'xml',
    '.csv': 'csv',
    '.py': 'code',
    '.js': 'code',
    '.ts': 'code',
    '.go': 'code',
    '.rs': 'code',
    '.java': 'code',
    '.c': 'code',
    '.cpp': 'code',
    '.h': 'code',
    '.sh': 'code',
    '.yaml': 'yaml',
    '.yml': 'yaml',
    '.toml': 'toml',
    '.ini': 'ini',
    '.cfg': 'ini',
    '.conf': 'ini',
}

DEFAULT_EXTENSIONS = ['.md', '.txt', '.json', '.pdf', '.html']


def get_file_hash(file_path: str) -> str:
    """Compute SHA-256 hash of file contents."""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_source_type(file_path: str) -> str:
    """Determine source type from file extension."""
    ext = Path(file_path).suffix.lower()
    return FILE_TYPE_MAP.get(ext, 'text')


def collect_files(
    directory: str,
    extensions: list[str],
    recursive: bool = True,
) -> list[str]:
    """Collect all files matching extensions from directory."""
    files = []
    path = Path(directory)

    if recursive:
        for ext in extensions:
            files.extend(path.rglob(f'*{ext}'))
    else:
        for ext in extensions:
            files.extend(path.glob(f'*{ext}'))

    return [str(f) for f in files if f.is_file()]


async def ensure_tenant_exists(tenant_id: str, create: bool = False) -> bool:
    """Check if tenant exists, optionally create it."""
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker

    db_url = os.getenv('GAMI_DATABASE_URL')
    if not db_url:
        log.error("GAMI_DATABASE_URL environment variable not set")
        return False

    # Convert to async URL if needed
    if db_url.startswith('postgresql://'):
        db_url = db_url.replace('postgresql://', 'postgresql+asyncpg://')

    engine = create_async_engine(db_url)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as db:
        result = await db.execute(
            text("SELECT tenant_id FROM tenants WHERE tenant_id = :tid"),
            {"tid": tenant_id}
        )
        exists = result.fetchone() is not None

        if not exists and create:
            log.info(f"Creating tenant: {tenant_id}")
            await db.execute(
                text("""
                    INSERT INTO tenants (tenant_id, display_name, created_at)
                    VALUES (:tid, :name, :now)
                """),
                {
                    "tid": tenant_id,
                    "name": tenant_id.replace('-', ' ').title(),
                    "now": datetime.now(timezone.utc),
                }
            )
            await db.commit()
            return True

        return exists


async def ingest_file(
    file_path: str,
    tenant_id: str,
    api_url: str,
    metadata: Optional[dict] = None,
) -> dict:
    """Ingest a single file via GAMI API."""
    import httpx

    source_type = get_source_type(file_path)
    title = Path(file_path).stem

    # Prepare metadata
    file_metadata = metadata or {}
    file_metadata.update({
        'original_path': file_path,
        'file_size': os.path.getsize(file_path),
        'ingested_at': datetime.now(timezone.utc).isoformat(),
    })

    async with httpx.AsyncClient(timeout=120.0) as client:
        with open(file_path, 'rb') as f:
            response = await client.post(
                f"{api_url}/api/v1/ingest/source",
                data={
                    'source_type': source_type,
                    'title': title,
                    'tenant_id': tenant_id,
                    'metadata_json': json.dumps(file_metadata),
                },
                files={'file': (Path(file_path).name, f)},
            )

        if response.status_code == 200:
            return response.json()
        else:
            return {
                'status': 'error',
                'error': response.text,
                'status_code': response.status_code,
            }


async def ingest_batch(
    files: list[str],
    tenant_id: str,
    api_url: str,
    metadata: Optional[dict] = None,
    concurrency: int = 4,
) -> dict:
    """Ingest multiple files with concurrency control."""
    from asyncio import Semaphore

    semaphore = Semaphore(concurrency)
    results = {
        'successful': 0,
        'failed': 0,
        'duplicates': 0,
        'segments_created': 0,
        'segments_embedded': 0,
        'manifold_computed': 0,
        'errors': [],
    }

    async def ingest_with_semaphore(file_path: str):
        async with semaphore:
            try:
                result = await ingest_file(file_path, tenant_id, api_url, metadata)

                if result.get('status') == 'completed':
                    results['successful'] += 1
                    results['segments_created'] += result.get('segments_created', 0)
                    results['segments_embedded'] += result.get('segments_embedded', 0)
                    results['manifold_computed'] += result.get('manifold_computed', 0)
                    log.info(f"Ingested: {file_path} ({result.get('segments_created', 0)} segments)")
                elif result.get('status') == 'duplicate':
                    results['duplicates'] += 1
                    log.info(f"Duplicate: {file_path}")
                else:
                    results['failed'] += 1
                    results['errors'].append({
                        'file': file_path,
                        'error': result.get('error', 'Unknown error'),
                    })
                    log.warning(f"Failed: {file_path} - {result.get('error', 'Unknown error')}")

            except Exception as e:
                results['failed'] += 1
                results['errors'].append({'file': file_path, 'error': str(e)})
                log.error(f"Exception ingesting {file_path}: {e}")

    # Process all files
    tasks = [ingest_with_semaphore(f) for f in files]
    await asyncio.gather(*tasks)

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Ingest files into a GAMI tenant',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments
    parser.add_argument(
        '--tenant', '-t',
        required=True,
        help='Tenant ID to ingest into',
    )

    # Source arguments (mutually exclusive)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        '--file', '-f',
        help='Single file to ingest',
    )
    source_group.add_argument(
        '--directory', '-d',
        help='Directory to ingest files from',
    )

    # Optional arguments
    parser.add_argument(
        '--extensions', '-e',
        default=','.join(DEFAULT_EXTENSIONS),
        help=f'Comma-separated file extensions to include (default: {",".join(DEFAULT_EXTENSIONS)})',
    )
    parser.add_argument(
        '--recursive', '-r',
        action='store_true',
        default=True,
        help='Recursively process subdirectories (default: True)',
    )
    parser.add_argument(
        '--no-recursive',
        action='store_false',
        dest='recursive',
        help='Do not process subdirectories',
    )
    parser.add_argument(
        '--concurrency', '-c',
        type=int,
        default=4,
        help='Number of concurrent ingestions (default: 4)',
    )
    parser.add_argument(
        '--api-url',
        default=os.getenv('GAMI_API_URL', 'http://127.0.0.1:9090'),
        help='GAMI API URL (default: http://127.0.0.1:9090)',
    )
    parser.add_argument(
        '--create-tenant',
        action='store_true',
        help='Create tenant if it does not exist',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be ingested without actually ingesting',
    )
    parser.add_argument(
        '--metadata',
        type=json.loads,
        default={},
        help='JSON metadata to attach to all ingested files',
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging',
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Collect files to ingest
    if args.file:
        if not os.path.isfile(args.file):
            log.error(f"File not found: {args.file}")
            sys.exit(1)
        files = [args.file]
    else:
        if not os.path.isdir(args.directory):
            log.error(f"Directory not found: {args.directory}")
            sys.exit(1)
        extensions = [e.strip() for e in args.extensions.split(',')]
        files = collect_files(args.directory, extensions, args.recursive)

    if not files:
        log.warning("No files found to ingest")
        sys.exit(0)

    log.info(f"Found {len(files)} files to ingest into tenant '{args.tenant}'")

    if args.dry_run:
        log.info("DRY RUN - Files that would be ingested:")
        for f in files[:50]:
            log.info(f"  {f}")
        if len(files) > 50:
            log.info(f"  ... and {len(files) - 50} more")
        sys.exit(0)

    # Check/create tenant
    async def run():
        # Verify tenant exists
        tenant_exists = await ensure_tenant_exists(args.tenant, args.create_tenant)
        if not tenant_exists:
            if args.create_tenant:
                log.error(f"Failed to create tenant: {args.tenant}")
            else:
                log.error(f"Tenant not found: {args.tenant}. Use --create-tenant to create it.")
            sys.exit(1)

        # Ingest files
        log.info(f"Starting ingestion with concurrency={args.concurrency}")
        results = await ingest_batch(
            files,
            args.tenant,
            args.api_url,
            args.metadata,
            args.concurrency,
        )

        # Summary
        log.info("=" * 60)
        log.info("INGESTION COMPLETE")
        log.info("=" * 60)
        log.info(f"Successful:        {results['successful']}")
        log.info(f"Duplicates:        {results['duplicates']}")
        log.info(f"Failed:            {results['failed']}")
        log.info(f"Segments created:  {results['segments_created']}")
        log.info(f"Segments embedded: {results['segments_embedded']}")
        log.info(f"Manifold computed: {results['manifold_computed']}")

        if results['errors']:
            log.warning(f"\nErrors ({len(results['errors'])}):")
            for err in results['errors'][:10]:
                log.warning(f"  {err['file']}: {err['error']}")
            if len(results['errors']) > 10:
                log.warning(f"  ... and {len(results['errors']) - 10} more errors")

        return results['failed'] == 0

    success = asyncio.run(run())
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
