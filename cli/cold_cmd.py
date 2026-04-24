#!/usr/bin/env python3
"""Cold Storage CLI for GAMI.

Usage:
    python -m cli.cold_cmd stats
    python -m cli.cold_cmd search <query> [--type <type>]
    python -m cli.cold_cmd archive <type> <id> --reason <reason>
    python -m cli.cold_cmd restore <type> <id>
    python -m cli.cold_cmd archive-stale <type> --days <N> [--tenant <id>] [--limit <N>]
"""
import argparse
import json
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("gami.cli.cold")

# Ensure GAMI root is in path
GAMI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if GAMI_ROOT not in sys.path:
    sys.path.insert(0, GAMI_ROOT)


def cmd_stats(args):
    """Show cold storage statistics."""
    from storage.cold.cold_index import get_stats

    stats = get_stats()

    print("\n=== GAMI Cold Storage Statistics ===\n")
    print(f"Total archived objects: {stats['total_objects']}")
    print(f"Disk usage:            {_format_bytes(stats['disk_usage_bytes'])}")
    print(f"Original data size:    {_format_bytes(stats['total_original_bytes'])}")
    print(f"Compressed size:       {_format_bytes(stats['total_compressed_bytes'])}")

    if stats["total_original_bytes"] and stats["total_compressed_bytes"]:
        ratio = stats["total_compressed_bytes"] / stats["total_original_bytes"]
        print(f"Compression ratio:     {ratio:.2%}")

    print(f"\nIndex path: {stats['index_path']}")

    if stats["by_type"]:
        print("\n--- By Object Type ---")
        for otype, info in sorted(stats["by_type"].items()):
            print(
                f"  {otype:20s}  {info['count']:>6d} objects  "
                f"{_format_bytes(info['compressed_bytes']):>10s} compressed"
            )

    if stats["by_tenant"]:
        print("\n--- By Tenant ---")
        for tenant, count in sorted(stats["by_tenant"].items()):
            print(f"  {tenant:30s}  {count:>6d} objects")

    print()


def cmd_search(args):
    """Search cold storage."""
    from storage.cold.cold_index import search_cold

    results = search_cold(
        query=args.query,
        object_type=args.type,
        tenant_id=args.tenant,
        limit=args.limit,
    )

    if not results:
        print("No results found.")
        return

    print(f"\nFound {len(results)} result(s):\n")

    for i, r in enumerate(results, 1):
        print(f"  [{i}] {r['object_type']}/{r['object_id']}")
        if r["name"]:
            print(f"      Name: {r['name']}")
        if r["description"]:
            print(f"      Desc: {r['description'][:100]}")
        if r["summary"]:
            print(f"      Summary: {r['summary'][:100]}")
        print(f"      Tenant: {r['owner_tenant_id']}")
        print(f"      Archived: {r['archived_at']}")
        print(f"      Reason: {r['archive_reason']}")
        print(
            f"      Size: {_format_bytes(r['compressed_size_bytes'] or 0)} "
            f"(was {_format_bytes(r['original_size_bytes'] or 0)})"
        )
        print()


def cmd_archive(args):
    """Archive an object to cold storage."""
    from api.services.db import get_sync_db
    from storage.cold.archiver import archive_object
    from storage.cold.cold_index import index_object

    db_gen = get_sync_db()
    db = next(db_gen)

    try:
        result = archive_object(
            db=db,
            object_type=args.type,
            object_id=args.id,
            reason=args.reason,
            archived_by="cli",
        )

        if result["status"] == "archived":
            # Also index in SQLite for searchability
            index_object(
                object_type=args.type,
                object_id=args.id,
                cold_path=result["cold_path"],
                archive_reason=args.reason,
                original_size_bytes=result.get("original_size_bytes"),
                compressed_size_bytes=result.get("compressed_size_bytes"),
            )
            print(f"Archived {args.type}/{args.id}")
            print(f"  Path: {result['cold_path']}")
            print(
                f"  Size: {_format_bytes(result['compressed_size_bytes'])} "
                f"(was {_format_bytes(result['original_size_bytes'])})"
            )
        else:
            print(f"Skip: {result.get('status', 'unknown')}")

    finally:
        try:
            next(db_gen)
        except StopIteration:
            pass


def cmd_restore(args):
    """Restore an object from cold storage."""
    from api.services.db import get_sync_db
    from storage.cold.cold_index import remove_from_index
    from storage.cold.restorer import restore_object

    db_gen = get_sync_db()
    db = next(db_gen)

    try:
        result = restore_object(
            db=db,
            object_type=args.type,
            object_id=args.id,
        )

        if result["status"] == "restored":
            remove_from_index(args.type, args.id)
            print(f"Restored {args.type}/{args.id} to tier '{result['restored_tier']}'")
        else:
            print(f"Status: {result.get('status', 'unknown')}")

    finally:
        try:
            next(db_gen)
        except StopIteration:
            pass


def cmd_archive_stale(args):
    """Archive stale objects in bulk."""
    from api.services.db import get_sync_db
    from storage.cold.archiver import archive_stale

    db_gen = get_sync_db()
    db = next(db_gen)

    try:
        result = archive_stale(
            db=db,
            object_type=args.type,
            days_stale=args.days,
            tenant_id=args.tenant,
            limit=args.limit,
        )

        print(f"Archive stale {args.type} (>{args.days} days):")
        print(f"  Archived: {result['archived']}")
        print(f"  Skipped:  {result['skipped']}")
        print(f"  Failed:   {result['failed']}")

        if result.get("errors"):
            print("  Errors:")
            for err in result["errors"][:10]:
                print(f"    {err['object_id']}: {err['error']}")

    finally:
        try:
            next(db_gen)
        except StopIteration:
            pass


def _format_bytes(n: int) -> str:
    """Human-readable byte count."""
    if n is None:
        return "0 B"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def main():
    parser = argparse.ArgumentParser(
        description="GAMI Cold Storage CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # stats
    subparsers.add_parser("stats", help="Show cold storage statistics")

    # search
    search_p = subparsers.add_parser("search", help="Search cold storage")
    search_p.add_argument("query", help="Search query")
    search_p.add_argument("--type", help="Filter by object type")
    search_p.add_argument("--tenant", help="Filter by tenant ID")
    search_p.add_argument("--limit", type=int, default=20, help="Max results")

    # archive
    archive_p = subparsers.add_parser("archive", help="Archive an object")
    archive_p.add_argument("type", help="Object type (entity, claim, segment, ...)")
    archive_p.add_argument("id", help="Object ID")
    archive_p.add_argument("--reason", required=True, help="Reason for archiving")

    # restore
    restore_p = subparsers.add_parser("restore", help="Restore from cold storage")
    restore_p.add_argument("type", help="Object type")
    restore_p.add_argument("id", help="Object ID")

    # archive-stale
    stale_p = subparsers.add_parser("archive-stale", help="Archive stale objects")
    stale_p.add_argument("type", help="Object type")
    stale_p.add_argument("--days", type=int, default=90, help="Days stale threshold")
    stale_p.add_argument("--tenant", help="Tenant ID filter")
    stale_p.add_argument("--limit", type=int, default=100, help="Max objects to archive")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    commands = {
        "stats": cmd_stats,
        "search": cmd_search,
        "archive": cmd_archive,
        "restore": cmd_restore,
        "archive-stale": cmd_archive_stale,
    }

    cmd_func = commands.get(args.command)
    if cmd_func:
        cmd_func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
