#!/usr/bin/env python3
"""GAMI Journal CLI — ingest and manage AI session journals.

Reads Claude Code JSONL session files and posts them to GAMI for
searchable, structured storage with full provenance.

Usage:
    gami-journal save --session <id> --session-file <path> --tenant <id>
    gami-journal save --session <id> --session-file <path> --trigger milestone
    gami-journal list --tenant <id> --last 10
    gami-journal search <query> --tenant <id>
    gami-journal export --tenant <id> --format markdown --output journal.md
"""
import argparse
import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from typing import Optional

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("gami.journal")

GAMI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if GAMI_ROOT not in sys.path:
    sys.path.insert(0, GAMI_ROOT)

DEFAULT_API = "http://127.0.0.1:9090"
DEFAULT_TENANT = "claude-opus"


def cmd_save(args):
    """Save a session journal to GAMI."""
    session_file = args.session_file
    session_id = args.session or f"session_{uuid.uuid4().hex[:12]}"
    tenant_id = args.tenant or DEFAULT_TENANT
    trigger = args.trigger or "manual"
    api_url = args.api or DEFAULT_API

    if not os.path.isfile(session_file):
        logger.error("Session file not found: %s", session_file)
        sys.exit(1)

    # Read the session file (JSONL format)
    entries = []
    with open(session_file, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                entries.append(entry)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed line %d: %s", line_num, exc)

    if not entries:
        logger.error("No valid entries found in %s", session_file)
        sys.exit(1)

    logger.info(
        "Read %d entries from %s (session: %s, trigger: %s)",
        len(entries), session_file, session_id, trigger,
    )

    # Build a consolidated document for ingestion
    doc_parts = []
    doc_parts.append(f"# Session Journal: {session_id}")
    doc_parts.append(f"**Trigger**: {trigger}")
    doc_parts.append(f"**Tenant**: {tenant_id}")
    doc_parts.append(f"**Source File**: {session_file}")
    doc_parts.append(f"**Entries**: {len(entries)}")
    doc_parts.append(f"**Captured**: {datetime.now(timezone.utc).isoformat()}")
    doc_parts.append("")

    # Extract conversation turns
    for i, entry in enumerate(entries):
        role = entry.get("role", entry.get("type", "unknown"))
        content = ""

        if isinstance(entry.get("content"), str):
            content = entry["content"]
        elif isinstance(entry.get("content"), list):
            # Handle structured content (Claude API format)
            text_parts = []
            for block in entry["content"]:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "tool_use":
                        text_parts.append(
                            f"[Tool: {block.get('name', 'unknown')}]"
                        )
                    elif block.get("type") == "tool_result":
                        text_parts.append(
                            f"[Tool Result: {str(block.get('content', ''))[:200]}]"
                        )
                elif isinstance(block, str):
                    text_parts.append(block)
            content = "\n".join(text_parts)
        elif isinstance(entry.get("message"), str):
            content = entry["message"]

        if not content:
            continue

        timestamp = entry.get("timestamp", entry.get("created_at", ""))
        header = f"## Turn {i + 1} [{role}]"
        if timestamp:
            header += f" ({timestamp})"

        doc_parts.append(header)
        doc_parts.append(content[:10000])  # Cap individual entries
        doc_parts.append("")

    document = "\n".join(doc_parts)

    # Write temp file for ingestion
    temp_dir = os.path.join(GAMI_ROOT, "storage", "objects", "uploads")
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"journal_{session_id}.md")

    with open(temp_path, "w") as f:
        f.write(document)

    # Ingest via API
    try:
        resp = httpx.post(
            f"{api_url}/api/v1/ingest/source",
            data={
                "file_path": temp_path,
                "source_type": "conversation_session",
                "tenant_id": tenant_id,
                "title": f"Journal: {session_id}",
                "metadata_json": json.dumps({
                    "session_id": session_id,
                    "trigger": trigger,
                    "entry_count": len(entries),
                    "source_file": session_file,
                    "journal_type": "ai_session",
                }),
            },
            timeout=120.0,
        )
        resp.raise_for_status()
        result = resp.json()

        if result.get("status") == "completed":
            logger.info(
                "Journal saved: source=%s, segments=%d",
                result.get("source_id"), result.get("segments_created", 0),
            )
            print(f"Session {session_id} saved successfully.")
            print(f"  Source ID: {result.get('source_id')}")
            print(f"  Segments:  {result.get('segments_created', 0)}")
        elif result.get("status") == "duplicate":
            logger.info("Journal already ingested: %s", result.get("source_id"))
            print(f"Session already saved (source: {result.get('source_id')})")
        else:
            logger.warning("Unexpected result: %s", result)
            print(f"Result: {json.dumps(result, indent=2)}")

    except httpx.HTTPError as exc:
        logger.error("API request failed: %s", exc)
        sys.exit(1)


def cmd_list(args):
    """List recent journal entries."""
    tenant_id = args.tenant or DEFAULT_TENANT
    last_n = args.last or 10
    api_url = args.api or DEFAULT_API

    try:
        resp = httpx.get(
            f"{api_url}/api/v1/ingest/sources",
            params={
                "tenant_id": tenant_id,
                "source_type": "conversation_session",
                "limit": last_n,
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()

        sources = data.get("sources", [])
        if not sources:
            print("No journal entries found.")
            return

        print(f"\nJournal entries for tenant '{tenant_id}' (last {last_n}):\n")
        for i, src in enumerate(sources, 1):
            title = src.get("title", "Untitled")
            sid = src.get("source_id", "?")
            ingested = src.get("ingested_at", "?")
            meta = src.get("metadata_json", {})
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except json.JSONDecodeError:
                    meta = {}

            session_id = meta.get("session_id", "?")
            trigger = meta.get("trigger", "?")
            entries = meta.get("entry_count", "?")

            print(f"  [{i}] {title}")
            print(f"      ID: {sid}")
            print(f"      Session: {session_id}, Trigger: {trigger}, Entries: {entries}")
            print(f"      Ingested: {ingested}")
            print()

    except httpx.HTTPError as exc:
        logger.error("API request failed: %s", exc)
        sys.exit(1)


def cmd_search(args):
    """Search journal entries."""
    query = args.query
    tenant_id = args.tenant or DEFAULT_TENANT
    api_url = args.api or DEFAULT_API

    try:
        resp = httpx.post(
            f"{api_url}/api/v1/memory/search",
            json={
                "query": query,
                "tenant_id": tenant_id,
                "limit": args.limit or 10,
                "search_mode": "hybrid",
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()

        results = data.get("results", [])
        if not results:
            print("No results found.")
            return

        print(f"\nSearch results for '{query}' (tenant: {tenant_id}):\n")
        for i, r in enumerate(results, 1):
            score = r.get("combined_score", 0)
            text_preview = r.get("text", "")[:200]
            seg_type = r.get("segment_type", "?")
            heading = r.get("title_or_heading", "")

            print(f"  [{i}] Score: {score:.3f} | {seg_type}")
            if heading:
                print(f"      Heading: {heading}")
            print(f"      {text_preview}")
            print()

    except httpx.HTTPError as exc:
        logger.error("API request failed: %s", exc)
        sys.exit(1)


def cmd_export(args):
    """Export journal entries to a file."""
    tenant_id = args.tenant or DEFAULT_TENANT
    output_path = args.output or f"gami_journal_{tenant_id}.md"
    fmt = args.format or "markdown"
    api_url = args.api or DEFAULT_API

    try:
        # Fetch all journal sources
        resp = httpx.get(
            f"{api_url}/api/v1/ingest/sources",
            params={
                "tenant_id": tenant_id,
                "source_type": "conversation_session",
                "limit": 1000,
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        sources = resp.json().get("sources", [])

        if not sources:
            print("No journal entries to export.")
            return

        output_parts = []
        output_parts.append(f"# GAMI Journal Export — {tenant_id}")
        output_parts.append(f"Exported: {datetime.now(timezone.utc).isoformat()}")
        output_parts.append(f"Total sessions: {len(sources)}")
        output_parts.append("")

        for src in sources:
            source_id = src.get("source_id")
            title = src.get("title", "Untitled")
            output_parts.append(f"## {title}")
            output_parts.append(f"Source: {source_id}")
            output_parts.append("")

            # Fetch segments for this source
            seg_resp = httpx.get(
                f"{api_url}/api/v1/ingest/sources/{source_id}/segments",
                timeout=30.0,
            )
            if seg_resp.status_code == 200:
                segments = seg_resp.json().get("segments", [])
                for seg in segments:
                    text = seg.get("text", "")
                    heading = seg.get("title_or_heading", "")
                    if heading:
                        output_parts.append(f"### {heading}")
                    output_parts.append(text)
                    output_parts.append("")

            output_parts.append("---")
            output_parts.append("")

        document = "\n".join(output_parts)

        with open(output_path, "w") as f:
            f.write(document)

        print(f"Exported {len(sources)} sessions to {output_path}")
        print(f"  Format: {fmt}")
        print(f"  Size: {len(document)} chars")

    except httpx.HTTPError as exc:
        logger.error("API request failed: %s", exc)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="GAMI AI Journal CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    gami-journal save --session abc123 --session-file session.jsonl --tenant claude-opus
    gami-journal list --tenant claude-opus --last 10
    gami-journal search "database migration" --tenant claude-opus
    gami-journal export --tenant claude-opus --format markdown --output journal.md
        """,
    )
    parser.add_argument(
        "--api", default=DEFAULT_API,
        help=f"GAMI API URL (default: {DEFAULT_API})",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command")

    # save
    save_p = subparsers.add_parser("save", help="Save a session journal")
    save_p.add_argument("--session", help="Session ID")
    save_p.add_argument(
        "--session-file", required=True,
        help="Path to JSONL session file",
    )
    save_p.add_argument("--trigger", default="manual",
                        help="Trigger type (manual, milestone, error, periodic)")
    save_p.add_argument("--tenant", default=DEFAULT_TENANT, help="Tenant ID")

    # list
    list_p = subparsers.add_parser("list", help="List recent journals")
    list_p.add_argument("--tenant", default=DEFAULT_TENANT, help="Tenant ID")
    list_p.add_argument("--last", type=int, default=10, help="Number of entries")

    # search
    search_p = subparsers.add_parser("search", help="Search journal entries")
    search_p.add_argument("query", help="Search query")
    search_p.add_argument("--tenant", default=DEFAULT_TENANT, help="Tenant ID")
    search_p.add_argument("--limit", type=int, default=10, help="Max results")

    # export
    export_p = subparsers.add_parser("export", help="Export journals")
    export_p.add_argument("--tenant", default=DEFAULT_TENANT, help="Tenant ID")
    export_p.add_argument(
        "--format", choices=["markdown", "json"], default="markdown",
        help="Export format",
    )
    export_p.add_argument("--output", help="Output file path")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    commands = {
        "save": cmd_save,
        "list": cmd_list,
        "search": cmd_search,
        "export": cmd_export,
    }

    cmd_func = commands.get(args.command)
    if cmd_func:
        cmd_func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
