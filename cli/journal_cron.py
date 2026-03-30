#!/usr/bin/env python3
"""GAMI Journal Cron — ingest recently modified Claude Code JSONL sessions.

Scans for JSONL files modified in the last 35 minutes and ingests them
via the GAMI API. Intended to run every 30 minutes via cron as a fallback
for the Claude Code hooks.

Dedup is handled by the API's checksum-based source dedup: if the file
content changes (session grows), it gets re-ingested as a new source,
which is acceptable for journal capture.
"""
import glob
import os
import sys
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("gami.journal.cron")

GAMI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if GAMI_ROOT not in sys.path:
    sys.path.insert(0, GAMI_ROOT)

JSONL_GLOB = "/home/ai/.claude/projects/-home-ai/*.jsonl"
CUTOFF_SECONDS = 2100  # 35 minutes
TENANT_ID = "claude-opus"


def main():
    cutoff = time.time() - CUTOFF_SECONDS
    files = [f for f in glob.glob(JSONL_GLOB) if os.path.getmtime(f) > cutoff]

    if not files:
        logger.info("No recently modified JSONL files found.")
        return

    logger.info("Found %d recently modified JSONL files.", len(files))

    try:
        from parsers.conversation_parser import ConversationParser
        # Inline implementation
        from api.services.segment_service import store_segments_sync
    except ImportError as e:
        logger.error("Failed to import GAMI modules: %s", e)
        sys.exit(1)

    ingested = 0
    for f in files:
        try:
            parser = ConversationParser()
            result = parser.parse(f)
            sid = register_source_sync(
                TENANT_ID, f, "conversation_session", os.path.basename(f)
            )
            store_segments_sync(sid, TENANT_ID, result.segments)
            ingested += 1
            logger.info("Ingested: %s (source_id=%s, segments=%d)",
                        os.path.basename(f), sid, len(result.segments))
        except Exception as exc:
            logger.warning("Failed to ingest %s: %s", os.path.basename(f), exc)

    logger.info("Cron complete: %d/%d files ingested.", ingested, len(files))


if __name__ == "__main__":
    main()
