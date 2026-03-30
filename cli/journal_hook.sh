#!/usr/bin/env bash
# GAMI Journal Hook for Claude Code
# Called by PreCompact and SessionEnd hooks.
# Reads hook input JSON from stdin, extracts transcript_path and session_id,
# then invokes gami_journal.py to ingest the session.
set -euo pipefail

# Read hook input from stdin
INPUT=$(cat)

# Extract session_id and transcript_path from hook input JSON
SESSION_ID=$(echo "$INPUT" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('session_id',''))" 2>/dev/null || echo "")
TRANSCRIPT=$(echo "$INPUT" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('transcript_path',''))" 2>/dev/null || echo "")

# Determine trigger type from hook event name
TRIGGER=$(echo "$INPUT" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('hook_event_name','manual'))" 2>/dev/null || echo "manual")

# Find the JSONL session file
JSONL_FILE=""
if [ -n "$SESSION_ID" ]; then
    # Look for the JSONL file matching this session
    CANDIDATE="/home/ai/.claude/projects/-home-ai/${SESSION_ID}.jsonl"
    if [ -f "$CANDIDATE" ]; then
        JSONL_FILE="$CANDIDATE"
    else
        # Search other project dirs
        for d in /home/ai/.claude/projects/*/; do
            if [ -f "${d}${SESSION_ID}.jsonl" ]; then
                JSONL_FILE="${d}${SESSION_ID}.jsonl"
                break
            fi
        done
    fi
fi

# Fallback: use transcript_path if it's a JSONL
if [ -z "$JSONL_FILE" ] && [ -n "$TRANSCRIPT" ] && [[ "$TRANSCRIPT" == *.jsonl ]]; then
    JSONL_FILE="$TRANSCRIPT"
fi

if [ -z "$JSONL_FILE" ] || [ ! -f "$JSONL_FILE" ]; then
    # Nothing to ingest — exit silently
    exit 0
fi

# Ingest via the journal CLI (async, don't block Claude Code)
export PYTHONPATH=/opt/gami
/usr/local/anaconda3/bin/python /opt/gami/cli/gami_journal.py save \
    --session "${SESSION_ID:-unknown}" \
    --session-file "$JSONL_FILE" \
    --trigger "$TRIGGER" \
    --tenant claude-opus \
    >> /tmp/gami-journal-hook.log 2>&1 || true

exit 0
