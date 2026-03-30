#!/usr/bin/env bash
# GAMI Segment Processor — runs after session ingestion
# Spawns a Claude Code Haiku agent to extract entities from new segments.
# Uses OAuth billing (Claude Code subscription), no API key needed.
# Falls back gracefully if claude CLI isn't available.
set -euo pipefail

LOGFILE="/tmp/gami-process-segments.log"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
GAMI_API="${GAMI_API:-http://localhost:9090}"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') [PROCESS] $*" >> "$LOGFILE"; }

log "Starting segment processing..."

# ---------------------------------------------------------------
# Step 1: Ensure Ollama is running for embeddings
# ---------------------------------------------------------------
if ! curl -sf "${OLLAMA_URL}/api/tags" > /dev/null 2>&1; then
    log "Ollama not responding, attempting to start..."
    if command -v ollama &>/dev/null; then
        nohup ollama serve >> /tmp/ollama-autostart.log 2>&1 &
        for i in $(seq 1 15); do
            if curl -sf "${OLLAMA_URL}/api/tags" > /dev/null 2>&1; then
                log "Ollama started successfully"
                break
            fi
            sleep 2
        done
        # Ensure embedding model is available
        if ! ollama list 2>/dev/null | grep -q nomic-embed-text; then
            log "Pulling nomic-embed-text model..."
            ollama pull nomic-embed-text >> "$LOGFILE" 2>&1 || true
        fi
    else
        log "WARNING: Ollama not installed. Embeddings will be skipped."
        log "Install: curl -fsSL https://ollama.com/install.sh | sh"
    fi
else
    log "Ollama is running"
fi

# ---------------------------------------------------------------
# Step 2: Check GAMI API is healthy
# ---------------------------------------------------------------
if ! curl -sf "${GAMI_API}/health" > /dev/null 2>&1; then
    log "GAMI API not responding, skipping processing"
    exit 0
fi

# ---------------------------------------------------------------
# Step 3: Check if claude CLI is available
# ---------------------------------------------------------------
if ! command -v claude &>/dev/null; then
    log "Claude CLI not found, skipping agent extraction"
    exit 0
fi

# ---------------------------------------------------------------
# Step 4: Run Claude Code agent for entity extraction
# The agent uses Bash+curl to talk to GAMI API (no MCP needed)
# ---------------------------------------------------------------
log "Spawning Haiku agent for entity extraction..."

claude -p 'You are a GAMI knowledge extraction agent. Extract entities from unprocessed text segments using the GAMI HTTP API at http://localhost:9090.

Steps:
1. Get unprocessed segments:
   curl -s "http://localhost:9090/api/v1/segments/unprocessed?limit=10&tenant_id=claude-opus"
   If that endpoint does not exist, use this SQL via the helper:
   cd /opt/gami && PYTHONPATH=/opt/gami python3 -c "
   from sqlalchemy import create_engine, text
   from api.config import settings
   import json
   engine = create_engine(settings.DATABASE_URL_SYNC)
   with engine.connect() as conn:
       rows = conn.execute(text(\"SELECT segment_id, left(text, 2000) as text, source_id FROM segments WHERE length(text) BETWEEN 200 AND 3000 AND owner_tenant_id = '\''claude-opus'\'' AND segment_type NOT IN ('\''tool_call'\'', '\''tool_result'\'', '\''chunk'\'') AND NOT EXISTS (SELECT 1 FROM provenance p WHERE p.segment_id = segments.segment_id) ORDER BY created_at DESC LIMIT 10\")).fetchall()
       print(json.dumps([{\"segment_id\": r[0], \"text\": r[1], \"source_id\": r[2]} for r in rows]))
   "

2. For EACH segment, analyze its text and identify entities (people, infrastructure, services, technologies, credentials, concepts).

3. Store each segments entities by running:
   cd /opt/gami && PYTHONPATH=/opt/gami python3 -c "
   import hashlib, json
   from sqlalchemy import create_engine, text
   from api.config import settings
   engine = create_engine(settings.DATABASE_URL_SYNC)
   entities = <YOUR_JSON_ARRAY>  # [{\"name\":\"...\",\"type\":\"...\",\"description\":\"...\"}]
   segment_id = \"<SEGMENT_ID>\"
   source_id = \"<SOURCE_ID>\"
   with engine.connect() as conn:
       for e in entities:
           eid = \"ENT_\" + e[\"type\"][:10] + \"_\" + e[\"name\"][:20].replace(\" \",\"_\") + \"_\" + hashlib.md5(e[\"name\"].encode()).hexdigest()[:8]
           conn.execute(text(\"INSERT INTO entities (entity_id, owner_tenant_id, entity_type, canonical_name, description, status, first_seen_at, last_seen_at, source_count, mention_count) VALUES (:eid, '\''claude-opus'\'', :etype, :name, :desc, '\''active'\'', NOW(), NOW(), 1, 1) ON CONFLICT (entity_id) DO UPDATE SET mention_count = entities.mention_count + 1, last_seen_at = NOW()\"), {\"eid\": eid, \"etype\": e[\"type\"], \"name\": e[\"name\"], \"desc\": e.get(\"description\",\"\")})
           conn.execute(text(\"INSERT INTO provenance (provenance_id, target_type, target_id, source_id, segment_id, extraction_method, extractor_version, confidence) VALUES (:pid, '\''entity'\'', :eid, :src, :seg, '\''agent_haiku'\'', '\''v1'\'', 0.85) ON CONFLICT DO NOTHING\"), {\"pid\": \"PROV_\" + eid[:20] + \"_\" + segment_id[:20], \"eid\": eid, \"src\": source_id, \"seg\": segment_id})
       conn.commit()
       print(f\"Stored {len(entities)} entities for {segment_id}\")
   "

Process up to 10 segments. Be thorough — extract ALL meaningful entities. Skip segments that are just code output or logs with no meaningful entities. Report what you processed.' \
    --model haiku \
    --allowedTools "Bash,Read" \
    >> "$LOGFILE" 2>&1 || log "Agent exited with code $?"

log "Segment processing complete"
exit 0
