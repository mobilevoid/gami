# GAMI — Graph-Augmented Memory Interface

## What is this?

GAMI is a persistent memory system for Claude Code. It replaces the flat markdown memory files with a searchable, structured knowledge base that:

- **Remembers everything** across sessions — conversations, entities, credentials, infrastructure
- **Searches intelligently** — hybrid vector + keyword search across all your data
- **Extracts knowledge automatically** — entities, claims, relationships from every session
- **Works without GPU** — uses Haiku (OAuth) for extraction, Ollama CPU for embeddings
- **Auto-saves sessions** — hooks into compact/clear/exit to capture everything

## Quick Install (Claude Code)

Tell Claude Code:

> "Read /opt/gami/install/SETUP.md and follow the installation instructions to set up GAMI on this machine."

Or if you have the repo:

> "Run /opt/gami/install/install.sh and then configure the MCP server."

## What Gets Installed

| Component | Purpose | Resource |
|-----------|---------|----------|
| PostgreSQL 16 | Storage (segments, entities, claims, graph) | ~200MB RAM, ~1GB disk |
| pgvector | Vector similarity search | PG extension |
| Redis 7 | Cache, session state | ~100MB RAM |
| Ollama | CPU embeddings (nomic-embed-text, 274MB) | CPU only |
| GAMI API | FastAPI on port 9090 | ~100MB RAM |
| GAMI MCP | Claude Code tool integration | stdio process |

**Total: ~500MB RAM, ~2GB disk.** No GPU required.

## For Claude Code: Installation Steps

Run these steps in order. The install script handles most of it automatically.

### Step 1: Run the installer

```bash
bash /opt/gami/install/install.sh
```

This will:
- Install PostgreSQL 16, pgvector, Redis 7, Ollama
- Create the GAMI database and run migrations
- Install Python dependencies
- Pull the embedding model (nomic-embed-text)
- Start all services
- Run a health check

### Step 2: Register MCP server in Claude Code

```bash
claude mcp add gami \
  --transport stdio \
  --command /usr/local/anaconda3/bin/python \
  --args "-m" "mcp_tools.server" \
  --env "PYTHONPATH=/opt/gami" \
  -- --cwd /opt/gami
```

Or manually add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "gami": {
      "command": "python3",
      "args": ["-m", "mcp_tools.server"],
      "cwd": "/opt/gami",
      "env": {"PYTHONPATH": "/opt/gami"}
    }
  }
}
```

### Step 3: Set up session auto-save hooks

Add to `~/.claude/settings.json` under `"hooks"`:

```json
{
  "hooks": {
    "PreCompact": [{"hooks": [{"type": "command", "command": "/opt/gami/cli/journal_hook.sh", "timeout": 30, "async": true}]}],
    "SessionEnd": [{"hooks": [{"type": "command", "command": "/opt/gami/cli/journal_hook.sh", "timeout": 30, "async": true}]}],
    "PreClear": [{"hooks": [{"type": "command", "command": "/opt/gami/cli/journal_hook.sh", "timeout": 30, "async": true}]}]
  }
}
```

### Step 4: Verify

```bash
curl http://localhost:9090/health
```

Then in Claude Code, try: "Recall what you know about this system"

## Available MCP Tools (20)

After setup, Claude Code has these tools:

### Memory Operations
| Tool | Description |
|------|-------------|
| `memory_recall` | Search memories with token budget — the main retrieval tool |
| `memory_remember` | Store a new memory (fact, project info, credential) |
| `memory_forget` | Archive a memory (never deletes) |
| `memory_update` | Update an existing memory |
| `memory_search` | Raw hybrid search across all data |
| `memory_cite` | Get citations for a piece of evidence |
| `memory_verify` | Check if a claim is supported by stored evidence |
| `memory_context` | Get/set session working memory |

### Knowledge Management
| Tool | Description |
|------|-------------|
| `ingest_file` | Ingest a local file (markdown, text, etc.) into GAMI |
| `ingest_source` | Ingest from a URL or upload |
| `get_unprocessed_segments` | Get segments needing entity extraction |
| `store_extractions` | Store extracted entities from analysis |
| `graph_explore` | Explore entity relationships in the knowledge graph |
| `admin_stats` | Get system statistics |

### Dream Cycle (Background Knowledge Synthesis)
| Tool | Description |
|------|-------------|
| `dream_start` | Start dream cycle using local vLLM GPU (~500 seg/hr) |
| `dream_haiku` | Start dream cycle using Haiku/OAuth (~300 seg/hr, no GPU) |
| `dream_stop` | Stop the running dream cycle |
| `dream_status` | Check dream cycle progress |
| `run_haiku_extraction` | One-shot Haiku extraction with health checks |
| `review_proposals` | Review/approve/reject dream cycle proposals |

## How Claude Should Use GAMI

### At session start
1. Call `memory_recall` with a query about the current task to load relevant context
2. Call `review_proposals` with action="list" to check pending knowledge changes

### During conversation
- When learning new facts: call `memory_remember` to store them
- When asked about past work: call `memory_recall` to search
- When ingesting new files: call `ingest_file` then `dream_haiku` to process

### At session end (automatic)
- Hooks auto-save the session transcript
- Post-save Haiku agent auto-extracts entities from new segments

### For bulk processing
- With GPU: call `dream_start` for fast local processing
- Without GPU: call `dream_haiku` for Haiku-powered processing
- Either way: entities, summaries, and relationships get built automatically

## Configuration

### Environment variables (`/opt/gami/.env`)
```
DATABASE_URL=postgresql+asyncpg://gami:PASSWORD@localhost:5433/gami
DATABASE_URL_SYNC=postgresql://gami:PASSWORD@localhost:5433/gami
REDIS_URL=redis://localhost:6380/0
OLLAMA_URL=http://localhost:11434
VLLM_URL=http://localhost:8000/v1
EMBEDDING_MODEL=nomic-embed-text
```

### Ports
| Port | Service |
|------|---------|
| 5433 | PostgreSQL 16 |
| 6380 | Redis 7 |
| 9090 | GAMI API |
| 11434 | Ollama (embeddings) |
| 8000 | vLLM (optional, GPU) |

## Architecture

```
Claude Code ──MCP──> GAMI MCP Server (stdio)
                         │
                    GAMI API (:9090)
                         │
              ┌──────────┼──────────┐
              │          │          │
         PostgreSQL    Redis     Ollama
         :5433         :6380     :11434
         (pgvector)    (cache)   (embed)
         (AGE graph)             (CPU)
```

## Troubleshooting

- **MCP tools not showing**: Restart Claude Code after adding MCP config
- **Ollama 500 errors**: `sudo systemctl restart ollama` or `ollama serve &`
- **PostgreSQL connection refused**: `sudo systemctl start postgresql@16-main`
- **Redis down**: `sudo systemctl start redis-server` or `redis-server --port 6380 &`
- **Slow recall**: Check embeddings — `SELECT count(*) FROM segments WHERE embedding IS NULL`
