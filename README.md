# GAMI - Graph-Augmented Memory Interface

> **Version 2.0** | Multi-tenant embedded document database with hybrid search, MCP server, and dream-mode knowledge synthesis.

## Overview

GAMI is a persistent AI memory system designed for use with AI assistants like Claude Code. It provides:

- **Hybrid Search**: Vector + lexical search with cross-encoder reranking
- **Multi-Index Retrieval**: Query routing to optimal indexes (entities, claims, procedures, relations)
- **Dream Cycle**: Background knowledge synthesis (17 phases) for entity extraction, consolidation, and workflow learning
- **MCP Integration**: 27 tools for AI agent memory access
- **Multi-Tenant**: Isolated knowledge bases per tenant
- **Multi-Backend LLM**: Support for vLLM, Ollama, OpenAI, and Anthropic APIs

## Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/mobilevoid/gami.git
cd gami

# Copy and configure environment
cp .env.example .env
# Edit .env with your settings

# Start all services
docker compose up -d

# Pull embedding model (if using Ollama)
docker exec gami-ollama ollama pull nomic-embed-text

# Verify
curl http://localhost:9090/health
```

### Option 2: Manual Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up PostgreSQL 16 with pgvector
sudo apt install postgresql-16 postgresql-16-pgvector

# Set up Redis
sudo apt install redis-server

# Configure environment
export DATABASE_URL="postgresql://gami:password@localhost:5433/gami"
export REDIS_URL="redis://localhost:6379/0"
export OLLAMA_URL="http://localhost:11434"
export GAMI_LLM_BACKEND="ollama"

# Initialize database
psql -U gami -d gami -f install/schema.sql

# Start API
uvicorn api.main:app --port 9090

# Start dream cycle (optional)
python scripts/dream_cycle.py --duration 24
```

## LLM Backend Configuration

GAMI supports multiple LLM backends for embeddings and dream cycle synthesis. Configure via environment variables:

### Ollama (Default, Local)
```bash
GAMI_LLM_BACKEND=ollama
OLLAMA_URL=http://localhost:11434
EMBEDDING_MODEL=nomic-embed-text
GAMI_LLM_MODEL=llama3.2  # Or any Ollama model
```

### vLLM (Local, GPU)
```bash
GAMI_LLM_BACKEND=vllm
VLLM_URL=http://localhost:8000/v1
GAMI_LLM_MODEL=your-model-name
```

### OpenAI API
```bash
GAMI_LLM_BACKEND=openai
OPENAI_API_KEY=sk-...
GAMI_LLM_MODEL=gpt-4o-mini
```

### Anthropic API
```bash
GAMI_LLM_BACKEND=anthropic
ANTHROPIC_API_KEY=sk-ant-...
GAMI_LLM_MODEL=claude-sonnet-4-20250514
```

## MCP Tools (27 total)

| Tool | Description |
|------|-------------|
| `memory_recall` | Recall relevant memories with token budget |
| `memory_remember` | Store a new memory (auto-consolidation) |
| `memory_search` | Direct hybrid search |
| `memory_context` | Get entity context and relations |
| `memory_suggest_procedure` | Suggest workflow patterns |
| `memory_correct` | Fix incorrect information |
| `memory_verify` | Verify claims against knowledge |
| `memory_update` | Update existing memory |
| `memory_forget` | Archive a memory |
| `memory_feedback` | Provide feedback on retrieval |
| `memory_cite` | Get source citations |
| `ingest_file` | Add files to knowledge base |
| `ingest_source` | Add source document |
| `bulk_ingest` | Batch ingestion |
| `graph_explore` | Explore entity relationships |
| `dream_start` | Start dream cycle |
| `dream_stop` | Stop dream cycle |
| `dream_status` | Check dream cycle status |
| `dream_haiku` | Generate knowledge haiku |
| `tenant_search` | Search within tenant |
| `tenant_stats` | Tenant statistics |
| `create_tenant` | Create new tenant |
| `admin_stats` | System-wide statistics |
| `run_haiku_extraction` | Extract haiku from text |
| `store_extractions` | Store extracted entities |
| `review_proposals` | Review pending proposals |
| `get_unprocessed_segments` | Get segments needing processing |

## Claude Code Integration

### Configure MCP Server

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "gami": {
      "command": "python",
      "args": ["-m", "mcp_tools.server"],
      "cwd": "/path/to/gami",
      "env": {
        "PYTHONPATH": "/path/to/gami",
        "DATABASE_URL": "postgresql://gami:password@localhost:5433/gami"
      }
    }
  }
}
```

### Session Hooks (Optional)

Add hooks to preserve context before compaction:

```json
{
  "hooks": {
    "PreCompact": [{"hooks": [{"type": "command", "command": "/path/to/gami/cli/journal_hook.sh"}]}],
    "SessionEnd": [{"hooks": [{"type": "command", "command": "/path/to/gami/cli/journal_hook.sh"}]}]
  }
}
```

## Dream Cycle

The dream cycle runs 17 phases in the background to synthesize and consolidate knowledge:

| Phase | Description |
|-------|-------------|
| `extract` | Entity extraction from segments |
| `summarize` | Generate summaries |
| `resolve` | Alias resolution |
| `reconcile` | Conflict resolution |
| `verify_memories` | Memory verification |
| `relate` | Build entity relations |
| `score` | Importance scoring |
| `embed` | Generate embeddings |
| `manifold_embeddings` | Multi-manifold embeddings |
| `deep_dream` | Deep synthesis |
| `auto_approve` | Auto-approve high-confidence items |
| `learning` | Pattern learning |
| `causal` | Causal relation extraction |
| `consolidate` | Memory consolidation |
| `compress` | Lossless compression with deltas |
| `extract_workflows` | Workflow pattern extraction |
| `trust` | Agent trust scoring |

## Database Schema

Core tables (36 total):
- `segments` - Text chunks with embeddings
- `sources` - Source documents
- `entities` - Extracted entities
- `claims` - Factual claims (SPO format)
- `relations` - Entity relationships
- `assistant_memories` - Semantic memories
- `memory_clusters` - Consolidated abstractions
- `procedures` - Workflow storage
- `compression_deltas` - Lossless compression facts
- `causal_relations` - Cause-effect relationships

## Directory Structure

```
gami/
├── api/                 # FastAPI application
│   ├── search/          # Reranker, hybrid search
│   ├── services/        # Business logic
│   ├── llm/             # Multi-backend LLM providers
│   └── routers/         # API endpoints
├── manifold/            # Multi-manifold retrieval
│   └── retrieval/       # Query routing, multi-index
├── mcp_tools/           # MCP server and tools
├── scripts/             # Dream cycle, utilities
├── storage/sql/         # SQL migrations
├── install/             # Installation scripts
├── cli/                 # CLI tools and hooks
└── tests/               # Test suite
```

## Configuration Reference

See `.env.example` for all available configuration options.

Key settings:
- `DATABASE_URL` - PostgreSQL connection string
- `REDIS_URL` - Redis connection string
- `GAMI_LLM_BACKEND` - LLM provider (ollama, vllm, openai, anthropic)
- `GAMI_RERANKER_ENABLED` - Enable cross-encoder reranking
- `GAMI_API_KEY` - Optional API key for authentication

## License

This software is licensed under the **Stalwart LLC Source Available License v1.0**.

You may view, download, and use this software for personal, educational, and non-commercial evaluation purposes. Commercial use requires a license from Stalwart LLC.

See [LICENSE](LICENSE) for full terms.

For commercial licensing: choll@stalwartresources.com

## Contributing

Contributions are welcome under the terms of the license. By submitting contributions, you grant Stalwart LLC a perpetual, irrevocable license to use, modify, and sublicense your contributions.

Before contributing:
1. Review the [LICENSE](LICENSE)
2. Ensure your contribution doesn't include proprietary code
3. Test your changes locally
4. Submit a pull request with a clear description

## Support

For issues and questions:
- Open an issue on GitHub
- Contact: choll@stalwartresources.com
