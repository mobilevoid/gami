# GAMI - Graph-Augmented Memory Interface

> **Give your AI a mind that never forgets.**

GAMI is long-term memory for AI agents. While other systems treat memory as flat vector search, GAMI understands that knowledge has *shape* — hierarchies, categories, and relationships that flat embeddings destroy.

## Why GAMI?

**The Problem:** AI assistants forget everything between sessions. RAG systems retrieve documents but don't *understand* them. Vector databases lose hierarchical structure — "PostgreSQL" and "database" end up equidistant from "MySQL" even though one contains the other.

**The Solution:** GAMI embeds knowledge into a *product manifold* — a geometric space where:
- **Hierarchies curve naturally** (Poincaré hyperbolic space)
- **Categories cluster on spheres** (like topics on a globe)
- **Semantics stay searchable** (Euclidean for fast pgvector queries)

Then it *dreams* — an 18-phase background cycle that extracts entities, resolves aliases, learns workflows, and consolidates memories while your agent sleeps.

## Features

| Feature | What It Does |
|---------|--------------|
| **Product Manifold H³² × S¹⁶ × E⁶⁴** | Embeds knowledge in curved space that preserves hierarchy and category structure |
| **Dream Cycle** | 18 background phases: extract, consolidate, learn workflows, compress, verify |
| **Hybrid Search** | Vector + lexical + cross-encoder reranking for 25-40% better precision |
| **Multi-Index Routing** | Queries automatically route to entities, claims, procedures, or relations |
| **Universal Integration** | MCP server (27 tools), REST API, or direct Python — works with any agent framework |
| **Real-Time Manifold** | New memories get manifold coordinates immediately — no waiting for batch |
| **Multi-Tenant** | Isolated knowledge bases with tenant-level access control |
| **LLM Agnostic** | Works with vLLM, Ollama, OpenAI, Anthropic, or any OpenAI-compatible API |

## How Product Manifolds Work

Flat vector spaces can't represent "PostgreSQL is a type of database" without losing precision. GAMI solves this with a **product manifold** — three geometric spaces working together:

```
H³² (Poincaré Ball)     S¹⁶ (Sphere)           E⁶⁴ (Euclidean)
     ┌─────┐              ┌─────┐               ┌─────┐
     │  ·  │ ← origin     │     │               │     │
     │ / \ │              │  ·  │ ← cluster     │ · · │ ← similar
     │·   ·│ ← children   │ · · │   together    │· · ·│   items
     └─────┘              └─────┘               └─────┘
    Hierarchies          Categories            Semantics
```

| Space | Dimensions | Geometry | Captures |
|-------|------------|----------|----------|
| **Hyperbolic** | 32 | Poincaré ball | Parent-child relationships — "database" closer to origin than "PostgreSQL" |
| **Spherical** | 16 | Unit sphere | Type clusters — all "person" entities cluster together |
| **Euclidean** | 64 | Flat space | Semantic similarity — enables fast pgvector ANN queries |

**Two-stage retrieval:**
1. **Fast pre-filter** — pgvector on 64d Euclidean component (milliseconds)
2. **Precise rerank** — full manifold geodesic distance on top candidates

Falls back to standard vector search when manifold coordinates aren't yet populated.

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

The dream cycle runs 18 phases in the background to synthesize and consolidate knowledge:

| Phase | Description |
|-------|-------------|
| `extract` | Entity extraction from segments |
| `summarize` | Generate summaries |
| `resolve` | Alias resolution |
| `reconcile` | Conflict resolution |
| `verify_memories` | Memory verification |
| `relate` | Build entity relations |
| `score` | Importance scoring |
| `embed` | Generate base 768d embeddings |
| `manifold_embeddings` | Generate legacy manifold projections |
| `product_manifold_coords` | Compute H^32 × S^16 × E^64 manifold coordinates |
| `deep_dream` | Deep synthesis |
| `auto_approve` | Auto-approve high-confidence items |
| `learning` | Pattern learning |
| `causal` | Causal relation extraction |
| `consolidate` | Memory consolidation |
| `compress` | Lossless compression with deltas |
| `extract_workflows` | Workflow pattern extraction |
| `trust` | Agent trust scoring |

## Database Schema

Core tables (38 total):
- `segments` - Text chunks with embeddings
- `sources` - Source documents
- `entities` - Extracted entities
- `claims` - Factual claims (SPO format)
- `relations` - Entity relationships
- `assistant_memories` - Semantic memories
- `memory_clusters` - Consolidated abstractions
- `product_manifold_coords` - H^32 × S^16 × E^64 coordinates
- `procedures` - Workflow storage
- `compression_deltas` - Lossless compression facts
- `causal_relations` - Cause-effect relationships

## Directory Structure

```
gami/
├── api/                 # FastAPI application
│   ├── search/          # Reranker, hybrid search, manifold search
│   ├── services/        # Business logic
│   ├── llm/             # LLM providers + manifold embeddings
│   └── routers/         # API endpoints
├── manifold/            # Query routing and multi-index retrieval
│   └── retrieval/       # Query classification, index routing
├── mcp_tools/           # MCP server and tools
├── scripts/             # Dream cycle, utilities
├── storage/sql/         # SQL migrations
├── install/             # Installation scripts
├── cli/                 # CLI tools and hooks
└── tests/               # Test suite
```

## Scripts

### Data Ingestion

```bash
# Ingest files into a tenant
python scripts/ingest_tenant.py --tenant my-project --directory ./docs/

# Ingest with specific file types
python scripts/ingest_tenant.py --tenant my-project --directory ./docs/ --extensions .md,.txt,.pdf

# Create tenant and ingest
python scripts/ingest_tenant.py --tenant new-project --create-tenant --file ./README.md

# Dry run (show what would be ingested)
python scripts/ingest_tenant.py --tenant my-project --directory ./docs/ --dry-run
```

### Manifold Coordinate Population

For GPU-accelerated manifold coordinate generation:

```bash
# Requires GAMI_DATABASE_URL environment variable
export GAMI_DATABASE_URL="postgresql://user:pass@localhost:5432/gami"

# Populate all segments and entities
conda run -n gami-embed python scripts/populate_product_manifold.py --batch-size 1024 --type all

# Populate only segments
python scripts/populate_product_manifold.py --type segment --limit 10000
```

### Dream Cycle

```bash
# Run dream cycle for 24 hours
python scripts/dream_cycle.py --duration 24

# Run specific phase
python scripts/dream_cycle.py --phase product_manifold_coords --duration 1
```

## Configuration Reference

See `.env.example` for all available configuration options.

Key settings:
- `DATABASE_URL` - PostgreSQL connection string
- `REDIS_URL` - Redis connection string
- `GAMI_LLM_BACKEND` - LLM provider (ollama, vllm, openai, anthropic)
- `GAMI_RERANKER_ENABLED` - Enable cross-encoder reranking
- `GAMI_API_KEY` - Optional API key for authentication

## Project History

GAMI evolved from research into persistent AI memory systems, driven by frustration with RAG systems that couldn't preserve knowledge structure.

| Date | Milestone |
|------|-----------|
| **2024 Q3** | Initial research into hyperbolic embeddings for hierarchical knowledge |
| **2024 Q4** | First prototype with Poincaré ball embeddings, proof-of-concept MCP server |
| **2025 Q1** | Multi-tenant architecture, dream cycle concept, hybrid search |
| **2025 Q2** | Production deployment, 6 tenants, cross-encoder reranking |
| **2025 Q3** | Workflow extraction, memory consolidation, lossless compression |
| **2026 Q1** | Product manifold (H³² × S¹⁶ × E⁶⁴), real-time manifold coordinates |
| **2026 Q2** | Public release, 2M+ segments, 18-phase dream cycle |

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
