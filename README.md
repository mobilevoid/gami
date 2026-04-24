# GAMI - Graph-Augmented Memory Intelligence

> **Version 2.0** | Multi-tenant embedded document database with hybrid search, MCP server, and dream-mode knowledge synthesis.

## Overview

GAMI is an AI memory system that provides:
- **Hybrid Search**: Vector + lexical search with cross-encoder reranking
- **Multi-Index Retrieval**: Query routing to optimal indexes (entities, claims, procedures, relations, etc.)
- **Dream Cycle**: Background knowledge synthesis (17 phases) for entity extraction, consolidation, and workflow learning
- **MCP Integration**: 24+ tools for AI agent memory access
- **Multi-Tenant**: Isolated knowledge bases per tenant

## Key Features

### Phase 1: Cross-Encoder Reranking
- 25-40% precision improvement on retrieval
- Uses `ms-marco-MiniLM-L-6-v2` for reranking
- Configurable via `GAMI_RERANKER_ENABLED`

### Phase 2: Mem0-Style Memory Operations
- Intelligent duplicate detection (ADD/UPDATE/DELETE/NOOP)
- ~60% storage reduction over time
- LLM-assisted consolidation decisions

### Phase 3: Lossless Compression
- Delta storage for unique facts not in cluster abstractions
- `detail_level` parameter: summary, normal, full
- Original text never deleted, only tiered

### Phase 4 & 8: Workflow Learning
- Extracts workflow patterns from conversation sessions
- Stores as workflow memories (not rigid procedures)
- Natural consolidation via dream cycle

### Phase 5: Bi-Temporal Queries
- Filter by `event_time` (when it happened) vs `ingested_at` (when learned)
- Parameters: `event_after`, `event_before`, `ingested_after`, `ingested_before`

### Phase 6: Query Routing & Multi-Index
- 8 query modes with specialized routing
- 8 index types: segments, entities, claims, relations, procedures, memories, clusters, causal
- Weighted fusion based on query intent

### Phase 7: Contradiction-Aware Retrieval
- Detects conflicting information
- Reports contradictions in recall results
- Evidence scoring with corroboration/contradiction factors

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export DATABASE_URL="postgresql://gami:password@localhost:5433/gami"
export REDIS_URL="redis://localhost:6379/0"
export VLLM_URL="http://localhost:8000"

# Start the MCP server
python -m mcp_tools.server

# Start the API
uvicorn api.main:app --port 8123

# Start dream cycle (background)
python scripts/dream_cycle.py --duration 24
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `memory_recall` | Recall relevant memories with token budget |
| `memory_remember` | Store a new memory (auto-consolidation) |
| `memory_search` | Direct hybrid search |
| `memory_suggest_procedure` | Suggest workflow patterns |
| `memory_correct` | Fix incorrect information |
| `memory_verify` | Verify claims against knowledge |
| `memory_context` | Get entity context and relations |
| `ingest_file` | Add files to knowledge base |
| `dream_start/stop/status` | Control dream cycle |
| `admin_stats` | System statistics |

## Database Schema

Core tables:
- `segments` - Text chunks with embeddings
- `sources` - Source documents
- `entities` - Extracted entities
- `claims` - Factual claims (SPO format)
- `relations` - Entity relationships
- `assistant_memories` - Semantic memories
- `memory_clusters` - Consolidated abstractions
- `procedures` - Legacy workflow storage
- `compression_deltas` - Lossless compression facts

## Dream Cycle Phases (17)

1. `extract` - Entity extraction
2. `summarize` - Generate summaries
3. `resolve` - Alias resolution
4. `reconcile` - Conflict resolution
5. `verify_memories` - Memory verification
6. `relate` - Build relations
7. `score` - Importance scoring
8. `embed` - Generate embeddings
9. `manifold_embeddings` - Multi-manifold embeddings
10. `deep_dream` - Deep synthesis
11. `auto_approve` - Auto-approve high-confidence
12. `learning` - Pattern learning
13. `causal` - Causal relation extraction
14. `consolidate` - Memory consolidation (type-agnostic)
15. `compress` - Lossless compression
16. `extract_procedures` - Workflow extraction (creates workflow memories)
17. `trust` - Agent trust scoring

## Configuration

Environment variables:
```bash
# Database
DATABASE_URL=postgresql://gami:pass@localhost:5433/gami
REDIS_URL=redis://localhost:6379/0

# LLM
VLLM_URL=http://localhost:8000
OLLAMA_URL=http://localhost:11434

# Reranker
GAMI_RERANKER_ENABLED=true
GAMI_RERANKER_TOP_K=50
GAMI_RERANKER_BLEND_RATIO=0.7

# Dream
GAMI_DREAM_IDLE_CHECK=true
GAMI_DREAM_MAX_DURATION=28800
```

## Directory Structure

```
/opt/gami/
├── api/               # FastAPI application
│   ├── search/        # Reranker, hybrid search
│   ├── services/      # Business logic
│   └── routers/       # API endpoints
├── manifold/          # Multi-manifold retrieval
│   └── retrieval/     # Query routing, multi-index
├── mcp_tools/         # MCP server and tools
├── scripts/           # Dream cycle, migrations
├── storage/           # SQL migrations
└── tests/             # Test suite
```

## License

Proprietary - All rights reserved.

## Contributing

Internal project - contact the maintainers for contribution guidelines.
