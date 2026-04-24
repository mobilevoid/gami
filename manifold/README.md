# Multi-Manifold Memory System

> **Version 2.0** | **Status:** Production Ready
>
> A polyglot RAG system using 8 specialized indexes with query routing, cross-encoder reranking, and contradiction awareness.

## Overview

Replaces single-embedding RAG with **8 specialized indexes** that activate based on query routing:

| Index | Dimension | Purpose | Implementation |
|-------|-----------|---------|----------------|
| Segments | 768d dense | General text chunks | `retrieval/anchor_retrieval.py` |
| Entities | 768d | Named entities with descriptions | `retrieval/multi_index_retriever.py` |
| Claims | 768d SPO | Factual claims (subject-predicate-object) | `retrieval/multi_index_retriever.py` |
| Relations | Graph-derived | Entity relationships | `retrieval/multi_index_retriever.py` |
| Procedures | 768d | Workflow memories (consolidated) | `retrieval/multi_index_retriever.py` |
| Memories | 768d | Assistant memories | `retrieval/multi_index_retriever.py` |
| Clusters | 768d | Memory cluster abstractions | `retrieval/multi_index_retriever.py` |
| Causal | 768d | Cause-effect relationships | `retrieval/multi_index_retriever.py` |

### Enhancements (v2.0)
- **Cross-encoder reranking**: 25-40% precision improvement via `ms-marco-MiniLM-L-6-v2`
- **Query routing**: Pattern-based routing to optimal indexes
- **Contradiction detection**: Surfaces conflicting information in results
- **Bi-temporal filtering**: Query by event time vs ingestion time
- **Workflow memories**: Extracted patterns that consolidate naturally

## Features

- **Query-conditioned retrieval**: Different query types activate different manifold weights
- **Dual-date tracking**: Tracks both when data was ingested AND dates mentioned in content
- **Promotion scoring**: 7-factor scoring determines which objects get specialized embeddings
- **Shadow mode**: A/B comparison for safe rollout without breaking existing system
- **Evidence verification**: Contradiction detection and corroboration scoring
- **MCP integration**: Tools for AI agent memory access

## Quick Start

```bash
# Install dependencies
pip install asyncpg redis httpx celery

# Set environment variables
export DATABASE_URL="postgresql://user:pass@localhost:5432/mydb"
export REDIS_URL="redis://localhost:6379/0"
export MANIFOLD_OLLAMA_URL="http://localhost:11434"

# Run tests
pytest manifold/tests/ -v
```

```python
from manifold import recall, classify_query_v2, ManifoldConfig

# Classify a query
result = classify_query_v2("When did the server crash?")
print(result.mode)  # QueryModeV2.TIMELINE

# Recall memories (async)
result = await recall("How do I deploy the application?", top_k=10)
for c in result.candidates:
    print(f"{c.fused_score:.2f}: {c.text[:80]}")
```

## Query Modes (8 types)

| Mode | Example | High-Weight Manifolds |
|------|---------|----------------------|
| `fact_lookup` | "What's the API endpoint?" | claim, evidence |
| `timeline` | "When did X happen?" | time |
| `procedure` | "How do I deploy?" | procedure |
| `verification` | "Is it true that...?" | evidence, claim |
| `comparison` | "Difference between A and B?" | relation |
| `synthesis` | "Summarize the architecture" | topic |
| `report` | "List all services" | topic |
| `assistant_memory` | "What did we discuss?" | time, topic |

## Architecture

```
Query → Classifier → α Weights → Parallel Retrieval → Fusion → Ranking
                                        ↓
                    ┌───────┬───────┬───────┬───────┐
                    Topic   Claim   Proc    Time    ...
                    ↓       ↓       ↓       ↓
                    └───────┴───────┴───────┴───────┘
                                    ↓
                    Score = Σ αₘ·sₘ + β_lex + β_cache - penalties
```

## Configuration

All settings via environment variables or `ManifoldConfig`:

```bash
DATABASE_URL=postgresql://localhost:5432/manifold
REDIS_URL=redis://localhost:6379/0
MANIFOLD_EMBEDDING_MODEL=nomic-embed-text
MANIFOLD_OLLAMA_URL=http://localhost:11434
MANIFOLD_PROMOTION_THRESHOLD=0.65
MANIFOLD_SHADOW_MODE=true
```

## Components

### Core Retrieval
- `retrieval/orchestrator.py` - Main coordinator with parallel manifold search
- `retrieval/query_classifier_v2.py` - Mode detection + weight selection
- `retrieval/query_routing.py` - Pattern-based query routing to indexes
- `retrieval/multi_index_retriever.py` - Multi-index search with fusion
- `retrieval/anchor_retrieval.py` - Primary segment retrieval
- `retrieval/manifold_fusion.py` - Score fusion with alpha weights
- `retrieval/shadow_runner.py` - A/B comparison for safe rollout

### Scoring
- `scoring/promotion.py` - 7-factor promotion scoring
- `scoring/evidence.py` - 5-dimensional evidence with contradiction detection
- `scoring/relation.py` - Graph fingerprints via Apache AGE

### Infrastructure
- `config.py` - Central configuration (env/JSON)
- `repository.py` - Async database layer (asyncpg)
- `embedding.py` - Embedding client with LRU cache/batch/retry
- `exceptions.py` - 40+ structured error codes
- `validation.py` - Input validation
- `metrics.py` - Prometheus-compatible metrics
- `tasks.py` - Celery background tasks
- `cli.py` - Admin CLI

### MCP Tools
- `memory_recall` - Primary retrieval with classification
- `memory_search` - Direct manifold search
- `memory_classify` - Classification only
- `memory_verify` - Claim verification
- `manifold_stats` - System statistics

## CLI

```bash
python -m manifold.cli config --show     # Show configuration
python -m manifold.cli config --validate # Validate config
python -m manifold.cli stats --json      # System statistics
python -m manifold.cli embed --type all  # Generate embeddings
python -m manifold.cli promote --type segments  # Compute promotion scores
python -m manifold.cli shadow --stats    # Shadow mode analysis
```

## Database Requirements

- **PostgreSQL 14+** with extensions:
  - `pgvector` - Vector similarity search
  - `pg_trgm` - Trigram text search
  - `age` - Apache AGE for graph queries (optional)
- **Redis 6+** - Caching layer

Run migrations:
```bash
psql -f migrations/002_manifold_tables.sql
```

## Testing

```bash
pytest manifold/tests/ -v                    # All tests (254 tests)
pytest manifold/tests/test_integration.py -v # Integration tests
pytest manifold/tests/test_orchestrator.py -v # Orchestrator tests
```

## Migration Path (for existing systems)

1. Run `migrations/002_manifold_tables.sql`
2. `python scripts/migrate_canonical_claims.py`
3. `python scripts/compute_promotion_scores.py`
4. `python scripts/embed_promoted.py`
5. Enable shadow mode (`MANIFOLD_SHADOW_MODE=true`)
6. Analyze comparisons via CLI or API
7. Switch when metrics confirm parity

## Key Formulas

**Anchor Score:**
```
S = Σ αₘ·s'ₘ + β_lex + β_alias + β_cache - penalties
```

**Promotion (7 factors):**
```
P = w₁·importance + w₂·retrieval + w₃·diversity + w₄·confidence 
  + w₅·novelty + w₆·centrality + w₇·relevance
```

**Evidence (5 factors):**
```
E = 0.25·authority + 0.30·corroboration + 0.15·recency 
  + 0.10·specificity + 0.20·non_contradiction
```

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions welcome! Please read CONTRIBUTING.md for guidelines.
