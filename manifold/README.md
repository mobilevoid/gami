# GAMI Multi-Manifold Memory System

> **Version 0.1.0** | **Branch:** feature/multi-manifold | **Status:** Ready for integration testing
>
> This is an isolated module. No imports into production code until shadow mode validation completes.

## Overview

Replaces single-embedding RAG with **6 specialized manifolds** and **query-conditioned weighting**:

| Manifold | Dimension | Purpose | Implementation |
|----------|-----------|---------|----------------|
| Topic | 768d dense | General similarity | `embedding.py` |
| Claim | 768d SPO | Fact extraction | `canonical/claim_normalizer.py` |
| Procedure | 768d steps | Ordered sequences | `canonical/procedure_normalizer.py` |
| Relation | Graph-derived | Structure similarity | `scoring/relation.py` |
| Time | 12 features | Temporal relevance | `temporal/feature_extractor.py` |
| Evidence | 5 scores | Verification confidence | `scoring/evidence.py` |

## Quick Start

```python
from manifold import recall, classify_query_v2, ManifoldConfig

# Classify a query
result = classify_query_v2("What's the database password?")
print(result.mode)  # QueryModeV2.FACT_LOOKUP

# Recall memories (async)
result = await recall("When did the deployment happen?", top_k=10)
for c in result.candidates:
    print(f"{c.fused_score:.2f}: {c.text[:80]}")
```

## Query Modes (8 types)

| Mode | Example | High-Weight Manifolds |
|------|---------|----------------------|
| `fact_lookup` | "What's the password?" | claim, evidence |
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

## Components (58 files, ~14,000 lines)

### Core
- `retrieval/orchestrator.py` - Main coordinator
- `retrieval/query_classifier_v2.py` - Mode detection + weight selection
- `retrieval/manifold_fusion.py` - Score fusion with alpha weights
- `retrieval/shadow_mode.py` - A/B comparison for safe rollout

### Scoring
- `scoring/promotion.py` - 7-factor promotion scoring
- `scoring/evidence.py` - 5-dimensional evidence
- `scoring/relation.py` - Graph fingerprints

### Infrastructure
- `config.py` - Central configuration (env/JSON)
- `repository.py` - Async database layer
- `embedding.py` - Ollama client with cache/batch/retry
- `exceptions.py` - 40+ error codes
- `validation.py` - Input validation
- `metrics.py` - Prometheus metrics
- `tasks.py` - Celery background tasks
- `cli.py` - Admin CLI

### MCP Tools
- `memory_recall` - Primary retrieval
- `memory_search` - Direct manifold search
- `memory_classify` - Classification only
- `memory_verify` - Claim verification
- `manifold_stats` - System stats

## CLI

```bash
python -m manifold.cli config --show     # Show configuration
python -m manifold.cli config --validate # Validate config
python -m manifold.cli stats --json      # System statistics
python -m manifold.cli migrate --claims  # Run migrations
python -m manifold.cli shadow --stats    # Shadow mode analysis
```

## Testing (~175+ tests)

```bash
pytest manifold/tests/ -v                    # All tests
pytest manifold/tests/test_integration.py -v # Integration tests
```

## Migration Path

1. Run `migrations/002_manifold_tables.sql`
2. `python scripts/migrate_canonical_claims.py`
3. `python scripts/compute_promotion_scores.py`
4. `python scripts/embed_promoted.py`
5. Enable shadow mode, analyze comparisons
6. Switch when metrics confirm parity

## Key Formulas

**Anchor Score:** `S = Σ αₘ·s'ₘ + β_lex + β_alias + β_cache - penalties`

**Promotion (7 factors):** `P = Σ wᵢ·factorᵢ` (importance, retrieval, diversity, confidence, novelty, centrality, relevance)

**Evidence (5 factors):** `E = 0.25·authority + 0.30·corroboration + 0.15·recency + 0.10·specificity + 0.20·non_contradiction`

## Directory Structure

```
manifold/
├── __init__.py, config.py, repository.py, embedding.py
├── exceptions.py, validation.py, metrics.py, tasks.py, cli.py
├── models/schemas.py
├── retrieval/{orchestrator,query_classifier_v2,manifold_fusion,shadow_mode}.py
├── scoring/{promotion,evidence,relation}.py
├── canonical/{claim_normalizer,procedure_normalizer}.py
├── temporal/feature_extractor.py
├── mcp/tools.py
├── scripts/{migrate_canonical_claims,compute_promotion_scores,embed_promoted}.py
├── migrations/002_manifold_tables.sql
└── tests/test_*.py, golden_queries.py
```
