# GAMI Multi-Manifold Memory System — Implementation Roadmap

> **Status**: Planning/Development — NOT FOR PRODUCTION USE
> **Branch**: `feature/multi-manifold`
> **Author**: Claude Opus 4.6
> **Date**: 2026-04-10

---

## Executive Summary

This document specifies the complete migration path from GAMI's current single-embedding retrieval system to a multi-manifold memory architecture. The new system maintains six semantic manifolds (topic, claim, procedure, relation, time, evidence), supports query-conditioned manifold weighting, and preserves all existing invariants around provenance, contradiction handling, and audit trails.

**Key principle**: This is an evolutionary enhancement, not a rewrite. Current functionality continues uninterrupted while new capabilities are built alongside.

---

## Table of Contents

1. [Current State Assessment](#1-current-state-assessment)
2. [Target Architecture](#2-target-architecture)
3. [Manifold Definitions](#3-manifold-definitions)
4. [Mathematical Framework](#4-mathematical-framework)
5. [Database Schema Additions](#5-database-schema-additions)
6. [MCP Tool Changes](#6-mcp-tool-changes)
7. [Migration Strategy](#7-migration-strategy)
8. [Neo4j Migration Path](#8-neo4j-migration-path)
9. [Safeguards and Invariants](#9-safeguards-and-invariants)
10. [Implementation Phases](#10-implementation-phases)
11. [Testing Strategy](#11-testing-strategy)
12. [Rollback Procedures](#12-rollback-procedures)

---

## 1. Current State Assessment

### 1.1 Existing Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Current GAMI Architecture                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  PostgreSQL 5433                                                 │
│  ├── Relational Tables                                          │
│  │   ├── sources (raw corpus)                                   │
│  │   ├── segments (hierarchical chunks)                         │
│  │   ├── entities (extracted named entities)                    │
│  │   ├── claims (extracted propositions)                        │
│  │   ├── assistant_memories (durable memories)                  │
│  │   ├── provenance (source tracking)                           │
│  │   └── tenants (multi-tenant config)                          │
│  │                                                              │
│  ├── pgvector Extension                                         │
│  │   └── Single embedding column per table (768d nomic)         │
│  │                                                              │
│  └── Apache AGE Extension                                       │
│      └── manifold_graph (entities, relations, paths)                │
│                                                                  │
│  Redis 6380                                                      │
│  └── Hot cache (query anchors, neighborhoods, session state)    │
│                                                                  │
│  Ollama :11434                                                   │
│  └── nomic-embed-text (embedding), qwen3:8b (classification)    │
│                                                                  │
│  vLLM :8000                                                      │
│  └── Qwen3.5-27B (extraction, summarization)                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Current Retrieval Pipeline

```python
# Simplified current flow
query → embed(query) → hybrid_search(vector + lexical) → score → budget → cite
```

**Current Query Modes** (7 total):
- `factual` — specific fact lookup
- `synthesis` — combining multiple sources
- `timeline` — temporal/chronological queries
- `entity_centric` — about a specific entity
- `assistant_memory` — referencing prior conversations
- `verification` — checking claim truth
- `report` — comprehensive coverage

**Current Scoring Formula**:
```
score = max(vector_similarity, lexical_score * 0.7)
effective_score = score * importance * recency_factor * type_boost
```

### 1.3 Current Scale (as of 2026-04-10)

| Metric | books tenant | example-tenant tenant | Total |
|--------|-------------|-------------------|-------|
| Segments | 260,690 | ~50,000 | ~320,000 |
| Entities | 89,000+ | ~15,000 | ~105,000 |
| Claims | ~10,000 | ~5,000 | ~15,000 |
| Summaries | 72,600+ | ~8,000 | ~80,000 |
| Memories | - | ~500 | ~500 |

### 1.4 Current Limitations

1. **Single semantic space** — topical similarity conflated with procedural, temporal, and evidential similarity
2. **No canonical forms** — claims stored as prose, not structured SPO tuples
3. **Limited promotion scoring** — no formal promotion criteria for durable memory
4. **Query classification** — pattern-based + LLM, but no manifold weighting
5. **Graph underutilized** — AGE present but relation phase not yet producing rich connections

---

## 2. Target Architecture

### 2.1 Multi-Manifold Design

```
┌─────────────────────────────────────────────────────────────────┐
│                  Multi-Manifold Architecture                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Query Layer                            │   │
│  │  query → classify → compute α(q) → retrieve_manifolds    │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Manifold Retrieval Layer                     │   │
│  │                                                          │   │
│  │   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       │   │
│  │   │  Topic  │ │  Claim  │ │Procedure│ │Relation │       │   │
│  │   │   768d  │ │   768d  │ │   768d  │ │ derived │       │   │
│  │   └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘       │   │
│  │        │           │           │           │             │   │
│  │   ┌────┴────┐ ┌────┴────┐ ┌────┴────┐ ┌────┴────┐       │   │
│  │   │  Time   │ │Evidence │ │ Lexical │ │  Cache  │       │   │
│  │   │features │ │  score  │ │  BM25   │ │   hot   │       │   │
│  │   └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘       │   │
│  │        │           │           │           │             │   │
│  │        └───────────┴───────────┴───────────┘             │   │
│  │                        │                                  │   │
│  │                        ▼                                  │   │
│  │              Fusion: S_anchor(q, O, α)                   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                  Graph Expansion Layer                    │   │
│  │  anchor_set → beam_expand(G, depth=3, width=20)          │   │
│  │  → drift_penalty → contradiction_check → path_score      │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Reranking Layer                        │   │
│  │  S_final = w_anchor*S_anchor + w_graph*S_graph           │   │
│  │          + w_prov*S_prov + w_conf*S_conf + ...           │   │
│  │          - w_contra*P_contra - w_drift*P_drift           │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                Context Assembly Layer                     │   │
│  │  budget_manager → priority_sort → dedup → cite → output  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 What Changes

| Component | Current | New |
|-----------|---------|-----|
| Semantic embeddings | 1 per object (topic) | Up to 3 dense (topic, claim, procedure) + derived (relation, time, evidence) |
| Query classification | 7 modes | 8 modes + α(q) manifold weights |
| Anchor retrieval | Single vector search | Multi-manifold fusion |
| Canonical forms | None | SPO claims, ordered procedures |
| Promotion scoring | Ad-hoc | Formal 7-factor formula |
| Shadow mode | None | Parallel old/new comparison |

### 2.3 What Stays The Same

- PostgreSQL as relational truth store
- Apache AGE as graph engine (for now)
- Redis as cache layer only
- Provenance tracking unchanged
- Contradiction preservation unchanged
- Tenant isolation unchanged
- All existing MCP tools continue working

---

## 3. Manifold Definitions

### 3.1 The Six Manifolds

| Manifold | Type | Dimensionality | Scope | Purpose |
|----------|------|----------------|-------|---------|
| **Topic** | Dense embedding | 768d | All objects | Coarse semantic affinity, "aboutness" |
| **Claim** | Dense embedding | 768d | Promoted claims, claim-heavy summaries | Propositional equivalence |
| **Procedure** | Dense embedding | 768d | Procedures, instructional content | Workflow/action sequence similarity |
| **Relation** | Derived features | 64d | High-centrality entities | Graph neighborhood fingerprint |
| **Time** | Structured features | 12d | Events, time-scoped claims | Temporal compatibility |
| **Evidence** | Composite score | 5 scalars | All provenance-tracked objects | Source quality and verification value |

### 3.2 Canonical Forms

**Claim Manifold Input**:
```
[SUBJECT] | [PREDICATE] | [OBJECT] | modality=[factual|possible|negated] | qualifiers=[list] | time=[temporal scope]
```

Example:
```
Vietnamese Communist Party | exercised control over | CPK | modality=factual | qualifiers=[significant, formative period] | time=[through 1973]
```

**Procedure Manifold Input**:
```
title=[procedure name] | prerequisites=[list] | steps=[ordered list] | outcome=[expected result]
```

Example:
```
title=Deploy GAMI to Production | prerequisites=[PostgreSQL 16, Redis 7, Python 3.11] | steps=[1. Run migrations, 2. Start API, 3. Start MCP server] | outcome=[GAMI accessible on :9000]
```

**Time Manifold Features**:
```python
[
    year_normalized,      # 0-1 scaled
    month_sin,           # cyclical encoding
    month_cos,
    day_sin,
    day_cos,
    is_range,            # boolean
    range_days,          # duration if range
    sequence_position,   # 0-1 within source
    has_explicit_date,   # boolean
    is_relative,         # "last week" vs "March 15"
    temporal_precision,  # year/month/day/hour
    is_ongoing,          # "since 2020"
]
```

**Evidence Manifold Scores**:
```python
{
    "source_count": int,           # how many sources support this
    "provenance_density": float,   # 0-1, how traceable
    "confidence": float,           # extraction confidence
    "source_diversity": float,     # 0-1, variety of source types
    "verification_status": str,    # unverified/supported/contested
}
```

### 3.3 Tiered Embedding Strategy

To control storage costs:

| Object Type | Topic | Claim | Procedure | Relation | Time | Evidence |
|-------------|-------|-------|-----------|----------|------|----------|
| All segments | ✓ | - | - | - | extracted | computed |
| Promoted claims | ✓ | ✓ | - | - | ✓ | computed |
| Promoted summaries | ✓ | if claim-heavy | if instructional | - | extracted | computed |
| High-centrality entities | ✓ | - | - | ✓ | - | computed |
| Procedures | ✓ | - | ✓ | - | - | computed |
| Assistant memories | ✓ | if claim-like | if procedural | - | ✓ | computed |

---

## 4. Mathematical Framework

### 4.1 Manifold Weighting (α)

For each query mode, define manifold weights:

```python
ALPHA_WEIGHTS = {
    "fact_lookup": {
        "topic": 0.15, "claim": 0.35, "procedure": 0.05,
        "relation": 0.20, "time": 0.05, "evidence": 0.20
    },
    "synthesis": {
        "topic": 0.35, "claim": 0.20, "procedure": 0.10,
        "relation": 0.20, "time": 0.05, "evidence": 0.10
    },
    "comparison": {
        "topic": 0.25, "claim": 0.25, "procedure": 0.05,
        "relation": 0.25, "time": 0.05, "evidence": 0.15
    },
    "timeline": {
        "topic": 0.10, "claim": 0.15, "procedure": 0.00,
        "relation": 0.15, "time": 0.40, "evidence": 0.20
    },
    "procedure": {
        "topic": 0.15, "claim": 0.10, "procedure": 0.45,
        "relation": 0.15, "time": 0.00, "evidence": 0.15
    },
    "assistant_memory": {
        "topic": 0.20, "claim": 0.20, "procedure": 0.15,
        "relation": 0.10, "time": 0.15, "evidence": 0.20
    },
    "verification": {
        "topic": 0.10, "claim": 0.20, "procedure": 0.00,
        "relation": 0.10, "time": 0.10, "evidence": 0.50
    },
    "report": {
        "topic": 0.30, "claim": 0.20, "procedure": 0.15,
        "relation": 0.20, "time": 0.05, "evidence": 0.10
    },
}
```

### 4.2 Anchor Score Fusion

```python
def compute_anchor_score(query, obj, alpha, scores):
    """
    S_anchor(q, O) = Σ_m α_m(q) · s'_m(q, O)
                   + β_lex · s_lex(q, O)
                   + β_alias · s_alias(q, O)
                   + β_cache · s_cache(q, O)
                   + β_prior · s_prior(O)
                   + β_type · s_type(O, q)
                   - β_dup · p_dup(O)
                   - β_noise · p_noise(O)
    """
    # Manifold scores (percentile-normalized)
    manifold_score = sum(
        alpha[m] * percentile_normalize(scores.get(m, 0))
        for m in MANIFOLDS
    )
    
    # Additional signals
    return (
        manifold_score
        + 0.10 * scores.get("lexical", 0)
        + 0.08 * scores.get("alias_match", 0)
        + 0.05 * scores.get("cache_hit", 0)
        + 0.05 * scores.get("prior_importance", 0)
        + 0.04 * scores.get("type_fit", 0)
        - 0.06 * scores.get("duplicate_penalty", 0)
        - 0.04 * scores.get("noise_penalty", 0)
    )
```

### 4.3 Promotion Score Formula

```python
def compute_promotion_score(obj):
    """
    promotion_score = 0.25 * importance
                    + 0.20 * retrieval_frequency
                    + 0.15 * source_diversity
                    + 0.15 * confidence
                    + 0.10 * novelty
                    + 0.10 * graph_centrality
                    + 0.05 * user_relevance
    
    Thresholds:
    - < 0.45: keep raw only
    - 0.45-0.70: provisional structured memory
    - > 0.70: promote to durable memory
    - > 0.85 and repeated: candidate for manifold embedding
    """
    return (
        0.25 * obj.importance_score
        + 0.20 * log_normalize(obj.retrieval_count)
        + 0.15 * obj.source_diversity
        + 0.15 * obj.confidence
        + 0.10 * compute_novelty(obj)
        + 0.10 * obj.graph_centrality
        + 0.05 * obj.user_relevance
    )
```

### 4.4 Graph Expansion Scoring

```python
RELATION_PRIORS = {
    "SUPPORTS": 1.00,
    "DERIVED_FROM": 0.95,
    "SUMMARIZES": 0.90,
    "PART_OF": 0.85,
    "INSTANCE_OF": 0.80,
    "UPDATED_BY": 0.80,
    "CONTRADICTS": 0.75,
    "RELATED_TO": 0.35,
}

def compute_path_score(query, path, anchor_score):
    """
    S_path(q, P) = S_anchor(q, O_0)
                 + Σ_i [λ_rel(type(r_i)) · weight(r_i) · confidence(r_i)]
                 - γ_depth · depth(P)
                 - γ_drift · D(q, P)
                 - γ_contra · C(P)
    """
    edge_score = sum(
        RELATION_PRIORS.get(e.type, 0.5) * e.weight * e.confidence
        for e in path.edges
    )
    
    drift_penalty = 1 - cosine(query.topic_embedding, path.centroid)
    contra_penalty = path.contradiction_count / (len(path.edges) + 1)
    
    return (
        anchor_score
        + edge_score
        - 0.15 * len(path.edges)  # depth penalty
        - 0.20 * drift_penalty
        - 0.25 * contra_penalty
    )
```

---

## 5. Database Schema Additions

### 5.1 New Tables

```sql
-- Manifold embeddings (separate from main tables for isolation)
CREATE TABLE manifold_embeddings (
    id BIGSERIAL PRIMARY KEY,
    target_id TEXT NOT NULL,
    target_type TEXT NOT NULL,  -- 'segment', 'claim', 'entity', 'summary', 'memory'
    manifold_type TEXT NOT NULL,  -- 'topic', 'claim', 'procedure'
    embedding vector(768),
    embedding_model TEXT NOT NULL DEFAULT 'nomic-embed-text',
    embedding_version INTEGER NOT NULL DEFAULT 1,
    canonical_form TEXT,  -- The text that was actually embedded
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(target_id, target_type, manifold_type)
);

CREATE INDEX idx_manifold_embeddings_target ON manifold_embeddings(target_id, target_type);
CREATE INDEX idx_manifold_embeddings_type ON manifold_embeddings(manifold_type);
CREATE INDEX idx_manifold_topic_vec ON manifold_embeddings 
    USING ivfflat (embedding vector_cosine_ops) 
    WHERE manifold_type = 'topic';
CREATE INDEX idx_manifold_claim_vec ON manifold_embeddings 
    USING ivfflat (embedding vector_cosine_ops) 
    WHERE manifold_type = 'claim';
CREATE INDEX idx_manifold_procedure_vec ON manifold_embeddings 
    USING ivfflat (embedding vector_cosine_ops) 
    WHERE manifold_type = 'procedure';

-- Canonical claims (structured SPO form)
CREATE TABLE canonical_claims (
    id BIGSERIAL PRIMARY KEY,
    claim_id TEXT NOT NULL REFERENCES claims(claim_id),
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT,
    modality TEXT DEFAULT 'factual',  -- factual, possible, negated
    qualifiers JSONB DEFAULT '[]',
    temporal_scope TEXT,
    canonical_text TEXT NOT NULL,  -- Full canonical form for embedding
    confidence FLOAT DEFAULT 0.5,
    extraction_method TEXT DEFAULT 'llm',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(claim_id)
);

CREATE INDEX idx_canonical_claims_subject ON canonical_claims(subject);
CREATE INDEX idx_canonical_claims_predicate ON canonical_claims(predicate);

-- Canonical procedures (structured workflow form)
CREATE TABLE canonical_procedures (
    id BIGSERIAL PRIMARY KEY,
    source_id TEXT,
    segment_id TEXT,
    title TEXT NOT NULL,
    prerequisites JSONB DEFAULT '[]',
    steps JSONB NOT NULL,  -- Array of {order, text, optional}
    expected_outcome TEXT,
    canonical_text TEXT NOT NULL,
    owner_tenant_id TEXT NOT NULL,
    confidence FLOAT DEFAULT 0.5,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(source_id, segment_id, title)
);

CREATE INDEX idx_canonical_procedures_tenant ON canonical_procedures(owner_tenant_id);
CREATE INDEX idx_canonical_procedures_title ON canonical_procedures(title);

-- Temporal features (structured time extraction)
CREATE TABLE temporal_features (
    id BIGSERIAL PRIMARY KEY,
    target_id TEXT NOT NULL,
    target_type TEXT NOT NULL,
    features FLOAT[12] NOT NULL,  -- Fixed 12-dimensional feature vector
    raw_temporal_text TEXT,
    start_date DATE,
    end_date DATE,
    is_range BOOLEAN DEFAULT FALSE,
    precision TEXT DEFAULT 'day',  -- year, month, day, hour
    is_relative BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(target_id, target_type)
);

CREATE INDEX idx_temporal_features_target ON temporal_features(target_id, target_type);
CREATE INDEX idx_temporal_features_dates ON temporal_features(start_date, end_date);

-- Promotion scores (for deciding what gets manifold treatment)
CREATE TABLE promotion_scores (
    id BIGSERIAL PRIMARY KEY,
    target_id TEXT NOT NULL,
    target_type TEXT NOT NULL,
    importance FLOAT DEFAULT 0.5,
    retrieval_frequency FLOAT DEFAULT 0.0,
    source_diversity FLOAT DEFAULT 0.0,
    confidence FLOAT DEFAULT 0.5,
    novelty FLOAT DEFAULT 0.5,
    graph_centrality FLOAT DEFAULT 0.0,
    user_relevance FLOAT DEFAULT 0.0,
    total_score FLOAT GENERATED ALWAYS AS (
        0.25 * importance +
        0.20 * retrieval_frequency +
        0.15 * source_diversity +
        0.15 * confidence +
        0.10 * novelty +
        0.10 * graph_centrality +
        0.05 * user_relevance
    ) STORED,
    promotion_status TEXT DEFAULT 'raw',  -- raw, provisional, promoted, manifold
    computed_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(target_id, target_type)
);

CREATE INDEX idx_promotion_scores_status ON promotion_scores(promotion_status);
CREATE INDEX idx_promotion_scores_total ON promotion_scores(total_score DESC);

-- Query logs (for retrieval frequency tracking)
CREATE TABLE query_logs (
    id BIGSERIAL PRIMARY KEY,
    query_text TEXT NOT NULL,
    query_hash TEXT NOT NULL,
    query_mode TEXT,
    tenant_ids TEXT[],
    result_ids TEXT[],
    result_scores FLOAT[],
    latency_ms INTEGER,
    manifold_weights JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_query_logs_hash ON query_logs(query_hash);
CREATE INDEX idx_query_logs_time ON query_logs(created_at DESC);

-- Manifold configuration (tunable per tenant)
CREATE TABLE manifold_config (
    id SERIAL PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    config_key TEXT NOT NULL,
    config_value JSONB NOT NULL,
    description TEXT,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(tenant_id, config_key)
);

-- Shadow comparison results (for A/B testing)
CREATE TABLE shadow_comparisons (
    id BIGSERIAL PRIMARY KEY,
    query_hash TEXT NOT NULL,
    query_text TEXT,
    old_result_ids TEXT[],
    old_scores FLOAT[],
    new_result_ids TEXT[],
    new_scores FLOAT[],
    overlap_count INTEGER,
    rank_correlation FLOAT,
    latency_old_ms INTEGER,
    latency_new_ms INTEGER,
    winner TEXT,  -- 'old', 'new', 'tie', 'unknown'
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_shadow_comparisons_time ON shadow_comparisons(created_at DESC);
```

### 5.2 Migration File Location

The migration SQL is stored at:
```
/opt/gami/manifold/migrations/002_manifold_tables.sql
```

**This migration is NOT in the Alembic chain.** It will only be applied when explicitly run during activation.

---

## 6. MCP Tool Changes

### 6.1 New Tools

| Tool | Description |
|------|-------------|
| `memory_recall_v2` | Manifold-aware recall with α weights |
| `memory_search_v2` | Manifold-aware search with mode selection |
| `manifold_stats` | Admin tool showing manifold coverage, embedding counts |
| `query_explain` | Debug tool showing manifold weights and scoring breakdown |
| `promote_object` | Manually promote an object to manifold treatment |
| `canonicalize_claim` | Convert a prose claim to SPO form |

### 6.2 Tool Definitions

```python
MANIFOLD_TOOLS = {
    "memory_recall_v2": {
        "name": "memory_recall_v2",
        "description": (
            "Manifold-aware memory recall. Uses multi-manifold embeddings and "
            "query-conditioned weights for better retrieval. Falls back to v1 "
            "behavior if manifold embeddings not available for a tenant."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Query text"},
                "tenant_id": {"type": "string", "default": "example-tenant"},
                "tenant_ids": {"type": "array", "items": {"type": "string"}},
                "max_tokens": {"type": "integer", "default": 2000},
                "mode": {
                    "type": "string",
                    "enum": ["auto", "fact_lookup", "synthesis", "timeline", 
                             "procedure", "comparison", "verification", "report"],
                    "default": "auto",
                    "description": "Query mode (auto uses classifier)"
                },
                "manifold_override": {
                    "type": "object",
                    "description": "Manual manifold weight overrides",
                    "properties": {
                        "topic": {"type": "number"},
                        "claim": {"type": "number"},
                        "procedure": {"type": "number"},
                        "relation": {"type": "number"},
                        "time": {"type": "number"},
                        "evidence": {"type": "number"},
                    }
                },
                "explain": {
                    "type": "boolean",
                    "default": False,
                    "description": "Include scoring breakdown in response"
                }
            },
            "required": ["query"],
        },
    },
    "manifold_stats": {
        "name": "manifold_stats",
        "description": "Get statistics on manifold embedding coverage and quality.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "tenant_id": {"type": "string"},
            },
        },
    },
    "query_explain": {
        "name": "query_explain",
        "description": (
            "Explain how a query would be processed: classification, manifold "
            "weights, candidate sources, and scoring factors. For debugging."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "tenant_id": {"type": "string", "default": "example-tenant"},
            },
            "required": ["query"],
        },
    },
    "promote_object": {
        "name": "promote_object",
        "description": "Promote an object to manifold treatment (generate canonical form and embeddings).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "target_id": {"type": "string"},
                "target_type": {"type": "string", "enum": ["claim", "segment", "entity", "memory"]},
                "manifolds": {"type": "array", "items": {"type": "string"}, "default": ["topic"]},
            },
            "required": ["target_id", "target_type"],
        },
    },
}
```

### 6.3 Backward Compatibility

**All existing v1 tools remain unchanged.** The v2 tools are additive. If v2 is called but manifold data doesn't exist for a tenant, it falls back to v1 behavior with a warning in the response.

---

## 7. Migration Strategy

### 7.1 Parallel Operation

```
┌─────────────────────────────────────────────────────────────────┐
│                    Migration Timeline                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Phase 1: Build (current)                                        │
│  ├── Create manifold/ directory with all code                   │
│  ├── Push to feature/multi-manifold branch                      │
│  └── No database changes, no imports into current code          │
│                                                                  │
│  Phase 2: Schema (future)                                        │
│  ├── Run 002_manifold_tables.sql migration                      │
│  ├── Tables exist but are empty                                 │
│  └── Current system unaffected                                  │
│                                                                  │
│  Phase 3: Backfill Topic (future)                                │
│  ├── Add topic manifold embeddings for all segments             │
│  ├── Copy from existing embedding columns                       │
│  └── Current retrieval still uses old path                      │
│                                                                  │
│  Phase 4: Canonical Forms (future)                               │
│  ├── Generate canonical claims from existing claims             │
│  ├── Generate canonical procedures from instructional segments  │
│  └── Embed claim/procedure manifolds for promoted objects       │
│                                                                  │
│  Phase 5: Shadow Mode (future)                                   │
│  ├── Enable shadow comparison in retrieval                      │
│  ├── Both paths run, new results logged but not returned        │
│  └── Analyze quality metrics                                    │
│                                                                  │
│  Phase 6: Gradual Cutover (future)                               │
│  ├── Enable v2 retrieval for specific tenants                   │
│  ├── Monitor latency and quality                                │
│  └── Full cutover when confident                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 Backfill Scripts

```python
# migrate_topic_embeddings.py
# Copies existing embeddings to manifold_embeddings table with manifold_type='topic'

# migrate_canonical_claims.py
# Uses vLLM to convert prose claims to SPO form, stores in canonical_claims

# migrate_procedures.py
# Identifies instructional segments, extracts procedures, stores canonical form

# compute_promotion_scores.py
# Calculates promotion scores for all objects, populates promotion_scores table

# embed_promoted.py
# Generates claim/procedure embeddings for objects above promotion threshold
```

### 7.3 Rollback Points

Every migration step has a rollback:

| Step | Rollback |
|------|----------|
| Create tables | DROP TABLE IF EXISTS ... |
| Backfill topic | DELETE FROM manifold_embeddings WHERE manifold_type='topic' |
| Generate canonical claims | DELETE FROM canonical_claims |
| Enable shadow mode | Set MANIFOLD_SHADOW_MODE=false in config |
| Enable v2 retrieval | Set MANIFOLD_V2_ENABLED=false in config |

---

## 8. Neo4j Migration Path

### 8.1 Current State (AGE)

Apache AGE is integrated into PostgreSQL 5433. Graph queries use a hybrid SQL + Cypher syntax:

```sql
SELECT * FROM cypher('manifold_graph', $$
    MATCH (e:Entity)-[:MENTIONS]->(s:Segment)
    WHERE e.id = $1
    RETURN s
$$, $2) AS (segment agtype)
```

### 8.2 Trigger Conditions for Neo4j

Move to Neo4j only if one of these becomes true:

1. **Vector-native graph retrieval needed** — Want to search vectors scoped to graph neighborhoods
2. **Graph algorithms needed** — PageRank, community detection, node similarity as retrieval signals
3. **AGE becomes limiting** — Cypher expressiveness, performance, or operational complexity

### 8.3 Neo4j Migration Blueprint

**Infrastructure**:
```yaml
# docker-compose.neo4j.yml
services:
  neo4j:
    image: neo4j:5.18-enterprise
    ports:
      - "7474:7474"  # HTTP
      - "7687:7687"  # Bolt
    volumes:
      - /opt/gami/neo4j/data:/data
      - /opt/gami/neo4j/logs:/logs
    environment:
      NEO4J_AUTH: neo4j/GamiGraph2026!
      NEO4J_PLUGINS: '["graph-data-science"]'
      NEO4J_dbms_memory_heap_max__size: 4G
      NEO4J_dbms_memory_pagecache_size: 2G
```

**Data Export** (AGE → intermediate format):
```python
# scripts/export_age_graph.py
# Exports all nodes and edges from AGE to JSONL files
```

**Data Import** (intermediate → Neo4j):
```python
# scripts/import_neo4j.py
# Loads JSONL into Neo4j using LOAD CSV or Cypher UNWIND
```

**Sync Mechanism**:
```python
# workers/neo4j_sync_worker.py
# Transactional outbox pattern:
# 1. Write to PG outbox table
# 2. Worker reads outbox, writes to Neo4j
# 3. Mark outbox entry complete
```

**Driver Integration**:
```python
# api/graph/neo4j_driver.py
from neo4j import AsyncGraphDatabase

class Neo4jDriver:
    def __init__(self, uri, auth):
        self.driver = AsyncGraphDatabase.driver(uri, auth=auth)
    
    async def expand_neighborhood(self, entity_id, depth=2, limit=50):
        async with self.driver.session() as session:
            result = await session.run("""
                MATCH (e:Entity {id: $entity_id})-[r*1..$depth]-(n)
                RETURN e, r, n
                LIMIT $limit
            """, entity_id=entity_id, depth=depth, limit=limit)
            return [record async for record in result]
```

### 8.4 Neo4j is NOT Implemented in This Branch

The Neo4j code in this branch is **blueprints and stubs only**. No actual Neo4j connections are made. The AGE integration continues to be the active graph engine.

---

## 9. Safeguards and Invariants

### 9.1 Invariants That Must Never Break

1. **Raw sources preserved** — Never delete sources because summaries/claims exist
2. **Provenance traceable** — Every promoted object links to source evidence
3. **Relational = truth** — PostgreSQL is authoritative for lineage, jobs, audit
4. **Graph = derived** — Graph mirrors relational, can be rebuilt
5. **Redis = cache** — Never sole storage for facts
6. **Anchor-first retrieval** — Never blind graph walk without semantic anchors
7. **Contradictions preserved** — Never silently merge conflicting claims
8. **Abstention on low confidence** — Return uncertainty, don't fabricate

### 9.2 Code Isolation

The manifold/ directory:
- **Does NOT import from** api/, scripts/, workers/, mcp_tools/
- **Is NOT imported by** api/, scripts/, workers/, mcp_tools/
- Has its own `__init__.py` that is empty
- Has a README.md stating "NOT FOR PRODUCTION USE"

### 9.3 Database Isolation

- Migration file exists but is not in Alembic versions/
- Tables use prefix `manifold_` or are clearly new names
- No modifications to existing table schemas
- All new tables have `IF NOT EXISTS` safety

### 9.4 Runtime Isolation

- No new systemd services
- No modifications to existing services
- No entries in cron
- No modifications to .env

### 9.5 Feature Flags

When activated (future), these flags control behavior:

```python
# In api/config.py (future addition)
MANIFOLD_ENABLED = os.getenv("GAMI_MANIFOLD_ENABLED", "false").lower() == "true"
MANIFOLD_SHADOW_MODE = os.getenv("GAMI_MANIFOLD_SHADOW", "false").lower() == "true"
MANIFOLD_V2_ENABLED = os.getenv("GAMI_MANIFOLD_V2", "false").lower() == "true"
```

---

## 10. Implementation Phases

### Phase 0: Foundation (This PR)
- [x] Create manifold/ directory structure
- [x] Write ROADMAP.md (this document)
- [x] Create database model files (not run)
- [x] Create canonical form generators
- [x] Create query classifier v2
- [x] Create manifold embedder
- [x] Create fusion and retrieval modules
- [x] Create shadow comparison module
- [x] Create migration scripts (not run)
- [x] Create MCP tool definitions
- [x] Create Neo4j blueprint (stubs only)
- [x] Push to feature/multi-manifold branch

### Phase 1: Schema (Future)
- [ ] Review and approve migration SQL
- [ ] Run migration in dev environment
- [ ] Verify no impact on existing queries
- [ ] Run migration in production

### Phase 2: Topic Backfill (Future)
- [ ] Run migrate_topic_embeddings.py
- [ ] Verify counts match
- [ ] Benchmark retrieval latency

### Phase 3: Canonical Forms (Future)
- [ ] Run migrate_canonical_claims.py (batch, rate-limited)
- [ ] Review sample canonical claims
- [ ] Run migrate_procedures.py
- [ ] Review sample procedures

### Phase 4: Promotion Scoring (Future)
- [ ] Run compute_promotion_scores.py
- [ ] Review score distribution
- [ ] Set promotion thresholds

### Phase 5: Manifold Embeddings (Future)
- [ ] Run embed_promoted.py for claim manifold
- [ ] Run embed_promoted.py for procedure manifold
- [ ] Verify embedding counts

### Phase 6: Shadow Mode (Future)
- [ ] Enable MANIFOLD_SHADOW_MODE
- [ ] Collect comparison data for 1 week
- [ ] Analyze rank correlation, overlap, latency
- [ ] Decide on cutover

### Phase 7: V2 Activation (Future)
- [ ] Enable MANIFOLD_V2_ENABLED for test tenant
- [ ] Monitor for 1 week
- [ ] Enable for all tenants
- [ ] Deprecate v1 tools (keep working, log warning)

---

## 11. Testing Strategy

### 11.1 Unit Tests (In This Branch)

```
manifold/tests/
├── test_canonical_forms.py      # SPO parsing, procedure extraction
├── test_query_classifier.py     # Mode detection, α computation
├── test_manifold_fusion.py      # Score fusion math
├── test_promotion_scoring.py    # Promotion formula
└── test_time_features.py        # Temporal feature extraction
```

### 11.2 Integration Tests (Future)

Require database connection, run after schema migration:
- `test_manifold_embeddings.py` — Store and retrieve embeddings
- `test_shadow_comparison.py` — Run both paths, compare
- `test_end_to_end_v2.py` — Full recall_v2 pipeline

### 11.3 Golden Query Set

20 test queries with expected behaviors:

1. "What's the root password for GitLab?" → Credential, claim manifold high
2. "What caused the edge failover flapping?" → Timeline, time manifold high
3. "How is Stargate connected to Walter?" → Entity, relation manifold high
4. "How do I deploy GAMI?" → Procedure, procedure manifold high
5. "Did we discuss this before?" → Memory, assistant_memory mode
... (15 more in test file)

---

## 12. Rollback Procedures

### 12.1 Schema Rollback

```sql
-- rollback_002_manifold_tables.sql
DROP TABLE IF EXISTS shadow_comparisons;
DROP TABLE IF EXISTS manifold_config;
DROP TABLE IF EXISTS query_logs;
DROP TABLE IF EXISTS promotion_scores;
DROP TABLE IF EXISTS temporal_features;
DROP TABLE IF EXISTS canonical_procedures;
DROP TABLE IF EXISTS canonical_claims;
DROP TABLE IF EXISTS manifold_embeddings;
```

### 12.2 Code Rollback

```bash
# Remove manifold imports from api/config.py (if added)
# Set all feature flags to false
# Restart services
```

### 12.3 Full Rollback

```bash
# 1. Disable feature flags
export GAMI_MANIFOLD_ENABLED=false
export GAMI_MANIFOLD_SHADOW=false
export GAMI_MANIFOLD_V2=false

# 2. Restart services
sudo systemctl restart gami-api gami-mcp

# 3. Drop tables (optional, can leave dormant)
psql -p 5433 -U gami -d gami -f rollback_002_manifold_tables.sql

# 4. Remove code (optional, can leave on feature branch)
git checkout main
```

---

## Appendix A: File Structure

```
/opt/gami/manifold/
├── README.md                    # "NOT FOR PRODUCTION USE"
├── ROADMAP.md                   # This document
├── __init__.py                  # Empty, no exports
│
├── models/
│   ├── __init__.py
│   ├── manifold_models.py       # SQLAlchemy models
│   └── schemas.py               # Pydantic schemas
│
├── canonical/
│   ├── __init__.py
│   ├── claim_normalizer.py      # Prose → SPO
│   ├── procedure_normalizer.py  # Prose → steps
│   ├── temporal_extractor.py    # Date/time → features
│   └── forms.py                 # Canonical form templates
│
├── embeddings/
│   ├── __init__.py
│   ├── manifold_embedder.py     # Multi-manifold embedding
│   └── promotion.py             # Promotion scoring
│
├── retrieval/
│   ├── __init__.py
│   ├── query_classifier_v2.py   # Enhanced classifier with α
│   ├── manifold_fusion.py       # Score fusion
│   ├── anchor_retrieval.py      # Manifold-aware anchors
│   └── shadow_runner.py         # A/B comparison
│
├── graph/
│   ├── __init__.py
│   ├── neo4j_blueprint.py       # Neo4j stubs (not connected)
│   └── expansion.py             # Graph expansion (uses AGE)
│
├── migrations/
│   ├── 002_manifold_tables.sql  # Schema (not in Alembic)
│   └── rollback_002.sql         # Rollback script
│
├── scripts/
│   ├── migrate_topic_embeddings.py
│   ├── migrate_canonical_claims.py
│   ├── migrate_procedures.py
│   ├── compute_promotion_scores.py
│   └── embed_promoted.py
│
├── mcp/
│   ├── __init__.py
│   └── tool_definitions_v2.py   # New MCP tools
│
└── tests/
    ├── __init__.py
    ├── test_canonical_forms.py
    ├── test_query_classifier.py
    ├── test_manifold_fusion.py
    ├── test_promotion_scoring.py
    └── test_time_features.py
```

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **Manifold** | A semantic embedding space tuned for a specific type of similarity |
| **Anchor** | A high-scoring candidate retrieved before graph expansion |
| **α (alpha)** | Query-conditioned manifold weight vector |
| **Canonical form** | Structured representation of a claim/procedure for embedding |
| **Promotion** | Upgrading an object to receive richer manifold treatment |
| **Shadow mode** | Running new retrieval alongside old without affecting output |
| **SPO** | Subject-Predicate-Object triple structure for claims |

---

*End of Roadmap*
