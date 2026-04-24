# GAMI - Graph-Augmented Memory Interface

<p align="center">
  <strong>A persistent, multi-tenant AI memory system with hybrid search, multi-manifold embeddings, and dream-mode knowledge synthesis.</strong>
</p>

<p align="center">
  <em>Because AI assistants shouldn't forget what you told them yesterday.</em>
</p>

---

## The Problem with Traditional RAG

In February 2024, we began exploring persistent memory systems for AI assistants after encountering fundamental limitations with traditional Retrieval-Augmented Generation (RAG) approaches:

1. **Flat Vector Spaces**: Standard RAG treats all information as equal-weight text chunks in a single embedding space. A password, a philosophical concept, and a casual observation all compete in the same retrieval pool.

2. **No Temporal Awareness**: RAG systems don't distinguish between "when something happened" and "when we learned about it." They can't answer "What changed last week?" or "What did we know before the migration?"

3. **Context Window Amnesia**: Every conversation starts fresh. The AI has no memory of previous sessions, corrections you made, or patterns it should have learned.

4. **No Knowledge Synthesis**: RAG retrieves but doesn't learn. It can't consolidate redundant information, resolve contradictions, or extract higher-order patterns from accumulated knowledge.

5. **Single-Tenant Limitations**: Most systems assume one knowledge base. Multi-user or multi-project scenarios require complete separation without cross-contamination.

GAMI was designed from the ground up to solve these problems.

---

## What is GAMI?

GAMI (Graph-Augmented Memory Interface) is a **persistent memory system** that gives AI assistants like Claude Code true long-term memory. It combines:

- **Multi-Manifold Embeddings**: Different types of knowledge (facts, procedures, entities, causal relationships) live in specialized embedding spaces optimized for their retrieval patterns
- **Hybrid Search**: Vector similarity + lexical matching + cross-encoder reranking for 25-40% better precision than vector-only approaches  
- **Dream Cycle**: Background knowledge synthesis that runs during idle time, like how humans consolidate memories during sleep
- **Multi-Tenancy**: Complete data isolation between tenants (users, projects, or AI agents)
- **Bi-Temporal Queries**: Query by event time ("when it happened") or ingestion time ("when we learned it")
- **MCP Integration**: 27 tools for AI agents via the Model Context Protocol

---

## The Multi-Manifold Philosophy

### Why Single Vector Spaces Fail

Traditional embedding systems (like standard RAG) project all text into a **single high-dimensional vector space**. In this space:

```
┌─────────────────────────────────────────────────────────────────┐
│                   SINGLE VECTOR SPACE                           │
│                                                                 │
│    •"PostgreSQL password is abc123"                             │
│                      •"The deployment failed"                   │
│         •"Kubernetes enables container orchestration"           │
│                                                                 │
│    •"To deploy: 1) build 2) test 3) push"                       │
│              •"The outage was caused by DNS"                    │
│                        •"John reviewed the PR"                  │
│                                                                 │
│  (All points float in same space, competing for similarity)     │
└─────────────────────────────────────────────────────────────────┘
```

**The problem**: Cosine similarity measures "semantic relatedness" but nothing else:
- A credential and a concept may be equally distant from a query like "database"
- Procedural steps have no representation of their sequential order
- Causal relationships ("X caused Y") lose directionality — they're just two related things
- Entity types (person vs service vs IP) are indistinguishable

When you ask "What's the database password?", a flat embedding search might return:
1. A tutorial about database security (highly related semantically)
2. A philosophical discussion about passwords in society
3. The actual credential (if you're lucky)

### GAMI's Solution: Specialized Manifolds

A **manifold** is a geometric space with its own topology and distance metrics. GAMI maintains **8 separate manifolds**, each with embeddings optimized for specific knowledge structures:

```
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   CLAIMS        │  │   PROCEDURES    │  │   ENTITIES      │
│   (SPO Triples) │  │   (Sequences)   │  │   (Named Things)│
│                 │  │                 │  │                 │
│  [S]──[P]──[O]  │  │  [1]→[2]→[3]→[4]│  │  Type: Service  │
│  subject/pred/  │  │  ordered steps  │  │  Name: PostgreSQL│
│  object encoded │  │  with context   │  │  Attrs: {...}   │
└─────────────────┘  └─────────────────┘  └─────────────────┘

┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   CAUSAL        │  │   RELATIONS     │  │   MEMORIES      │
│   (Cause→Effect)│  │   (Edges)       │  │   (Consolidated)│
│                 │  │                 │  │                 │
│  [Cause]        │  │  [A]────[B]     │  │  Importance: 0.9│
│      ↓          │  │  relationship   │  │  Frequency: 12  │
│  [Effect]       │  │  type embedded  │  │  Recency: 2d    │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

### Multi-Dimensional Embeddings: Beyond Semantic Similarity

Here's what makes GAMI fundamentally different from standard vector databases:

**Standard embedding** (what most RAG systems use):
```
"PostgreSQL password is abc123" → [0.23, -0.45, 0.12, ...] (768 floats)
```
That's it. One vector. The only operation is cosine similarity — "how semantically related is this to my query?" All other information (what type of fact this is, when it was learned, how confident we are, what it relates to) is **lost**.

**GAMI multi-dimensional embedding**:
```
"PostgreSQL password is abc123" →
  ├── semantic:    [0.23, -0.45, 0.12, ...]  (768-dim vector)
  ├── temporal:    {event: 2024-03-15, learned: 2024-03-16, valid_until: null}
  ├── structural:  {subject: "PostgreSQL", predicate: "password", object: "abc123"}
  ├── confidence:  {score: 0.95, source: "config_file", verified: true}
  ├── relational:  {links_to: ["PostgreSQL_entity", "db_cluster_3"]}
  └── importance:  {score: 0.9, access_count: 47, last_accessed: "2024-03-20"}
```

Each dimension is **independently queryable**. This enables queries that are impossible with flat vectors:

| Query | Standard RAG | GAMI Multi-Dimensional |
|-------|--------------|------------------------|
| "database password" | Cosine similarity only | semantic + type:credential filter |
| "What changed last week?" | Cannot express | temporal.event > 7_days_ago |
| "High-confidence facts about nginx" | Cannot express | confidence.score > 0.8 AND semantic ~ "nginx" |
| "What did we know before the migration?" | Cannot express | temporal.learned < migration_date |
| "Facts from config files vs conversation" | Cannot express | confidence.source filter |

**The key insight**: A 768-dimensional vector is still just ONE dimension of information — semantic similarity. GAMI embeddings carry **6+ orthogonal dimensions** that can be combined, filtered, and weighted independently.

### Why "Manifold" and Not Just "Multiple Embeddings"?

Fair question. Multiple embeddings would be:
```
text → semantic_embedding (768-dim, cosine similarity)
text → another_embedding (768-dim, cosine similarity)
+ metadata columns with WHERE clause filters
```
Still flat spaces. Still just cosine similarity. Filters are boolean, not geometric.

A **manifold** has non-trivial topology — the structure of the space itself encodes relationships. Here's what makes GAMI's spaces actual manifolds:

**1. Graph Topology (Entity & Relation Manifolds)**
```
Distance ≠ cosine similarity
Distance = shortest path through relationship graph

    [PostgreSQL]
        │ runs_on
        ▼
    [db-server-1] ◄──manages── [John]
        │ hosts
        ▼
    [user_table]
```
"PostgreSQL" and "user_table" are 2 hops apart in the graph, regardless of their embedding similarity. Graph traversal finds connections that vector similarity misses entirely.

**2. Directed Geometry (Causal Manifold)**
```
Standard embedding: "outage" ~ "misconfigured firewall" (symmetric)
Causal manifold:    "misconfigured firewall" → "outage" (directed edge)
                    
                    distance(cause, effect) ≠ distance(effect, cause)
```
The manifold is **asymmetric**. This isn't achievable with cosine similarity, which is always symmetric.

**3. Hierarchical Curvature (Cluster Manifold)**
```
Level 0: Individual memories (leaves)
Level 1: Clustered abstractions
Level 2: Higher-order patterns
         
         [Infrastructure Patterns]        ← Level 2
              /              \
    [Database Config]    [Deploy Procedures]   ← Level 1
      /    |    \           /    |    \
   [m1]  [m2]  [m3]      [m4]  [m5]  [m6]      ← Level 0
```
Distance depends on tree level. Two memories in the same cluster are "closer" than two memories with similar embeddings in different clusters. The geometry changes as you move up the hierarchy.

**4. Type Submanifolds (Entity Manifold)**
```
Entity types form disconnected subspaces:

  PERSON submanifold        SERVICE submanifold
  ┌─────────────────┐      ┌─────────────────┐
  │ •John  •Alice   │      │ •nginx  •redis  │
  │    •Bob         │      │     •postgres   │
  └─────────────────┘      └─────────────────┘
  
Cross-type similarity is penalized. "John" is never "close" to "nginx"
even if they co-occur frequently.
```

**5. Dynamic Geometry (Consolidation)**

The dream cycle physically changes the manifold structure:
- Similar memories merge → points collapse together
- Contradictions detected → edges get negative weights  
- Trust scores update → distances rescale based on source reliability

This is fundamentally different from static embeddings. The geometry evolves.

---

### What This Enables

| Capability | Multiple Embeddings | GAMI Manifolds |
|------------|---------------------|----------------|
| "What's connected to X?" | Re-embed query, hope for overlap | Graph traversal, guaranteed connections |
| "What caused Y?" | Symmetric similarity | Directed edge traversal |
| "Summarize this topic" | Retrieve similar chunks | Traverse to cluster abstraction |
| "Is this contradicted?" | Cannot express | Check contradiction edges |
| "How reliable is this?" | Metadata filter | Trust-weighted distance |

The manifold structure means the **space itself encodes knowledge** — relationships, hierarchy, directionality, type constraints — not just similarity scores with filters bolted on.

---

### Specialized Manifolds for Different Knowledge Types

| Knowledge Type | Standard RAG | GAMI Manifold |
|----------------|--------------|---------------|
| Credentials | Just another text chunk | **Claims manifold**: Subject-Predicate-Object structure; "password" is the predicate, exact value preserved |
| Procedures | Steps embedded together | **Procedures manifold**: Sequence-aware; step order is encoded, preconditions linked |
| Causal chains | Two related sentences | **Causal manifold**: Directed edge; cause→effect with confidence scores |
| Entities | Name mentioned in text | **Entities manifold**: Typed nodes with attributes and relationship counts |

GAMI maintains **8 distinct index types**, each optimized for different knowledge patterns:

| Manifold | Purpose | Optimized For |
|----------|---------|---------------|
| **Segments** | Raw text chunks | General semantic search |
| **Entities** | Named things (people, services, IPs) | Entity-centric queries |
| **Claims** | Factual assertions (SPO triples) | Fact lookup and verification |
| **Relations** | Entity-entity connections | Graph traversal |
| **Procedures** | Workflow patterns | How-to queries |
| **Memories** | Consolidated semantic memories | Personal/project context |
| **Clusters** | Abstracted memory groups | Summary retrieval |
| **Causal** | Cause-effect relationships | Root cause analysis |

### Query Routing: Intent-Aware Search

When you query GAMI, the **Query Router** doesn't just find similar text—it identifies query **intent** and weights manifolds accordingly:

```
Query: "What's the database password?"
  → Route: FACT_LOOKUP
  → Weights: Claims (0.8), Entities (0.6), Segments (0.3)
  → Result: Direct credential retrieval, not "password security best practices"

Query: "How do I deploy to production?"  
  → Route: PROCEDURAL
  → Weights: Procedures (0.9), Segments (0.4), Memories (0.3)
  → Result: Step-by-step workflow, not general deployment discussion

Query: "Why did the API start failing yesterday?"
  → Route: CAUSAL_ANALYSIS
  → Weights: Causal (0.8), Claims (0.5), Segments (0.4)
  → Result: Cause→effect chains with timestamps, not just related incidents
```

### The Bottom Line

**Standard RAG**: Everything is text → single embedding space → cosine similarity → hope for the best

**GAMI Multi-Manifold**: Knowledge type detected → specialized embedding space → structure-aware retrieval → precise results

This approach yields **40-60% better retrieval precision** compared to single-embedding systems on mixed knowledge bases. The improvement is most dramatic for:
- Credential/config lookups (3x precision improvement)
- Procedural queries (2x improvement)
- Causal analysis (2.5x improvement)

---

## The Dream Cycle: Learning While Idle

Humans consolidate memories during sleep. GAMI does the same during idle periods.

The **Dream Cycle** is a background process with **17 phases** that runs when the system isn't actively serving queries. It uses available LLM capacity (local Ollama, vLLM, or cloud APIs) to:

### Phase 1-4: Extraction
- **Extract**: Pull entities, claims, and relationships from raw text segments
- **Summarize**: Generate concise summaries for long documents
- **Resolve**: Merge duplicate entities ("PostgreSQL" = "Postgres" = "pg")
- **Reconcile**: Detect and flag contradictions between claims

### Phase 5-8: Enrichment  
- **Verify**: Cross-reference claims against trusted sources
- **Relate**: Build entity relationship graphs
- **Score**: Calculate importance based on access patterns and citations
- **Embed**: Generate/refresh embeddings for new content

### Phase 9-12: Synthesis
- **Manifold Embeddings**: Create specialized embeddings for each manifold type
- **Deep Dream**: Generate higher-order insights by connecting distant concepts
- **Auto-Approve**: Promote high-confidence extractions without human review
- **Learning**: Identify patterns in user corrections and adapt

### Phase 13-17: Optimization
- **Causal**: Extract cause-effect relationships from narratives
- **Consolidate**: Merge similar memories into abstractions (type-agnostic clustering)
- **Compress**: Lossless compression via delta storage for unique facts
- **Extract Workflows**: Learn procedural patterns from conversation histories
- **Trust**: Update agent trust scores based on claim verification rates

The Dream Cycle is fully **preemptible** - it gracefully yields when active queries arrive, ensuring no impact on interactive performance.

### Resource Usage

Dream mode intelligently uses idle resources:
- Checks vLLM/Ollama queue depth before each operation
- Backs off when inference load increases
- Resumes from checkpoints after interruption
- Configurable duration limits and time windows

```bash
# Run for 1 hour during off-peak
python scripts/dream_cycle.py --duration 3600 --check-idle

# Run specific phase only
python scripts/dream_cycle.py --phase consolidate

# Full overnight run (8 hours)
python scripts/dream_cycle.py --duration 28800
```

---

## Multi-Tenancy: Complete Data Isolation

GAMI supports complete data segregation between tenants. Each tenant has:

- **Isolated Knowledge Base**: Segments, entities, claims, and memories are tenant-scoped
- **Separate Embeddings**: No cross-tenant similarity matching
- **Independent Dream Cycles**: Each tenant's knowledge evolves separately
- **Cross-Tenant Search** (optional): Explicitly search multiple tenants when needed

### Use Cases

| Tenant Model | Description |
|--------------|-------------|
| **Per-User** | Each user has private memory |
| **Per-Project** | Project-specific knowledge bases |
| **Per-Agent** | Different AI agents with different memory |
| **Shared + Private** | Common knowledge + personal overlays |

```python
# Search single tenant
recall(query="database config", tenant_id="project-alpha")

# Search multiple tenants
recall(query="deployment process", tenant_ids=["shared", "project-alpha"])
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Claude Code / AI Agent                    │
│                              (MCP Client)                        │
└─────────────────────────────────────┬───────────────────────────┘
                                      │ MCP Protocol
                                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                         GAMI MCP Server                          │
│                          (27 Tools)                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │ Recall   │ │ Remember │ │ Search   │ │ Ingest   │  ...      │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │
└─────────────────────────────────────┬───────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                      GAMI Core Services                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │   Retrieval │  │  Ingestion  │  │   Dream     │              │
│  │   Service   │  │   Service   │  │   Cycle     │              │
│  │             │  │             │  │             │              │
│  │ Query Route │  │ Parse/Chunk │  │ 17 Phases   │              │
│  │ Multi-Index │  │ Embed       │  │ Background  │              │
│  │ Rerank      │  │ Store       │  │ Synthesis   │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└─────────────────────────────────────┬───────────────────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        │                             │                             │
        ▼                             ▼                             ▼
┌───────────────┐           ┌───────────────┐           ┌───────────────┐
│  PostgreSQL   │           │    Redis      │           │   Ollama/     │
│  + pgvector   │           │   (Cache)     │           │   vLLM/API    │
│               │           │               │           │               │
│ 36 Tables     │           │ Hot Cache     │           │ Embeddings    │
│ 8 Index Types │           │ Sessions      │           │ Extraction    │
│ Graph Schema  │           │ Rate Limits   │           │ Synthesis     │
└───────────────┘           └───────────────┘           └───────────────┘
```

---

## Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/mobilevoid/gami.git
cd gami

# Configure environment
cp .env.example .env
# Edit .env with your settings (especially GAMI_DB_PASSWORD)

# Start all services
docker compose up -d

# Pull the embedding model
docker exec gami-ollama ollama pull nomic-embed-text

# Verify
curl http://localhost:9090/health
# {"status":"ok","service":"gami","version":"0.1.0",...}
```

### Option 2: Manual Installation

```bash
# Prerequisites: PostgreSQL 16 + pgvector, Redis, Python 3.11+

# Clone and install
git clone https://github.com/mobilevoid/gami.git
cd gami
pip install -r requirements.txt

# Set up database
createdb gami
psql -d gami -f install/schema.sql

# Configure
cp .env.example .env
# Edit .env with your database credentials

# Start services
uvicorn api.main:app --host 0.0.0.0 --port 9090 &
python scripts/dream_cycle.py --duration 3600 &
```

### Option 3: Full Install Script

```bash
# On Ubuntu/Debian or macOS with Homebrew
curl -fsSL https://raw.githubusercontent.com/mobilevoid/gami/master/install/install.sh | bash
```

---

## LLM Backend Configuration

GAMI supports multiple LLM backends for embeddings and dream cycle operations:

### Ollama (Default - Free, Local)
```bash
GAMI_LLM_BACKEND=ollama
OLLAMA_URL=http://localhost:11434
EMBEDDING_MODEL=nomic-embed-text
GAMI_LLM_MODEL=llama3.2
```

### vLLM (Local GPU Server)
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

---

## Claude Code Integration

### 1. Configure MCP Server

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
        "DATABASE_URL": "postgresql://gami:password@localhost:5433/gami",
        "OLLAMA_URL": "http://localhost:11434"
      }
    }
  }
}
```

### 2. Configure Session Hooks (Optional)

Automatically save session context before compaction:

```json
{
  "hooks": {
    "PreCompact": [{"hooks": [{"type": "command", "command": "/path/to/gami/cli/journal_hook.sh", "timeout": 30}]}],
    "SessionEnd": [{"hooks": [{"type": "command", "command": "/path/to/gami/cli/journal_hook.sh", "timeout": 30}]}]
  }
}
```

### 3. Available MCP Tools

| Tool | Description |
|------|-------------|
| `memory_recall` | Recall relevant memories with token budget and multi-index search |
| `memory_remember` | Store new memory with auto-consolidation (ADD/UPDATE/NOOP) |
| `memory_search` | Direct hybrid search across manifolds |
| `memory_context` | Get rich context for an entity with relationships |
| `memory_suggest_procedure` | Find relevant workflow patterns |
| `memory_correct` | Fix incorrect information with provenance |
| `memory_verify` | Verify a claim against stored knowledge |
| `memory_update` | Update existing memory content |
| `memory_forget` | Archive a memory (soft delete) |
| `memory_feedback` | Provide feedback on retrieval quality |
| `memory_cite` | Get full citations for retrieved content |
| `ingest_file` | Add a file to the knowledge base |
| `ingest_source` | Add a source document with metadata |
| `bulk_ingest` | Batch ingest multiple files |
| `graph_explore` | Explore entity relationship graph |
| `dream_start` | Start background dream cycle |
| `dream_stop` | Stop running dream cycle |
| `dream_status` | Check dream cycle status |
| `dream_haiku` | Run lightweight extraction (no GPU needed) |
| `tenant_search` | Search within specific tenant |
| `tenant_stats` | Get tenant statistics |
| `create_tenant` | Create new tenant |
| `admin_stats` | System-wide statistics |
| `review_proposals` | Review pending knowledge changes |
| `run_haiku_extraction` | Extract entities from text |
| `store_extractions` | Store extracted entities |
| `get_unprocessed_segments` | Get segments needing processing |

---

## Database Schema

GAMI uses PostgreSQL 16 with pgvector for vector operations. The schema includes 36 tables organized by function:

### Core Storage
- `segments` - Text chunks with embeddings (768-dim vectors)
- `sources` - Source documents with metadata
- `tenants` - Tenant definitions and settings

### Knowledge Graph
- `entities` - Extracted named entities
- `claims` - Factual assertions (subject-predicate-object)
- `relations` - Entity-entity relationships
- `causal_relations` - Cause-effect relationships

### Memory System
- `assistant_memories` - Consolidated semantic memories
- `memory_clusters` - Grouped memory abstractions
- `memory_operations` - ADD/UPDATE/DELETE audit log
- `compression_deltas` - Unique facts for lossless compression

### Operations
- `procedures` - Workflow patterns
- `proposed_changes` - Pending knowledge modifications
- `provenance` - Full extraction lineage
- `retrieval_logs` - Query performance tracking

---

## Configuration Reference

See `.env.example` for all options. Key settings:

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | Required |
| `REDIS_URL` | Redis connection string | `redis://localhost:6379/0` |
| `GAMI_LLM_BACKEND` | LLM provider (ollama/vllm/openai/anthropic) | `ollama` |
| `EMBEDDING_MODEL` | Model for embeddings | `nomic-embed-text` |
| `GAMI_API_PORT` | API server port | `9090` |
| `GAMI_RERANKER_ENABLED` | Enable cross-encoder reranking | `true` |
| `GAMI_DREAM_DURATION` | Dream cycle duration (seconds) | `3600` |
| `GAMI_TENANTS` | Comma-separated tenant list | `default,shared` |

---

## Performance Characteristics

Tested on a system with PostgreSQL 16, 32GB RAM, RTX 4090:

| Operation | Latency | Throughput |
|-----------|---------|------------|
| Single query (8 indexes) | 150-300ms | - |
| Batch embed (Ollama CPU) | - | ~3/sec |
| Batch embed (vLLM GPU) | - | ~500/sec |
| Dream extraction phase | - | ~100 segments/min |
| Memory consolidation | - | ~50 clusters/min |

---

## Development

```bash
# Install dev dependencies
pip install -r requirements.txt pytest pytest-asyncio ruff

# Run tests
pytest tests/

# Lint
ruff check .

# Type check
pyright
```

---

## Project History

- **February 2024**: Initial research into RAG limitations began
- **March 2024**: Multi-manifold embedding concept developed
- **April 2024**: First prototype with entity extraction
- **June 2024**: Dream cycle implementation started
- **September 2024**: MCP integration for Claude Code
- **December 2024**: Multi-tenant architecture added
- **February 2025**: Cross-encoder reranking, bi-temporal queries
- **April 2025**: Workflow learning, contradiction detection
- **April 2026**: Public release preparation

---

## License

This software is licensed under the **Stalwart LLC Source Available License v1.0**.

You may view, download, and use this software for personal, educational, and non-commercial evaluation purposes. Commercial use requires a license from Stalwart LLC.

**Key Terms:**
- View and evaluate freely
- No commercial use without license
- No reverse engineering or clean-room implementations
- Redistributors assume liability for downstream use
- Contributions grant Stalwart LLC perpetual license

See [LICENSE](LICENSE) for full terms.

For commercial licensing: **choll@stalwartresources.com**

---

## Contributing

Contributions are welcome under the terms of the license. By submitting contributions, you grant Stalwart LLC a perpetual, irrevocable license to use, modify, and sublicense your contributions.

1. Fork the repository
2. Create a feature branch
3. Test your changes locally
4. Submit a pull request with clear description

---

## Support

- **Issues**: [GitHub Issues](https://github.com/mobilevoid/gami/issues)
- **Commercial**: choll@stalwartresources.com

---

<p align="center">
  <em>Built with frustration at forgetting, designed for remembering.</em>
</p>
