# GAMI Multi-Manifold Memory System

> **WARNING: NOT FOR PRODUCTION USE**
>
> This directory contains experimental code for the multi-manifold memory system.
> It is completely isolated from the running GAMI system.
>
> - No imports from this directory into api/, scripts/, workers/, or mcp_tools/
> - No database migrations have been run
> - No services are running this code
> - This exists on the `feature/multi-manifold` branch only

## What This Is

A replacement for GAMI's single-embedding retrieval with query-conditioned multi-manifold retrieval:

- **Topic manifold**: General semantic similarity (all objects)
- **Claim manifold**: Propositional equivalence (promoted claims only)
- **Procedure manifold**: Workflow/instruction similarity (procedures only)
- **Relation manifold**: Graph neighborhood fingerprints (derived)
- **Time manifold**: Temporal compatibility (structured features)
- **Evidence manifold**: Source quality metrics (computed scores)

## Status

| Component | Status |
|-----------|--------|
| Database models | Written, not migrated |
| Canonical form generators | Written, not tested |
| Query classifier v2 | Written, not integrated |
| Manifold embedder | Written, not tested |
| Shadow comparison | Written, not integrated |
| Migration scripts | Written, not run |
| MCP tools v2 | Defined, not registered |
| Neo4j blueprint | Stubs only |

## How To Activate (Future)

1. Review ROADMAP.md
2. Create a backup
3. Run `migrations/002_manifold_tables.sql`
4. Run backfill scripts in order
5. Set environment variables
6. Restart services

See ROADMAP.md for full procedure.

## Do Not

- Import anything from this directory into production code
- Run migration scripts without explicit approval
- Start any services from this directory
- Modify existing GAMI tables

## Directory Structure

```
manifold/
├── README.md              # This file
├── ROADMAP.md             # Full implementation plan
├── models/                # SQLAlchemy models
├── canonical/             # Canonical form generators
├── embeddings/            # Manifold embedding
├── retrieval/             # Query processing and fusion
├── graph/                 # Graph expansion (AGE + Neo4j stubs)
├── migrations/            # SQL migrations (not in Alembic)
├── scripts/               # Backfill and migration scripts
├── mcp/                   # MCP tool definitions
└── tests/                 # Unit tests
```
