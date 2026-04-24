# GAMI Full Implementation Migration Plan

**Created**: 2026-04-11
**Current State**: Phase 13 schema deployed, partial code integration
**Target State**: Full Innovation Extensions with all features active

---

## Executive Summary

This plan migrates GAMI from its current state (basic RAG with Phase 13 schema) to the full Innovation Extensions implementation including:
- Learning retrieval (feedback-driven scoring)
- Memory consolidation (clustering, abstraction, decay)
- Causal extraction (cause-effect relationships)
- Subconscious daemon (proactive context management)
- Multi-agent attribution and trust scoring

**Estimated Time**: 10-15 hours implementation + 2-4 hours testing
**Risk Level**: LOW (additive changes, no destructive migrations)
**Downtime Required**: ~5 minutes for worker restart

---

## Current System Inventory

### Infrastructure
| Component | Status | Location |
|-----------|--------|----------|
| PostgreSQL 16 | Running | localhost:5433 |
| pgvector 0.8.2 | Installed | Extension |
| Apache AGE 1.6.0 | Installed | Extension, `gami_graph` |
| Redis | Running | localhost:6380 |
| GAMI API | Running | localhost:9090 |
| Celery Workers | Running | 4 workers + beat |
| vLLM | Running | localhost:8000 |
| Ollama | Running | localhost:11434 |

### Database Statistics
| Table | Rows | Size | Notes |
|-------|------|------|-------|
| segments | 494,954 | 5.9 GB | 387,928 embedded (78%), 107,026 pending |
| entities | 118,824 | 492 MB | 114,726 active |
| summaries | 90,698 | 170 MB | |
| provenance | 272,973 | 92 MB | |
| sources | 16,758 | 15 MB | |
| relations | 5,370 | 4 MB | |
| assistant_memories | 565 | 6.6 MB | |
| claims | 454 | 5 MB | |
| jobs | 8,936 | 3 MB | |
| tenants | 18 | 48 KB | |

### Phase 13 Tables (Already Created)
| Table | Rows | Status |
|-------|------|--------|
| retrieval_logs | 1 | Schema ready, not being populated |
| agent_configs | 2 | Schema ready, needs more entries |
| prompt_templates | 10 | Seeded with defaults |
| causal_relations | 0 | Schema ready, not being populated |
| memory_clusters | 0 | Schema ready, not being populated |
| sessions | 0 | Schema ready, not being populated |
| subconscious_events | 0 | Schema ready, not being populated |
| agent_trust_history | 0 | Schema ready, not being populated |

### Phase 13 Columns (Already Added)
All attribution columns have been added to existing tables:
- segments: created_by_agent_id, created_by_user_id, derived_from, derivation_type, stability_score, decay_score, cluster_id, last_reinforced_at ✓
- entities: created_by_agent_id, created_by_user_id ✓
- claims: created_by_agent_id, created_by_user_id, derived_from, derivation_type ✓
- relations: created_by_agent_id ✓
- assistant_memories: created_by_agent_id, created_by_user_id, derived_from, cluster_id ✓

### Vector Indexes (Already Created)
- idx_segments_embedding (ivfflat)
- idx_entities_embedding (ivfflat)
- idx_claims_embedding (ivfflat)
- idx_summaries_embedding (ivfflat)
- idx_memories_embedding (ivfflat)
- idx_memory_clusters_embedding (ivfflat)
- idx_memory_clusters_abstraction_embedding (ivfflat)
- idx_retrieval_logs_query_embedding (ivfflat)

### Code Files (Already Created)
| File | Lines | Purpose |
|------|-------|---------|
| api/llm/providers.py | 973 | Multi-provider LLM/embedding system |
| api/llm/mcp_integration.py | 825 | MCP tools for Claude |
| api/services/agent_service.py | 1,100 | Agent config management |
| api/services/prompt_service.py | 700 | Prompt template system |
| api/services/causal_extractor.py | 720 | Causal relationship extraction |
| api/services/learning_service.py | ~500 | Learning signal processing |
| api/services/consolidation_service.py | ~400 | Memory consolidation |
| manifold/config_v2.py | 630 | Hierarchical config system |
| manifold/config_loader.py | ~300 | Hot-reload config manager |
| daemon/subconscious.py | 620 | Main daemon process |
| daemon/state_classifier.py | 320 | Conversation state classification |
| daemon/predictive_retriever.py | 410 | Predictive retrieval |
| daemon/context_injector.py | 350 | Context injection |

### Uncommitted Changes
- workers/celery_app.py (modified - added embedding backfill task)
- workers/parser_worker.py (modified - added inline embedding)
- Several new scripts (untracked)

---

## Pre-Migration Backup Procedures

### Step 1: Create Full Database Backup
```bash
# Create backup directory
mkdir -p /opt/gami/backups/$(date +%Y%m%d)

# Full PostgreSQL dump (preserves everything)
PGPASSWORD=gami pg_dump -h localhost -p 5433 -U gami -d gami \
  --format=custom \
  --file=/opt/gami/backups/$(date +%Y%m%d)/gami_full_$(date +%H%M).dump

# Also create SQL text backup for inspection
PGPASSWORD=gami pg_dump -h localhost -p 5433 -U gami -d gami \
  --format=plain \
  --file=/opt/gami/backups/$(date +%Y%m%d)/gami_full_$(date +%H%M).sql
```

### Step 2: Backup Redis State
```bash
# Trigger Redis background save
redis-cli -p 6380 BGSAVE

# Copy RDB file
cp /var/lib/redis/dump.rdb /opt/gami/backups/$(date +%Y%m%d)/redis_$(date +%H%M).rdb
```

### Step 3: Backup Code
```bash
# Commit current state
cd /opt/gami
git add -A
git stash  # If you don't want to commit yet

# Or create a tarball
tar -czf /opt/gami/backups/$(date +%Y%m%d)/gami_code_$(date +%H%M).tar.gz \
  --exclude=__pycache__ --exclude=.git --exclude=models \
  /opt/gami
```

### Step 4: Document Current Process State
```bash
# Save running processes
ps aux | grep -E "gami|celery|uvicorn" > /opt/gami/backups/$(date +%Y%m%d)/processes.txt

# Save Celery task state
cd /opt/gami && PYTHONPATH=/opt/gami python3 -c "
from workers.celery_app import celery_app
i = celery_app.control.inspect()
print('Active:', i.active())
print('Reserved:', i.reserved())
print('Scheduled:', i.scheduled())
" > /opt/gami/backups/$(date +%Y%m%d)/celery_state.txt
```

---

## Migration Steps

### Phase 1: Commit Current Changes (5 min)

**Objective**: Save all current work to git

```bash
cd /opt/gami
git add workers/celery_app.py workers/parser_worker.py
git commit -m "Fix inline embedding during ingestion

- parser_worker.py: Embed segments immediately after storage (CPU, batches of 32)
- celery_app.py: Add embedding-backfill-15m periodic task for catch-up

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

### Phase 2: Create Consolidation Module (2-3 hours)

**Objective**: Implement memory clustering, abstraction, and decay

Create `/opt/gami/manifold/consolidation/`:
```
consolidation/
├── __init__.py
├── clusterer.py      # Cluster similar memories using sklearn
├── abstractor.py     # Generate abstractions via LLM
└── decay.py          # Apply temporal decay, archive old memories
```

**clusterer.py** implementation:
- Use sklearn AgglomerativeClustering with cosine distance
- Threshold: 0.85 similarity (configurable via config_v2)
- Min cluster size: 3, Max: 50
- Output: List of segment_id clusters

**abstractor.py** implementation:
- Take cluster texts, call LLM to generate abstraction
- Store in memory_clusters.abstraction_text
- Generate embedding for abstraction

**decay.py** implementation:
- Calculate decay based on last_accessed_at and decay_rate
- Update decay_score on segments and clusters
- Archive items below threshold (move to cold storage)

### Phase 3: Add Dream Phases 9-13 (3-4 hours)

**Objective**: Extend dream_cycle.py with new processing phases

Add to `/opt/gami/scripts/dream_cycle.py`:

```python
# Phase 9: Learning Analysis
def dream_learn(config, max_logs=1000):
    """Process retrieval logs and adjust segment importance."""
    # Query unprocessed retrieval_logs with outcomes
    # Aggregate signals per segment
    # Apply bandit-style importance adjustments
    # Mark logs as processed

# Phase 10: Causal Extraction
def dream_causal(config, max_segments=500):
    """Extract causal relationships from segments."""
    # Query segments not yet processed for causality
    # Run causal_extractor on each
    # Store results in causal_relations table
    # Link to entities where possible

# Phase 11: Memory Consolidation
def dream_consolidate(config, max_clusters=50):
    """Cluster similar memories and create abstractions."""
    # Query unclustered memories with embeddings
    # Run clustering algorithm
    # Create memory_clusters entries
    # Update member segments with cluster_id
    # Generate abstractions for each cluster

# Phase 12: Inference Generation
def dream_infer(config, max_clusters=20):
    """Generate inferences from stable clusters."""
    # Query high-stability clusters
    # Use LLM to generate potential inferences
    # Store as new claims with derivation_type='inferred'
    # Link back to source cluster

# Phase 13: Decay & Archive
def dream_decay(config):
    """Apply decay to unreferenced memories."""
    # Query segments/clusters not accessed recently
    # Apply decay formula
    # Archive items below threshold
    # Update statistics
```

### Phase 4: Wire Retrieval Logging (30 min)

**Objective**: Log all retrieval queries for learning

Modify `/opt/gami/manifold/retrieval/orchestrator.py`:
```python
# After retrieval completes, log the query
async def log_retrieval(
    session_id: str,
    query_text: str,
    query_embedding: list,
    segments_returned: list[str],
    scores: list[float],
    tenant_id: str,
    agent_id: str = None
):
    # Insert into retrieval_logs table
    # Non-blocking (fire and forget)
```

### Phase 5: Wire Session Tracking (1 hour)

**Objective**: Track conversation sessions for state management

Modify `/opt/gami/api/routers/memory.py`:
```python
# On each memory operation, ensure session exists
async def ensure_session(
    session_id: str,
    tenant_id: str,
    agent_id: str = None
) -> str:
    # Create or update session in sessions table
    # Track message_count, retrieval_count
    # Update last_activity_at
```

Add session middleware or dependency to key routes.

### Phase 6: Create Agent Configs for Existing Tenants (30 min)

**Objective**: Populate agent_configs for existing tenants

```sql
-- Create default agent configs for each tenant
INSERT INTO agent_configs (agent_id, owner_tenant_id, agent_name, agent_type)
SELECT 
    tenant_id || '-default',
    tenant_id,
    tenant_id || ' Default Agent',
    'assistant'
FROM tenants
WHERE NOT EXISTS (
    SELECT 1 FROM agent_configs WHERE owner_tenant_id = tenants.tenant_id
);
```

### Phase 7: Configure and Start Subconscious Daemon (1-2 hours)

**Objective**: Enable proactive context management

1. Create systemd service:
```ini
# /etc/systemd/system/gami-subconscious.service
[Unit]
Description=GAMI Subconscious Daemon
After=network.target redis.service postgresql.service

[Service]
Type=simple
User=ai
WorkingDirectory=/opt/gami
ExecStart=/usr/local/anaconda3/bin/python -m daemon
Restart=always
RestartSec=10
Environment=PYTHONPATH=/opt/gami
Environment=GAMI_SUBCONSCIOUS_ENABLED=true

[Install]
WantedBy=multi-user.target
```

2. Configure Redis pubsub channels
3. Test state classification
4. Enable gradually (start disabled, monitor, then enable)

### Phase 8: Restart Workers with New Code (5 min)

**Objective**: Activate all changes

```bash
# Graceful worker restart
pkill -TERM -f "celery.*gami"
sleep 5

# Start new workers
cd /opt/gami
PYTHONPATH=/opt/gami nohup python -m celery -A workers.celery_app worker \
  --loglevel=info --concurrency=4 \
  --queues=parse,embed,extract,background,celery --beat \
  > /tmp/gami-celery.log 2>&1 &

# Restart API (optional, for new routes)
pkill -TERM -f "uvicorn.*api.main"
sleep 2
PYTHONPATH=/opt/gami nohup python -m uvicorn api.main:app \
  --host 0.0.0.0 --port 9090 --workers 1 \
  > /tmp/gami-api.log 2>&1 &
```

### Phase 9: Run Initial Dream Cycle with New Phases (2-4 hours)

**Objective**: Populate new tables with initial data

```bash
cd /opt/gami
PYTHONPATH=/opt/gami python scripts/dream_cycle.py --phases 9,10,11,12,13

# Or run phases individually for monitoring:
PYTHONPATH=/opt/gami python -c "
from scripts.dream_cycle import dream_learn, dream_causal, dream_consolidate
dream_learn(max_logs=100)  # Process retrieval feedback
dream_causal(max_segments=100)  # Extract causality
dream_consolidate(max_clusters=20)  # Create clusters
"
```

### Phase 10: Embed Backfill for 107K Segments (Background)

**Objective**: Complete embedding for all segments

```bash
# This will run automatically via the new periodic task
# Or trigger manually:
cd /opt/gami
PYTHONPATH=/opt/gami python -c "
from workers.embedder_worker import embed_segments
embed_segments(tenant_id='default', batch_size=100)
"

# Monitor progress:
watch -n 30 'psql -h localhost -p 5433 -U gami -d gami -c "SELECT COUNT(*) as pending FROM segments WHERE embedding IS NULL"'
```

---

## Verification Checklist

### Database Verification
```sql
-- Check all Phase 13 tables have data
SELECT 'retrieval_logs' as tbl, COUNT(*) FROM retrieval_logs
UNION ALL SELECT 'agent_configs', COUNT(*) FROM agent_configs
UNION ALL SELECT 'prompt_templates', COUNT(*) FROM prompt_templates
UNION ALL SELECT 'causal_relations', COUNT(*) FROM causal_relations
UNION ALL SELECT 'memory_clusters', COUNT(*) FROM memory_clusters
UNION ALL SELECT 'sessions', COUNT(*) FROM sessions;

-- Check embedding completion
SELECT 
    COUNT(*) as total,
    COUNT(embedding) as embedded,
    COUNT(*) - COUNT(embedding) as pending
FROM segments;

-- Check attribution columns are being used
SELECT COUNT(*) FROM segments WHERE created_by_agent_id IS NOT NULL;
SELECT COUNT(*) FROM entities WHERE created_by_agent_id IS NOT NULL;
```

### Service Verification
```bash
# API health
curl http://localhost:9090/health

# Celery workers
cd /opt/gami && PYTHONPATH=/opt/gami python -c "
from workers.celery_app import celery_app
i = celery_app.control.inspect()
print('Workers:', list(i.active().keys()))
print('Registered tasks:', len(celery_app.tasks))
"

# Subconscious daemon (if enabled)
systemctl status gami-subconscious
redis-cli -p 6380 PUBSUB CHANNELS "gami:*"
```

### Functional Verification
```bash
# Test retrieval logging
curl -X POST http://localhost:9090/api/v1/memory/recall \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "tenant_id": "test", "session_id": "test-session"}'

# Check log was created
psql -h localhost -p 5433 -U gami -d gami -c \
  "SELECT * FROM retrieval_logs ORDER BY created_at DESC LIMIT 1"

# Test dream cycle phases
cd /opt/gami && PYTHONPATH=/opt/gami python -c "
from scripts.dream_cycle import dream_learn
result = dream_learn(max_logs=10)
print(result)
"
```

---

## Rollback Procedures

### If Migration Fails

1. **Stop new services**:
```bash
pkill -f "celery.*gami"
pkill -f "uvicorn.*api.main"
systemctl stop gami-subconscious 2>/dev/null
```

2. **Restore code from git**:
```bash
cd /opt/gami
git checkout HEAD~1  # Or specific commit
```

3. **Restore database if needed** (only if data corruption):
```bash
# Drop and recreate database
psql -h localhost -p 5433 -U postgres -c "DROP DATABASE gami"
psql -h localhost -p 5433 -U postgres -c "CREATE DATABASE gami OWNER gami"

# Restore from backup
pg_restore -h localhost -p 5433 -U gami -d gami \
  /opt/gami/backups/YYYYMMDD/gami_full_HHMM.dump
```

4. **Restart original services**:
```bash
cd /opt/gami
PYTHONPATH=/opt/gami python -m celery -A workers.celery_app worker \
  --loglevel=info --concurrency=4 \
  --queues=parse,embed,extract,background,celery --beat &

PYTHONPATH=/opt/gami python -m uvicorn api.main:app \
  --host 0.0.0.0 --port 9090 --workers 1 &
```

### Partial Rollback (Disable New Features)

If new features cause issues but core system is fine:

```bash
# Disable subconscious daemon
systemctl stop gami-subconscious
systemctl disable gami-subconscious

# Set config flags to disable new features
export GAMI_LEARNING_ENABLED=false
export GAMI_CONSOLIDATION_ENABLED=false
export GAMI_CAUSAL_ENABLED=false
export GAMI_SUBCONSCIOUS_ENABLED=false

# Restart workers with flags
pkill -f "celery.*gami"
cd /opt/gami
PYTHONPATH=/opt/gami python -m celery -A workers.celery_app worker \
  --loglevel=info --concurrency=4 \
  --queues=parse,embed,extract,background,celery --beat &
```

---

## Data Protection Summary

| Data | Protection Method |
|------|-------------------|
| **segments** (5.9GB) | Pre-migration pg_dump, no destructive changes |
| **entities** (492MB) | Pre-migration pg_dump, only additive column updates |
| **summaries** (170MB) | Pre-migration pg_dump, read-only during migration |
| **provenance** (92MB) | Pre-migration pg_dump, read-only during migration |
| **All other tables** | Pre-migration pg_dump |
| **Redis state** | RDB snapshot before migration |
| **Code** | Git commit before changes |
| **Config** | Backup /opt/gami/*.env files |

**No data will be deleted during this migration.** All changes are additive:
- New tables (already created)
- New columns (already added)
- New indexes (already created)
- New processing phases (additive)

---

## Timeline

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Backup | 15 min | None |
| Commit changes | 5 min | Backup |
| Consolidation module | 2-3 hours | Commit |
| Dream phases 9-13 | 3-4 hours | Consolidation |
| Wire retrieval logging | 30 min | Dream phases |
| Wire session tracking | 1 hour | Dream phases |
| Agent configs | 30 min | None |
| Subconscious daemon | 1-2 hours | Session tracking |
| Worker restart | 5 min | All code changes |
| Initial dream run | 2-4 hours | Worker restart |
| Embed backfill | Background | Worker restart |
| Verification | 1-2 hours | All above |
| **Total** | **12-18 hours** | |

---

## Post-Migration Monitoring

### Daily Checks (First Week)
```bash
# Check table growth
psql -h localhost -p 5433 -U gami -d gami -c "
SELECT 'retrieval_logs' as tbl, COUNT(*) FROM retrieval_logs WHERE created_at > NOW() - INTERVAL '1 day'
UNION ALL SELECT 'sessions', COUNT(*) FROM sessions WHERE started_at > NOW() - INTERVAL '1 day'
UNION ALL SELECT 'causal_relations', COUNT(*) FROM causal_relations WHERE created_at > NOW() - INTERVAL '1 day'
UNION ALL SELECT 'memory_clusters', COUNT(*) FROM memory_clusters WHERE created_at > NOW() - INTERVAL '1 day';
"

# Check Celery task execution
grep -E "gami\.(learn|causal|consolidate)" /tmp/gami-celery.log | tail -20

# Check for errors
grep -i error /tmp/gami-*.log | tail -20
```

### Weekly Review
- Review agent_trust_history for accuracy trends
- Check memory_clusters stability scores
- Verify decay is archiving appropriately
- Monitor embedding queue depth

---

## Approval

- [ ] Backup procedures reviewed
- [ ] Rollback procedures tested
- [ ] Downtime window approved
- [ ] Monitoring alerts configured

**Ready to proceed?** Run backup first, then follow phases in order.
