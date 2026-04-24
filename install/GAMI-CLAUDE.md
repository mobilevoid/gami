# GAMI Memory System — Instructions for Claude Code

You have access to GAMI v2.0, a persistent memory system with advanced retrieval capabilities. Use it to remember and recall information across sessions.

## Setup (first time only)

If GAMI is not yet installed, run:
```bash
bash /opt/gami/install/install.sh
```
Then restart Claude Code to load the MCP tools.

## Core Usage Pattern

### Session Start
- Call `memory_recall` with a query about the current task to load relevant context
- Call `admin_stats` to check system health

### When You Learn Something Important
- Call `memory_remember` with the fact, credential, or project detail
- Types: `fact`, `credential`, `project`, `preference`, `workflow`
- Duplicates are auto-detected (ADD/UPDATE/NOOP)

### When Asked About Past Work
- Call `memory_recall` — it searches 8 indexes: memories, entities, claims, relations, workflows, clusters, causal, segments
- Results include citations and contradiction warnings
- Use `detail_level`: "summary", "normal", or "full"

### Temporal Queries
- Use `event_after`/`event_before` to filter by when events happened
- Use `ingested_after`/`ingested_before` to filter by when learned

### Finding Workflow Patterns
- Call `memory_suggest_procedure` with a task description
- Returns workflow patterns extracted from past sessions
- Workflows consolidate automatically via dream cycle

### Ingesting New Files
- Call `ingest_file` with the file path to add content to the knowledge base
- Then call `dream_haiku` to extract entities (no GPU needed)
- Or call `dream_start` if a local GPU with vLLM is available (faster)

### Correcting Wrong Information (IMPORTANT)
When `memory_recall` returns information you KNOW is wrong — wrong password, wrong IP, outdated fact, or the user tells you something has changed — call `memory_correct` IMMEDIATELY. Do not wait for the user to ask.

```
memory_correct(item_type="memory", search_text="CT231 password",
               wrong_value="old_pass", correct_value="new_pass",
               reason="User confirmed new password")
```

This is critical for data quality. Bad data that persists across sessions compounds errors.

### Reviewing Knowledge Changes
- Call `review_proposals` with action="list" to see pending changes
- Approve or reject with action="approve"/"reject" and the proposal_id

## Quick Reference

| Want to... | Tool | Example |
|-----------|------|---------|
| Search memory | `memory_recall` | query: "PostgreSQL credentials" |
| Search with time filter | `memory_recall` | query: "server changes", event_after: "2026-01-01" |
| Store a fact | `memory_remember` | text: "Server IP is 10.0.0.1", memory_type: "fact" |
| Find workflows | `memory_suggest_procedure` | query: "deploy nginx" |
| Fix wrong data | `memory_correct` | item_type: "memory", search_text: "old IP", correct_value: "10.0.0.2" |
| Add a file | `ingest_file` | file_path: "/path/to/doc.md" |
| Process new content | `dream_haiku` | limit: 50 |
| Check system | `admin_stats` | (no args) |
| Review changes | `review_proposals` | action: "list" |

## v2.0 Features

- **Cross-encoder reranking**: 25-40% better precision on retrieval
- **Multi-index search**: Queries 8 specialized indexes based on intent
- **Workflow learning**: Extracts patterns from sessions, consolidates naturally
- **Bi-temporal queries**: Filter by event time vs ingestion time
- **Contradiction detection**: Warns when results conflict
- **Auto-consolidation**: Duplicate memories merged automatically

## Important Notes

- Credentials are stored in full (not redacted) and are tenant-scoped
- Sessions auto-save on compact/clear/exit via hooks
- Haiku extraction runs automatically after session save
- All memories are searchable by hybrid vector + keyword search
- **When you see wrong data, fix it immediately with `memory_correct`** — proactive correction keeps the knowledge base healthy
- Workflow memories consolidate over time via the dream cycle
