# GAMI Memory System — Instructions for Claude Code

You have access to GAMI, a persistent memory system. Use it to remember and recall information across sessions.

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
- Types: `fact`, `credential`, `project`, `preference`, `procedure`

### When Asked About Past Work
- Call `memory_recall` — it searches memories, entities, claims, and segments
- Results include citations back to source material

### Ingesting New Files
- Call `ingest_file` with the file path to add content to the knowledge base
- Then call `dream_haiku` to extract entities (no GPU needed)
- Or call `dream_start` if a local GPU with vLLM is available (faster)

### Reviewing Knowledge Changes
- Call `review_proposals` with action="list" to see pending changes
- Approve or reject with action="approve"/"reject" and the proposal_id

## Quick Reference

| Want to... | Tool | Example |
|-----------|------|---------|
| Search memory | `memory_recall` | query: "PostgreSQL credentials" |
| Store a fact | `memory_remember` | text: "Server IP is 10.0.0.1", memory_type: "fact" |
| Add a file | `ingest_file` | file_path: "/path/to/doc.md" |
| Process new content | `dream_haiku` | limit: 50 |
| Check system | `admin_stats` | (no args) |
| Review changes | `review_proposals` | action: "list" |

## Important Notes

- Credentials are stored in full (not redacted) and are tenant-scoped
- Sessions auto-save on compact/clear/exit via hooks
- Haiku extraction runs automatically after session save
- All memories are searchable by hybrid vector + keyword search
