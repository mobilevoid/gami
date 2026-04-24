"""MCP Tool Definitions for GAMI.

JSON schemas for each tool's input/output following the MCP specification.
Each tool maps to a GAMI operation (memory recall, search, ingest, etc.).
"""

TOOL_DEFINITIONS = {
    "memory_recall": {
        "name": "memory_recall",
        "description": (
            "Recall relevant memories for a query within a token budget. "
            "Returns context text with source citations, suitable for "
            "injecting into LLM context windows."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to recall memories for",
                },
                "tenant_id": {
                    "type": "string",
                    "description": "Tenant ID to search within",
                    "default": "default",
                },
                "tenant_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Search multiple tenants (overrides tenant_id)",
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Maximum tokens in returned context",
                    "default": 2000,
                    "minimum": 100,
                    "maximum": 16000,
                },
                "detail_level": {
                    "type": "string",
                    "description": "Level of detail: summary (abstractions only), normal (+ important deltas), full (original text)",
                    "enum": ["summary", "normal", "full"],
                    "default": "normal",
                },
                "event_after": {
                    "type": "string",
                    "format": "date-time",
                    "description": "Filter: only return events that happened after this ISO timestamp",
                },
                "event_before": {
                    "type": "string",
                    "format": "date-time",
                    "description": "Filter: only return events that happened before this ISO timestamp",
                },
                "ingested_after": {
                    "type": "string",
                    "format": "date-time",
                    "description": "Filter: only return content ingested after this ISO timestamp",
                },
                "ingested_before": {
                    "type": "string",
                    "format": "date-time",
                    "description": "Filter: only return content ingested before this ISO timestamp",
                },
            },
            "required": ["query"],
        },
    },
    "memory_remember": {
        "name": "memory_remember",
        "description": (
            "Store a new assistant memory (preference, fact, correction, etc.). "
            "The memory is stored with provenance and can be recalled later."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The memory text to store",
                },
                "memory_type": {
                    "type": "string",
                    "description": "Type of memory",
                    "enum": [
                        "preference", "project", "correction", "identity",
                        "policy", "task", "relationship", "open_loop",
                        "fact", "style", "biography", "constraint",
                    ],
                    "default": "fact",
                },
                "subject_id": {
                    "type": "string",
                    "description": "Subject identifier (e.g., person name, project name)",
                    "default": "general",
                },
                "tenant_id": {
                    "type": "string",
                    "description": "Tenant ID that owns this memory",
                    "default": "default",
                },
                "importance": {
                    "type": "number",
                    "description": "Importance score 0.0-1.0",
                    "default": 0.5,
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "skip_consolidation": {
                    "type": "boolean",
                    "description": "Skip duplicate detection (force ADD)",
                    "default": False,
                },
            },
            "required": ["text"],
        },
    },
    "memory_forget": {
        "name": "memory_forget",
        "description": (
            "Mark a memory as archived/superseded. Does not delete — "
            "marks it so it won't appear in future recalls."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "string",
                    "description": "The memory ID to forget",
                },
                "reason": {
                    "type": "string",
                    "description": "Reason for forgetting",
                    "default": "user_requested",
                },
            },
            "required": ["memory_id"],
        },
    },
    "memory_update": {
        "name": "memory_update",
        "description": (
            "Update an existing memory with new information. Creates a new "
            "version and supersedes the old one."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "string",
                    "description": "The memory ID to update",
                },
                "new_text": {
                    "type": "string",
                    "description": "Updated memory text",
                },
                "reason": {
                    "type": "string",
                    "description": "Reason for the update",
                    "default": "correction",
                },
            },
            "required": ["memory_id", "new_text"],
        },
    },
    "memory_cite": {
        "name": "memory_cite",
        "description": (
            "Get provenance and source citations for a memory or claim. "
            "Shows where the information came from."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "target_id": {
                    "type": "string",
                    "description": "Memory ID, claim ID, or entity ID to cite",
                },
                "target_type": {
                    "type": "string",
                    "description": "Type of target",
                    "enum": ["memory", "claim", "entity", "event"],
                    "default": "memory",
                },
            },
            "required": ["target_id"],
        },
    },
    "memory_verify": {
        "name": "memory_verify",
        "description": (
            "Verify a claim or memory against known facts. Checks for "
            "contradictions, support, and confidence levels."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "statement": {
                    "type": "string",
                    "description": "The statement to verify",
                },
                "tenant_id": {
                    "type": "string",
                    "description": "Tenant context for verification",
                    "default": "default",
                },
            },
            "required": ["statement"],
        },
    },
    "memory_search": {
        "name": "memory_search",
        "description": (
            "Search GAMI memory using hybrid vector + lexical search. "
            "Returns ranked segments with relevance scores."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query",
                },
                "tenant_id": {
                    "type": "string",
                    "description": "Tenant ID",
                    "default": "default",
                },
                "tenant_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Search multiple tenants",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results",
                    "default": 20,
                    "minimum": 1,
                    "maximum": 100,
                },
                "search_mode": {
                    "type": "string",
                    "description": "Search mode",
                    "enum": ["hybrid", "vector", "lexical"],
                    "default": "hybrid",
                },
            },
            "required": ["query"],
        },
    },
    "memory_context": {
        "name": "memory_context",
        "description": (
            "Get contextual information about an entity, including its "
            "neighborhood, claims, relations, and recent mentions."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "entity_name": {
                    "type": "string",
                    "description": "Name of the entity to get context for",
                },
                "entity_id": {
                    "type": "string",
                    "description": "Entity ID (alternative to name)",
                },
                "tenant_id": {
                    "type": "string",
                    "description": "Tenant context",
                    "default": "default",
                },
                "include_claims": {
                    "type": "boolean",
                    "description": "Include related claims",
                    "default": True,
                },
                "include_relations": {
                    "type": "boolean",
                    "description": "Include relations",
                    "default": True,
                },
            },
            "required": [],
        },
    },
    "ingest_source": {
        "name": "ingest_source",
        "description": (
            "Ingest a source file into GAMI. Parses, segments, and stores "
            "the content for later retrieval."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to ingest",
                },
                "source_type": {
                    "type": "string",
                    "description": "Type of source",
                    "enum": [
                        "markdown", "conversation_session", "plaintext",
                        "sqlite_memory", "html", "pdf",
                    ],
                    "default": "markdown",
                },
                "tenant_id": {
                    "type": "string",
                    "description": "Tenant that owns this source",
                    "default": "shared",
                },
                "title": {
                    "type": "string",
                    "description": "Optional title for the source",
                },
            },
            "required": ["file_path"],
        },
    },
    "graph_explore": {
        "name": "graph_explore",
        "description": (
            "Explore the knowledge graph starting from an entity. "
            "Returns connected entities, relations, and paths."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "entity_id": {
                    "type": "string",
                    "description": "Starting entity ID",
                },
                "entity_name": {
                    "type": "string",
                    "description": "Starting entity name (resolved to ID)",
                },
                "depth": {
                    "type": "integer",
                    "description": "How many hops to explore",
                    "default": 2,
                    "minimum": 1,
                    "maximum": 4,
                },
                "relation_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter by relation types",
                },
                "tenant_id": {
                    "type": "string",
                    "description": "Tenant context",
                    "default": "default",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max nodes to return",
                    "default": 50,
                },
            },
            "required": [],
        },
    },
    "admin_stats": {
        "name": "admin_stats",
        "description": (
            "Get GAMI system statistics: counts of entities, claims, "
            "segments, sources, and storage usage."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "tenant_id": {
                    "type": "string",
                    "description": "Tenant to get stats for (omit for global)",
                },
            },
            "required": [],
        },
    },
    "get_unprocessed_segments": {
        "name": "get_unprocessed_segments",
        "description": (
            "Get segments that haven't been processed for entity extraction yet. "
            "Returns segment text and IDs ready for analysis."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Max segments to return",
                    "default": 20,
                },
                "tenant_id": {
                    "type": "string",
                    "description": "Tenant to search within",
                    "default": "default",
                },
                "min_length": {
                    "type": "integer",
                    "description": "Minimum segment text length",
                    "default": 200,
                },
                "max_length": {
                    "type": "integer",
                    "description": "Maximum segment text length",
                    "default": 3000,
                },
            },
            "required": [],
        },
    },
    "store_extractions": {
        "name": "store_extractions",
        "description": (
            "Store extracted entities from a segment analysis. "
            "Creates entity records and provenance links."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "segment_id": {
                    "type": "string",
                    "description": "ID of the segment that was analyzed",
                },
                "source_id": {
                    "type": "string",
                    "description": "Source ID the segment belongs to",
                },
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "type": {"type": "string"},
                            "description": {"type": "string"},
                        },
                    },
                    "description": "Array of extracted entities",
                },
                "tenant_id": {
                    "type": "string",
                    "description": "Tenant ID",
                    "default": "default",
                },
            },
            "required": ["segment_id", "entities"],
        },
    },
    "ingest_file": {
        "name": "ingest_file",
        "description": (
            "Ingest a local file into GAMI. Parses the file into segments and stores "
            "them in the database. Supports markdown, plaintext, and other text formats. "
            "After ingestion, segments are available for search and entity extraction."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to the file to ingest",
                },
                "tenant_id": {
                    "type": "string",
                    "description": "Tenant to own this content",
                    "default": "default",
                },
                "source_type": {
                    "type": "string",
                    "description": "Type of content: markdown, plaintext, conversation_session",
                    "default": "markdown",
                },
            },
            "required": ["file_path"],
        },
    },
    "dream_haiku": {
        "name": "dream_haiku",
        "description": (
            "Run dream-like knowledge synthesis using Haiku instead of vLLM. "
            "For machines without a GPU, or when the GPU is busy. Processes entity "
            "extraction at ~300 segments/hour using OAuth billing. "
            "Use dream_start for local GPU processing (~500 segments/hour)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Max segments to process (default 50)",
                    "default": 50,
                },
                "phases": {
                    "type": "string",
                    "description": "Which phases to run: all, extract, summarize",
                    "default": "all",
                },
            },
            "required": [],
        },
    },
    "run_haiku_extraction": {
        "name": "run_haiku_extraction",
        "description": (
            "Trigger the Haiku agent extraction pipeline. Spawns a Claude Code Haiku "
            "agent that extracts entities from unprocessed segments using OAuth billing. "
            "Works without GPU — use this when vLLM is down or the GPU is busy. "
            "Checks Ollama status, Claude CLI availability, and unprocessed segment count."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Max segments for the agent to process",
                    "default": 10,
                },
            },
            "required": [],
        },
    },
    "memory_correct": {
        "name": "memory_correct",
        "description": (
            "Fix wrong information in GAMI in real-time. When you discover a memory, "
            "entity, or claim contains incorrect data (wrong password, wrong IP, "
            "outdated fact), call this to correct it immediately. Old values are "
            "archived for audit trail. Use this PROACTIVELY when you notice errors "
            "during recall — don't wait for the user to ask."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "item_type": {
                    "type": "string",
                    "description": "Type of item to correct",
                    "enum": ["memory", "entity", "claim"],
                },
                "search_text": {
                    "type": "string",
                    "description": "Text to find the wrong item (name, subject, or content fragment)",
                },
                "wrong_value": {
                    "type": "string",
                    "description": "The incorrect value (helps narrow to the right item)",
                    "default": "",
                },
                "correct_value": {
                    "type": "string",
                    "description": "The correct value to replace it with",
                },
                "reason": {
                    "type": "string",
                    "description": "Why this correction is being made",
                    "default": "Corrected during conversation",
                },
                "tenant_id": {
                    "type": "string",
                    "description": "Tenant scope",
                    "default": "default",
                },
            },
            "required": ["item_type", "correct_value"],
        },
    },
    "memory_feedback": {
        "name": "memory_feedback",
        "description": (
            "Record feedback on retrieval quality for learning. Call this after "
            "using recalled memories to help GAMI learn which retrievals were useful. "
            "This improves future recall quality through machine learning."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Session ID from the recall request",
                },
                "feedback_type": {
                    "type": "string",
                    "description": "Type of feedback",
                    "enum": ["confirmed", "used", "continued", "ignored", "corrected", "wrong"],
                },
                "correction_text": {
                    "type": "string",
                    "description": "Correction text if feedback_type is 'corrected'",
                },
            },
            "required": ["session_id", "feedback_type"],
        },
    },
    "memory_suggest_procedure": {
        "name": "memory_suggest_procedure",
        "description": (
            "Suggest relevant workflow patterns for a task. Searches workflow memories "
            "(extracted from past sessions) and returns matching patterns. Workflow memories "
            "consolidate naturally over time via the dream cycle. Use when the user is "
            "attempting a task that might match a known workflow pattern."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Description of what the user is trying to accomplish",
                },
                "context": {
                    "type": "string",
                    "description": "Current conversation context or error message",
                },
                "tenant_id": {
                    "type": "string",
                    "description": "Tenant ID",
                    "default": "default",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum workflows to return",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 10,
                },
                "min_confidence": {
                    "type": "number",
                    "description": "Minimum workflow confidence/importance",
                    "default": 0.4,
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
            },
            "required": ["query"],
        },
    },
}

DREAM_TOOLS = [
    {
        "name": "dream_start",
        "description": "Start the GAMI dream cycle — background knowledge synthesis using idle GPU time. Extracts entities, generates summaries, resolves aliases, builds relations.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "phase": {"type": "string", "enum": ["extract", "summarize", "resolve", "relate", "score", "embed"], "description": "Run only a specific phase"},
                "duration": {"type": "integer", "default": 3600, "description": "Max duration in seconds"},
            },
        },
    },
    {
        "name": "dream_stop",
        "description": "Stop the running dream cycle gracefully. It will finish the current task and exit.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "dream_status",
        "description": "Check if the dream cycle is running and see recent log output.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]

# Merge list-style tool defs into the main dict
for tool in DREAM_TOOLS:
    TOOL_DEFINITIONS[tool["name"]] = tool

REVIEW_TOOLS = [
    {
        "name": "review_proposals",
        "description": "Review, approve, or reject pending proposals from the dream cycle. Use action='list' to see pending changes, 'approve' or 'reject' with a proposal_id to act on them. Credential changes always need explicit review.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["list", "approve", "reject"], "default": "list"},
                "proposal_id": {"type": "string", "description": "ID of proposal to approve/reject"},
                "reason": {"type": "string", "description": "Reason for rejection"},
            },
        },
    },
]

for tool in REVIEW_TOOLS:
    TOOL_DEFINITIONS[tool["name"]] = tool

TENANT_TOOLS = [
    {
        "name": "create_tenant",
        "description": (
            "Create a new tenant for organizing content. Tenants isolate content "
            "so it's only searched when explicitly requested."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "tenant_id": {
                    "type": "string",
                    "description": "Unique tenant identifier (lowercase, no spaces)",
                },
                "display_name": {
                    "type": "string",
                    "description": "Human-readable name for the tenant",
                },
                "description": {
                    "type": "string",
                    "description": "Description of what this tenant contains",
                    "default": "",
                },
            },
            "required": ["tenant_id", "display_name"],
        },
    },
    {
        "name": "bulk_ingest",
        "description": (
            "Ingest a directory of files into a tenant. Launches background processing "
            "with progress tracking. Supports PDF, markdown, and plaintext files. "
            "Auto-creates the tenant if it doesn't exist. Deduplicates by checksum."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path containing files to ingest",
                },
                "tenant_id": {
                    "type": "string",
                    "description": "Tenant ID to ingest into",
                },
                "file_type": {
                    "type": "string",
                    "description": "Type of files to process",
                    "enum": ["pdf", "markdown", "plaintext"],
                    "default": "pdf",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Recurse into subdirectories",
                    "default": True,
                },
            },
            "required": ["path", "tenant_id"],
        },
    },
    {
        "name": "tenant_stats",
        "description": (
            "Get detailed statistics for a specific tenant: source count, "
            "segment count, embedded vs unembedded, entity count, and text size."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "tenant_id": {
                    "type": "string",
                    "description": "Tenant ID to get stats for",
                },
            },
            "required": ["tenant_id"],
        },
    },
    {
        "name": "tenant_search",
        "description": (
            "Search within a specific tenant using hybrid vector + keyword search. "
            "Returns results with citations including source title, author, and page numbers. "
            "Use this to search books, documents, or any tenant-specific content."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query",
                },
                "tenant_id": {
                    "type": "string",
                    "description": "Tenant ID to search within",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum results to return",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 50,
                },
            },
            "required": ["query", "tenant_id"],
        },
    },
]

for tool in TENANT_TOOLS:
    TOOL_DEFINITIONS[tool["name"]] = tool
