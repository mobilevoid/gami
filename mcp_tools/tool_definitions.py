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
                    "default": "claude-opus",
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
                    "default": "claude-opus",
                },
                "importance": {
                    "type": "number",
                    "description": "Importance score 0.0-1.0",
                    "default": 0.5,
                    "minimum": 0.0,
                    "maximum": 1.0,
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
                    "default": "claude-opus",
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
                    "default": "claude-opus",
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
                    "default": "claude-opus",
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
                    "default": "claude-opus",
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
                    "default": "claude-opus",
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
                    "default": "claude-opus",
                },
            },
            "required": ["segment_id", "entities"],
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
