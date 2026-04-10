"""MCP Tool Definitions for manifold-aware retrieval.

Defines new v2 tools that use multi-manifold retrieval:
- memory_recall_v2: Manifold-aware recall
- memory_search_v2: Manifold-aware search
- manifold_stats: Manifold coverage statistics
- query_explain: Query classification explanation
- promote_object: Manual promotion to manifold treatment
- canonicalize_claim: Convert claim to canonical form

These tools are ADDITIVE — all v1 tools remain unchanged.
"""

MANIFOLD_TOOL_DEFINITIONS = {
    "memory_recall_v2": {
        "name": "memory_recall_v2",
        "description": (
            "Manifold-aware memory recall. Uses multi-manifold embeddings and "
            "query-conditioned weights (alpha) for better retrieval. Automatically "
            "selects manifold weights based on query type (fact lookup, timeline, "
            "procedure, verification, etc.). Falls back to v1 behavior if manifold "
            "embeddings are not available for a tenant. Use 'explain: true' to see "
            "how the query was classified and which manifolds were weighted."
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
                "mode": {
                    "type": "string",
                    "enum": [
                        "auto", "fact_lookup", "synthesis", "comparison",
                        "timeline", "procedure", "assistant_memory",
                        "verification", "report",
                    ],
                    "default": "auto",
                    "description": "Query mode (auto uses classifier)",
                },
                "manifold_override": {
                    "type": "object",
                    "description": "Manual manifold weight overrides (each 0-1)",
                    "properties": {
                        "topic": {"type": "number", "minimum": 0, "maximum": 1},
                        "claim": {"type": "number", "minimum": 0, "maximum": 1},
                        "procedure": {"type": "number", "minimum": 0, "maximum": 1},
                        "relation": {"type": "number", "minimum": 0, "maximum": 1},
                        "time": {"type": "number", "minimum": 0, "maximum": 1},
                        "evidence": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                },
                "explain": {
                    "type": "boolean",
                    "default": False,
                    "description": "Include scoring breakdown and classification reasoning",
                },
            },
            "required": ["query"],
        },
    },

    "memory_search_v2": {
        "name": "memory_search_v2",
        "description": (
            "Manifold-aware memory search. Like memory_search but uses multi-manifold "
            "embeddings when available. Returns ranked results with manifold scores."
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
                "mode": {
                    "type": "string",
                    "enum": [
                        "auto", "fact_lookup", "synthesis", "comparison",
                        "timeline", "procedure", "verification",
                    ],
                    "default": "auto",
                },
                "include_manifold_scores": {
                    "type": "boolean",
                    "default": False,
                    "description": "Include per-manifold similarity scores",
                },
            },
            "required": ["query"],
        },
    },

    "manifold_stats": {
        "name": "manifold_stats",
        "description": (
            "Get statistics on manifold embedding coverage. Shows how many objects "
            "have embeddings in each manifold, promotion status distribution, and "
            "embedding model versions."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "tenant_id": {
                    "type": "string",
                    "description": "Tenant to get stats for (omit for global)",
                },
                "include_distribution": {
                    "type": "boolean",
                    "default": False,
                    "description": "Include detailed score distributions",
                },
            },
            "required": [],
        },
    },

    "query_explain": {
        "name": "query_explain",
        "description": (
            "Explain how a query would be processed by manifold retrieval. Shows "
            "query classification, manifold weights, and reasoning. Useful for "
            "debugging retrieval behavior."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Query to explain",
                },
                "tenant_id": {
                    "type": "string",
                    "description": "Tenant context",
                    "default": "claude-opus",
                },
            },
            "required": ["query"],
        },
    },

    "promote_object": {
        "name": "promote_object",
        "description": (
            "Manually promote an object to manifold treatment. Generates canonical "
            "form (if applicable) and creates manifold embeddings. Use this when "
            "you know something is important but it hasn't been auto-promoted."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "target_id": {
                    "type": "string",
                    "description": "ID of the object to promote",
                },
                "target_type": {
                    "type": "string",
                    "enum": ["segment", "claim", "entity", "memory", "summary"],
                    "description": "Type of object",
                },
                "manifolds": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["topic", "claim", "procedure"],
                    },
                    "default": ["topic"],
                    "description": "Which manifolds to embed into",
                },
                "force": {
                    "type": "boolean",
                    "default": False,
                    "description": "Overwrite existing embeddings",
                },
            },
            "required": ["target_id", "target_type"],
        },
    },

    "canonicalize_claim": {
        "name": "canonicalize_claim",
        "description": (
            "Convert a prose claim to canonical SPO (Subject-Predicate-Object) form. "
            "The canonical form is used for claim manifold embedding and structured "
            "reasoning. Returns the parsed components and canonical text."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "claim_text": {
                    "type": "string",
                    "description": "The prose claim to canonicalize",
                },
                "claim_id": {
                    "type": "string",
                    "description": "Optional claim ID for reference",
                },
                "store": {
                    "type": "boolean",
                    "default": False,
                    "description": "Store the canonical form in database",
                },
            },
            "required": ["claim_text"],
        },
    },

    "shadow_comparison_stats": {
        "name": "shadow_comparison_stats",
        "description": (
            "Get statistics from shadow mode comparisons. Shows how often new "
            "retrieval wins vs old, overlap rates, latency differences, and "
            "rank correlations. Only available when shadow mode is enabled."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "since_hours": {
                    "type": "integer",
                    "default": 24,
                    "description": "Look at comparisons from last N hours",
                },
            },
            "required": [],
        },
    },
}


# Helper to merge with existing tool definitions
def get_all_manifold_tools() -> dict:
    """Get all manifold tool definitions.

    Returns:
        Dict of tool name to definition.
    """
    return MANIFOLD_TOOL_DEFINITIONS.copy()


def get_tool_names() -> list:
    """Get list of manifold tool names.

    Returns:
        List of tool names.
    """
    return list(MANIFOLD_TOOL_DEFINITIONS.keys())
