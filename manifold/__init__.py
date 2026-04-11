# GAMI Multi-Manifold Memory System
# NOT FOR PRODUCTION USE - See ROADMAP.md
"""
Multi-manifold memory system for GAMI.

This module provides semantic retrieval across 6 manifolds:
- Topic: Dense 768d embeddings for general similarity
- Claim: SPO-normalized claims with evidence scoring
- Procedure: Ordered step sequences
- Relation: Graph-derived fingerprints from AGE
- Time: 12-dimensional temporal features
- Evidence: 5-factor verification scoring

Public API:
- ManifoldTools: MCP tool implementations
- RetrievalOrchestrator: Main retrieval coordinator
- classify_query_v2: Query classification
- ManifoldConfig: Configuration
"""

__version__ = "0.1.0"

# Core retrieval
from .retrieval.orchestrator import RetrievalOrchestrator, recall
from .retrieval.query_classifier_v2 import classify_query_v2
from .retrieval.manifold_fusion import ManifoldFusion

# Configuration
from .config import ManifoldConfig, get_config, set_config

# MCP tools
from .mcp.tools import ManifoldTools, TOOL_DEFINITIONS

# Exceptions
from .exceptions import (
    ManifoldError,
    QueryError,
    RetrievalError,
    EmbeddingError,
    StorageError,
)

__all__ = [
    # Version
    "__version__",
    # Retrieval
    "RetrievalOrchestrator",
    "recall",
    "classify_query_v2",
    "ManifoldFusion",
    # Config
    "ManifoldConfig",
    "get_config",
    "set_config",
    # MCP
    "ManifoldTools",
    "TOOL_DEFINITIONS",
    # Exceptions
    "ManifoldError",
    "QueryError",
    "RetrievalError",
    "EmbeddingError",
    "StorageError",
]
