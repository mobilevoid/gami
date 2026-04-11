"""Golden query test set for manifold retrieval evaluation.

Each query has expected characteristics that the retrieval system
should satisfy. Use these for regression testing and A/B evaluation.
"""
from dataclasses import dataclass
from typing import List, Optional, Set
from enum import Enum

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from manifold.models.schemas import QueryModeV2


@dataclass
class GoldenQuery:
    """A test query with expected behavior."""

    query: str
    expected_mode: QueryModeV2
    min_confidence: float = 0.6

    # Expected manifold emphasis (which manifolds should be weighted highly)
    high_manifolds: List[str] = None  # e.g., ["topic", "claim"]
    low_manifolds: List[str] = None   # e.g., ["procedure"]

    # Expected results characteristics
    expected_keywords: List[str] = None  # Keywords that should appear in results
    expected_entity_types: List[str] = None  # Entity types to find
    min_results: int = 1
    max_latency_ms: int = 2000

    # Optional: specific segment IDs that must be returned (for regression)
    must_include_ids: List[str] = None

    # Description for test output
    description: str = ""

    def __post_init__(self):
        if self.high_manifolds is None:
            self.high_manifolds = []
        if self.low_manifolds is None:
            self.low_manifolds = []
        if self.expected_keywords is None:
            self.expected_keywords = []
        if self.expected_entity_types is None:
            self.expected_entity_types = []
        if self.must_include_ids is None:
            self.must_include_ids = []


# === Golden Query Set ===
# Organized by query mode for systematic coverage

GOLDEN_QUERIES = [
    # --- FACT_LOOKUP queries ---
    GoldenQuery(
        query="What's the root password for GitLab?",
        expected_mode=QueryModeV2.FACT_LOOKUP,
        min_confidence=0.8,
        high_manifolds=["claim", "topic"],
        expected_keywords=["password", "gitlab", "root"],
        expected_entity_types=["credential", "service"],
        description="Credential lookup - should find exact fact",
    ),
    GoldenQuery(
        query="What is the IP address of the database server?",
        expected_mode=QueryModeV2.FACT_LOOKUP,
        min_confidence=0.7,
        high_manifolds=["claim", "topic"],
        expected_keywords=["ip", "database", "server"],
        expected_entity_types=["infrastructure", "service"],
        description="Infrastructure fact lookup",
    ),
    GoldenQuery(
        query="What port does PostgreSQL run on?",
        expected_mode=QueryModeV2.FACT_LOOKUP,
        min_confidence=0.7,
        high_manifolds=["claim"],
        expected_keywords=["postgresql", "port"],
        description="Service configuration fact",
    ),

    # --- TIMELINE queries ---
    GoldenQuery(
        query="When did the edge failover happen?",
        expected_mode=QueryModeV2.TIMELINE,
        min_confidence=0.7,
        high_manifolds=["time", "topic"],
        expected_keywords=["failover", "edge"],
        description="Event timing query",
    ),
    GoldenQuery(
        query="What happened on March 28?",
        expected_mode=QueryModeV2.TIMELINE,
        min_confidence=0.8,
        high_manifolds=["time"],
        description="Date-specific timeline query",
    ),
    GoldenQuery(
        query="Show me the sequence of events during the outage",
        expected_mode=QueryModeV2.TIMELINE,
        min_confidence=0.6,
        high_manifolds=["time", "topic"],
        expected_keywords=["outage"],
        description="Event sequence reconstruction",
    ),

    # --- PROCEDURE queries ---
    GoldenQuery(
        query="How do I deploy the application to production?",
        expected_mode=QueryModeV2.PROCEDURE,
        min_confidence=0.8,
        high_manifolds=["procedure"],
        expected_keywords=["deploy", "gami", "production"],
        description="Deployment procedure",
    ),
    GoldenQuery(
        query="What are the steps to configure SSL?",
        expected_mode=QueryModeV2.PROCEDURE,
        min_confidence=0.7,
        high_manifolds=["procedure"],
        expected_keywords=["ssl", "configure"],
        description="Configuration procedure",
    ),
    GoldenQuery(
        query="How to restore a backup?",
        expected_mode=QueryModeV2.PROCEDURE,
        min_confidence=0.7,
        high_manifolds=["procedure"],
        expected_keywords=["backup", "restore"],
        description="Recovery procedure",
    ),

    # --- VERIFICATION queries ---
    GoldenQuery(
        query="Is it true that PostgreSQL runs on port 5433?",
        expected_mode=QueryModeV2.VERIFICATION,
        min_confidence=0.7,
        high_manifolds=["evidence", "claim"],
        expected_keywords=["postgresql", "port", "5433"],
        description="Fact verification",
    ),
    GoldenQuery(
        query="Can you confirm the backup runs at 1 AM?",
        expected_mode=QueryModeV2.VERIFICATION,
        min_confidence=0.7,
        high_manifolds=["evidence", "claim"],
        expected_keywords=["backup", "1 am"],
        description="Schedule verification",
    ),
    GoldenQuery(
        query="Verify that Redis is configured correctly",
        expected_mode=QueryModeV2.VERIFICATION,
        min_confidence=0.6,
        high_manifolds=["evidence"],
        expected_keywords=["redis", "configured"],
        description="Configuration verification",
    ),

    # --- COMPARISON queries ---
    GoldenQuery(
        query="What's the difference between AGE and Neo4j?",
        expected_mode=QueryModeV2.COMPARISON,
        min_confidence=0.7,
        high_manifolds=["relation", "topic"],
        expected_keywords=["age", "neo4j"],
        description="Technology comparison",
    ),
    GoldenQuery(
        query="Compare the old and new authentication systems",
        expected_mode=QueryModeV2.COMPARISON,
        min_confidence=0.6,
        high_manifolds=["relation"],
        expected_keywords=["authentication"],
        description="System comparison",
    ),

    # --- SYNTHESIS queries ---
    GoldenQuery(
        query="Summarize the architecture of the application",
        expected_mode=QueryModeV2.SYNTHESIS,
        min_confidence=0.7,
        high_manifolds=["topic"],
        expected_keywords=["gami", "architecture"],
        description="Architecture summary",
    ),
    GoldenQuery(
        query="Give me an overview of the monitoring setup",
        expected_mode=QueryModeV2.SYNTHESIS,
        min_confidence=0.6,
        high_manifolds=["topic"],
        expected_keywords=["monitoring"],
        description="System overview synthesis",
    ),

    # --- REPORT queries ---
    GoldenQuery(
        query="Give me a comprehensive list of all services",
        expected_mode=QueryModeV2.REPORT,
        min_confidence=0.7,
        high_manifolds=["topic"],
        expected_keywords=["services"],
        min_results=5,
        description="Exhaustive service enumeration",
    ),
    GoldenQuery(
        query="List all credentials and their locations",
        expected_mode=QueryModeV2.REPORT,
        min_confidence=0.6,
        high_manifolds=["claim", "topic"],
        expected_entity_types=["credential"],
        description="Credential inventory report",
    ),

    # --- ASSISTANT_MEMORY queries ---
    GoldenQuery(
        query="Do you remember what we discussed yesterday?",
        expected_mode=QueryModeV2.ASSISTANT_MEMORY,
        min_confidence=0.8,
        high_manifolds=["time", "topic"],
        description="Recent conversation recall",
    ),
    GoldenQuery(
        query="In our previous conversation, you mentioned a workaround",
        expected_mode=QueryModeV2.ASSISTANT_MEMORY,
        min_confidence=0.7,
        high_manifolds=["topic"],
        expected_keywords=["workaround"],
        description="Prior conversation reference",
    ),
]


def get_queries_by_mode(mode: QueryModeV2) -> List[GoldenQuery]:
    """Get all golden queries for a specific mode."""
    return [q for q in GOLDEN_QUERIES if q.expected_mode == mode]


def get_all_modes_covered() -> Set[QueryModeV2]:
    """Get set of all modes covered by golden queries."""
    return {q.expected_mode for q in GOLDEN_QUERIES}


def validate_coverage() -> List[str]:
    """Check that all query modes have golden queries."""
    covered = get_all_modes_covered()
    all_modes = set(QueryModeV2)
    missing = all_modes - covered

    warnings = []
    for mode in missing:
        warnings.append(f"No golden queries for mode: {mode.value}")

    return warnings


# Run basic validation on import
_coverage_warnings = validate_coverage()
if _coverage_warnings:
    import warnings
    for w in _coverage_warnings:
        warnings.warn(w)
