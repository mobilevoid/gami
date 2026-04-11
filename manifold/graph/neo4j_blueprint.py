"""Neo4j migration blueprint — stubs and specifications for future migration.

This module provides:
1. Neo4j schema specifications
2. Data migration interfaces
3. Driver integration stubs
4. Sync mechanism design

IMPORTANT: This is a BLUEPRINT only. No actual Neo4j connections are made.
The current production system uses Apache AGE.

Migration triggers (when to consider Neo4j):
1. Vector-native graph retrieval needed
2. Graph algorithms (PageRank, community detection) needed
3. AGE becomes limiting for query expressiveness
"""
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger("manifold.graph.neo4j")


# ---------------------------------------------------------------------------
# Neo4j Schema Specification
# ---------------------------------------------------------------------------

NEO4J_NODE_LABELS = [
    "Source",
    "Segment",
    "Entity",
    "Concept",
    "Claim",
    "Event",
    "Summary",
    "AssistantMemory",
    "TopicCluster",
    "ContradictionGroup",
    "Procedure",
]

NEO4J_RELATIONSHIP_TYPES = [
    "MENTIONS",
    "DERIVED_FROM",
    "SUMMARIZES",
    "PART_OF",
    "INSTANCE_OF",
    "RELATED_TO",
    "ALIAS_OF",
    "SUPPORTS",
    "CONTRADICTS",
    "CAUSED_BY",
    "PRECEDES",
    "FOLLOWS",
    "ABOUT",
    "PREFERENCE_OF",
    "CORRECTION_OF",
    "UPDATED_BY",
    "SUPERSEDED_BY",
    "LOCATED_IN",
    "AUTHORED_BY",
    "USES",
    "DEFINED_BY",
    "ELABORATES",
]

NEO4J_SCHEMA_CYPHER = """
// Node constraints
CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT segment_id IF NOT EXISTS FOR (s:Segment) REQUIRE s.id IS UNIQUE;
CREATE CONSTRAINT claim_id IF NOT EXISTS FOR (c:Claim) REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT source_id IF NOT EXISTS FOR (s:Source) REQUIRE s.id IS UNIQUE;
CREATE CONSTRAINT memory_id IF NOT EXISTS FOR (m:AssistantMemory) REQUIRE m.id IS UNIQUE;

// Indexes for common queries
CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name);
CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type);
CREATE INDEX segment_tenant IF NOT EXISTS FOR (s:Segment) ON (s.tenant_id);
CREATE INDEX claim_status IF NOT EXISTS FOR (c:Claim) ON (c.status);

// Vector indexes (Neo4j 5.x+ with GDS)
// These require the Graph Data Science plugin
// CALL db.index.vector.createNodeIndex('entity_embedding', 'Entity', 'embedding', 768, 'cosine')
// CALL db.index.vector.createNodeIndex('segment_embedding', 'Segment', 'embedding', 768, 'cosine')
"""


# ---------------------------------------------------------------------------
# Data Migration Interfaces
# ---------------------------------------------------------------------------

@dataclass
class Neo4jNode:
    """Node representation for Neo4j import."""
    id: str
    labels: List[str]
    properties: Dict[str, Any]

    def to_cypher(self) -> str:
        """Generate Cypher CREATE statement."""
        labels_str = ":".join(self.labels)
        props_str = ", ".join(
            f"{k}: ${k}" for k in self.properties.keys()
        )
        return f"CREATE (n:{labels_str} {{{props_str}}})"


@dataclass
class Neo4jRelationship:
    """Relationship representation for Neo4j import."""
    source_id: str
    target_id: str
    type: str
    properties: Dict[str, Any]

    def to_cypher(self) -> str:
        """Generate Cypher CREATE statement."""
        props_str = ", ".join(
            f"{k}: ${k}" for k in self.properties.keys()
        )
        return f"""
        MATCH (a {{id: $source_id}}), (b {{id: $target_id}})
        CREATE (a)-[r:{self.type} {{{props_str}}}]->(b)
        """


class Neo4jBlueprint:
    """Blueprint for Neo4j migration.

    Provides interfaces for:
    - Schema creation
    - Data export from AGE
    - Data import to Neo4j
    - Sync mechanism

    IMPORTANT: This class does NOT connect to Neo4j.
    It provides the specification and stubs for future implementation.
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        auth: tuple = ("neo4j", "password"),
    ):
        """Initialize blueprint (does not connect).

        Args:
            uri: Neo4j connection URI.
            auth: Authentication tuple (user, password).
        """
        self.uri = uri
        self.auth = auth
        self._connected = False

        logger.info("Neo4j blueprint initialized (NOT CONNECTED)")

    def get_schema_cypher(self) -> str:
        """Get schema creation Cypher.

        Returns:
            Cypher script for schema creation.
        """
        return NEO4J_SCHEMA_CYPHER

    def get_node_labels(self) -> List[str]:
        """Get node labels.

        Returns:
            List of node labels.
        """
        return NEO4J_NODE_LABELS.copy()

    def get_relationship_types(self) -> List[str]:
        """Get relationship types.

        Returns:
            List of relationship types.
        """
        return NEO4J_RELATIONSHIP_TYPES.copy()

    def export_from_age(self) -> Dict[str, Any]:
        """Export graph data from AGE.

        NOTE: This is a stub. Actual implementation will query AGE
        and transform to Neo4j format.

        Returns:
            Dict with nodes and relationships.
        """
        logger.warning("export_from_age is a stub - not connected to AGE")
        return {
            "nodes": [],
            "relationships": [],
            "node_count": 0,
            "relationship_count": 0,
        }

    def import_to_neo4j(
        self,
        nodes: List[Neo4jNode],
        relationships: List[Neo4jRelationship],
    ) -> Dict[str, Any]:
        """Import data to Neo4j.

        NOTE: This is a stub. Actual implementation will use the
        Neo4j Python driver to import data.

        Args:
            nodes: Nodes to import.
            relationships: Relationships to import.

        Returns:
            Import statistics.
        """
        logger.warning("import_to_neo4j is a stub - not connected to Neo4j")
        return {
            "nodes_created": 0,
            "relationships_created": 0,
            "status": "stub",
        }

    def sync_from_postgres(
        self,
        since_timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Sync changes from PostgreSQL to Neo4j.

        Uses transactional outbox pattern:
        1. Read outbox table for pending changes
        2. Apply to Neo4j
        3. Mark outbox entries as processed

        NOTE: This is a stub.

        Args:
            since_timestamp: Only sync changes after this time.

        Returns:
            Sync statistics.
        """
        logger.warning("sync_from_postgres is a stub")
        return {
            "synced_nodes": 0,
            "synced_relationships": 0,
            "status": "stub",
        }

    def vector_search(
        self,
        embedding: List[float],
        label: str = "Entity",
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Vector search in Neo4j.

        NOTE: This is a stub. Requires Neo4j 5.x+ with GDS plugin.

        Args:
            embedding: Query embedding.
            label: Node label to search.
            limit: Maximum results.

        Returns:
            List of matching nodes with scores.
        """
        logger.warning("vector_search is a stub - not connected to Neo4j")
        return []

    def run_pagerank(
        self,
        label: str = "Entity",
    ) -> Dict[str, float]:
        """Run PageRank on entity graph.

        NOTE: This is a stub. Requires Neo4j GDS plugin.

        Args:
            label: Node label for PageRank.

        Returns:
            Dict of node_id to PageRank score.
        """
        logger.warning("run_pagerank is a stub - not connected to Neo4j")
        return {}

    def detect_communities(
        self,
        label: str = "Entity",
    ) -> Dict[str, int]:
        """Detect communities in graph.

        NOTE: This is a stub. Requires Neo4j GDS plugin.

        Args:
            label: Node label for community detection.

        Returns:
            Dict of node_id to community_id.
        """
        logger.warning("detect_communities is a stub - not connected to Neo4j")
        return {}


# ---------------------------------------------------------------------------
# Migration Status
# ---------------------------------------------------------------------------

def get_migration_status() -> Dict[str, Any]:
    """Get current Neo4j migration status.

    Returns:
        Dict with migration status information.
    """
    return {
        "neo4j_enabled": False,
        "current_graph_engine": "apache_age",
        "migration_ready": False,
        "migration_triggers": [
            "vector_native_retrieval_needed",
            "graph_algorithms_needed",
            "age_limiting",
        ],
        "estimated_migration_hours": 50,
        "recommendation": "Keep AGE for now. Revisit after multi-manifold is working.",
    }
