"""Unit tests for relation manifold scoring.

Tests graph fingerprinting and relation score computation.
"""
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from manifold.scoring.relation import (
    GraphFingerprint,
    compute_graph_fingerprint,
    fingerprint_similarity,
    compute_relation_score,
    find_related_entities,
    categorize_edge_type,
)


class TestGraphFingerprint:
    """Tests for graph fingerprint computation."""

    def test_empty_graph(self):
        """Entity with no edges should have empty fingerprint."""
        fp = compute_graph_fingerprint(
            entity_id="e1",
            entity_type="service",
            edges=[],
            nodes={},
        )

        assert fp.in_degree == 0
        assert fp.out_degree == 0
        assert fp.out_edges == {}
        assert fp.in_edges == {}

    def test_outgoing_edges(self):
        """Should count outgoing edges correctly."""
        edges = [
            {"source": "e1", "target": "e2", "type": "uses"},
            {"source": "e1", "target": "e3", "type": "uses"},
            {"source": "e1", "target": "e4", "type": "depends_on"},
        ]
        nodes = {
            "e2": {"type": "database"},
            "e3": {"type": "cache"},
            "e4": {"type": "service"},
        }

        fp = compute_graph_fingerprint("e1", "service", edges, nodes)

        assert fp.out_degree == 3
        assert fp.out_edges["uses"] == 2
        assert fp.out_edges["depends_on"] == 1

    def test_incoming_edges(self):
        """Should count incoming edges correctly."""
        edges = [
            {"source": "e2", "target": "e1", "type": "calls"},
            {"source": "e3", "target": "e1", "type": "calls"},
        ]
        nodes = {"e2": {"type": "api"}, "e3": {"type": "api"}}

        fp = compute_graph_fingerprint("e1", "service", edges, nodes)

        assert fp.in_degree == 2
        assert fp.in_edges["calls"] == 2

    def test_connected_types(self):
        """Should track connected entity types."""
        edges = [
            {"source": "e1", "target": "e2", "type": "uses"},
            {"source": "e1", "target": "e3", "type": "uses"},
            {"source": "e4", "target": "e1", "type": "calls"},
        ]
        nodes = {
            "e2": {"type": "database"},
            "e3": {"type": "database"},
            "e4": {"type": "api"},
        }

        fp = compute_graph_fingerprint("e1", "service", edges, nodes)

        assert fp.connected_types["database"] == 2
        assert fp.connected_types["api"] == 1

    def test_signature_deterministic(self):
        """Same fingerprint should produce same signature."""
        edges = [{"source": "e1", "target": "e2", "type": "uses"}]
        nodes = {"e2": {"type": "database"}}

        fp1 = compute_graph_fingerprint("e1", "service", edges, nodes)
        fp2 = compute_graph_fingerprint("e1", "service", edges, nodes)

        assert fp1.signature() == fp2.signature()


class TestFingerprintSimilarity:
    """Tests for fingerprint similarity computation."""

    def test_identical_fingerprints(self):
        """Identical fingerprints should have similarity 1.0."""
        fp = GraphFingerprint(
            entity_id="e1",
            entity_type="service",
            out_edges={"uses": 2},
            in_edges={"calls": 1},
            connected_types={"database": 2},
        )

        sim = fingerprint_similarity(fp, fp)
        assert sim == 1.0

    def test_disjoint_fingerprints(self):
        """Completely different fingerprints should have low similarity."""
        fp1 = GraphFingerprint(
            entity_id="e1",
            entity_type="service",
            out_edges={"uses": 2},
            in_edges={},
            connected_types={"database": 2},
        )
        fp2 = GraphFingerprint(
            entity_id="e2",
            entity_type="api",
            out_edges={"calls": 5},
            in_edges={"triggers": 3},
            connected_types={"queue": 3},
        )

        sim = fingerprint_similarity(fp1, fp2)
        assert sim < 0.3

    def test_partial_overlap(self):
        """Partial overlap should yield intermediate similarity."""
        fp1 = GraphFingerprint(
            entity_id="e1",
            entity_type="service",
            out_edges={"uses": 2, "depends_on": 1},
            connected_types={"database": 2, "cache": 1},
        )
        fp2 = GraphFingerprint(
            entity_id="e2",
            entity_type="service",
            out_edges={"uses": 3, "calls": 2},
            connected_types={"database": 1, "api": 2},
        )

        sim = fingerprint_similarity(fp1, fp2)
        assert 0.2 < sim < 0.8


class TestRelationScore:
    """Tests for relation score computation."""

    def test_close_entities_high_score(self):
        """Closely related entities should score high."""
        fp1 = GraphFingerprint(entity_id="e1", entity_type="service",
                               out_edges={"uses": 2}, connected_types={"database": 2})
        fp2 = GraphFingerprint(entity_id="e2", entity_type="service",
                               out_edges={"uses": 2}, connected_types={"database": 2})

        score = compute_relation_score(
            "e1", "e2", fp1, fp2,
            shared_neighbors=3,
            path_length=1,
        )

        assert score >= 0.5

    def test_distant_entities_low_score(self):
        """Distant entities should score lower."""
        fp1 = GraphFingerprint(entity_id="e1", entity_type="service",
                               out_edges={"uses": 2})
        fp2 = GraphFingerprint(entity_id="e2", entity_type="config",
                               out_edges={"defines": 5})

        score = compute_relation_score(
            "e1", "e2", fp1, fp2,
            shared_neighbors=0,
            path_length=5,
        )

        assert score < 0.5

    def test_shared_neighbors_boost(self):
        """More shared neighbors should boost score."""
        fp = GraphFingerprint(entity_id="e1", entity_type="service")

        score_low = compute_relation_score("e1", "e2", fp, fp, shared_neighbors=0)
        score_high = compute_relation_score("e1", "e2", fp, fp, shared_neighbors=5)

        assert score_high > score_low


class TestFindRelatedEntities:
    """Tests for finding related entities."""

    def test_finds_similar_entities(self):
        """Should find entities with similar fingerprints."""
        query = GraphFingerprint(
            entity_id="q",
            entity_type="service",
            out_edges={"uses": 2},
            connected_types={"database": 2},
        )

        candidates = [
            GraphFingerprint(entity_id="c1", entity_type="service",
                           out_edges={"uses": 2}, connected_types={"database": 2}),
            GraphFingerprint(entity_id="c2", entity_type="api",
                           out_edges={"calls": 5}, connected_types={"queue": 3}),
            GraphFingerprint(entity_id="c3", entity_type="service",
                           out_edges={"uses": 1}, connected_types={"database": 1}),
        ]

        results = find_related_entities(query, candidates, min_similarity=0.3)

        # c1 and c3 should be most similar
        result_ids = [r[0] for r in results]
        assert "c1" in result_ids
        assert "c3" in result_ids

    def test_excludes_self(self):
        """Should not include query entity in results."""
        query = GraphFingerprint(entity_id="q", entity_type="service")
        candidates = [query]

        results = find_related_entities(query, candidates)
        assert len(results) == 0

    def test_respects_max_results(self):
        """Should limit results to max_results."""
        query = GraphFingerprint(entity_id="q", entity_type="service",
                                out_edges={"uses": 1})
        candidates = [
            GraphFingerprint(entity_id=f"c{i}", entity_type="service",
                           out_edges={"uses": 1})
            for i in range(10)
        ]

        results = find_related_entities(query, candidates, max_results=3)
        assert len(results) <= 3


class TestEdgeTypeTaxonomy:
    """Tests for edge type categorization."""

    def test_structural_types(self):
        """Should categorize structural edge types."""
        assert categorize_edge_type("contains") == "structural"
        assert categorize_edge_type("part_of") == "structural"
        assert categorize_edge_type("instance_of") == "structural"

    def test_operational_types(self):
        """Should categorize operational edge types."""
        assert categorize_edge_type("uses") == "operational"
        assert categorize_edge_type("depends_on") == "operational"
        assert categorize_edge_type("calls") == "operational"

    def test_temporal_types(self):
        """Should categorize temporal edge types."""
        assert categorize_edge_type("precedes") == "temporal"
        assert categorize_edge_type("triggers") == "temporal"

    def test_unknown_type(self):
        """Unknown types should be 'other'."""
        assert categorize_edge_type("custom_relation") == "other"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
