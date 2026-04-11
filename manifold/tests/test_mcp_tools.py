"""Unit tests for MCP tool implementations."""
import pytest
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from manifold.mcp.tools import ManifoldTools, TOOL_DEFINITIONS, MAX_QUERY_LENGTH
from manifold.exceptions import QueryError


class TestManifoldTools:
    """Tests for ManifoldTools class."""

    @pytest.fixture
    def tools(self):
        return ManifoldTools()

    @pytest.mark.asyncio
    async def test_memory_recall_basic(self, tools):
        """Should return result for valid query."""
        result = await tools.memory_recall(
            query="What is PostgreSQL?",
            top_k=5,
        )

        assert "query" in result
        assert "mode" in result
        assert "confidence" in result
        assert "results" in result
        assert isinstance(result["results"], list)

    @pytest.mark.asyncio
    async def test_memory_recall_empty_query(self, tools):
        """Should reject empty query."""
        with pytest.raises(QueryError) as exc_info:
            await tools.memory_recall(query="")

        assert exc_info.value.code.name == "EMPTY_QUERY"

    @pytest.mark.asyncio
    async def test_memory_recall_whitespace_query(self, tools):
        """Should reject whitespace-only query."""
        with pytest.raises(QueryError):
            await tools.memory_recall(query="   ")

    @pytest.mark.asyncio
    async def test_memory_recall_long_query(self, tools):
        """Should reject query exceeding max length."""
        long_query = "x" * (MAX_QUERY_LENGTH + 1)

        with pytest.raises(QueryError) as exc_info:
            await tools.memory_recall(query=long_query)

        assert exc_info.value.code.name == "QUERY_TOO_LONG"

    @pytest.mark.asyncio
    async def test_memory_recall_top_k_clamped(self, tools):
        """Should clamp top_k to valid range."""
        # Very high top_k
        result = await tools.memory_recall(
            query="test",
            top_k=1000,  # Over max
        )
        assert "results" in result

        # Zero top_k
        result = await tools.memory_recall(
            query="test",
            top_k=0,  # Under min
        )
        assert "results" in result

    @pytest.mark.asyncio
    async def test_memory_recall_mode_override(self, tools):
        """Should accept valid mode override."""
        result = await tools.memory_recall(
            query="test",
            mode="timeline",
        )

        assert result["mode"] == "timeline"

    @pytest.mark.asyncio
    async def test_memory_recall_invalid_mode_ignored(self, tools):
        """Should ignore invalid mode override."""
        result = await tools.memory_recall(
            query="test",
            mode="invalid_mode",
        )

        # Should proceed with auto-classification
        assert result["mode"] in [
            "fact_lookup", "timeline", "procedure", "verification",
            "comparison", "synthesis", "report", "assistant_memory",
        ]

    @pytest.mark.asyncio
    async def test_memory_recall_with_explanation(self, tools):
        """Should include explanation when requested."""
        result = await tools.memory_recall(
            query="How to deploy?",
            explain=True,
        )

        assert "explanation" in result

    @pytest.mark.asyncio
    async def test_memory_classify(self, tools):
        """Should classify query without retrieving."""
        result = await tools.memory_classify(
            query="When did the deployment happen?",
        )

        assert result["mode"] == "timeline"
        assert "manifold_weights" in result
        assert "explanation" in result

    @pytest.mark.asyncio
    async def test_memory_search(self, tools):
        """Should search specific manifold."""
        result = await tools.memory_search(
            query="test",
            manifold="topic",
        )

        assert result["manifold"] == "topic"
        assert "results" in result

    @pytest.mark.asyncio
    async def test_memory_verify(self, tools):
        """Should attempt claim verification."""
        result = await tools.memory_verify(
            claim="PostgreSQL runs on port 5433",
        )

        assert "claim" in result
        assert "verification_status" in result
        assert "supporting_evidence" in result

    @pytest.mark.asyncio
    async def test_manifold_stats(self, tools):
        """Should return stats dict."""
        result = await tools.manifold_stats()

        assert "embeddings" in result
        assert "promoted_objects" in result


class TestToolDefinitions:
    """Tests for tool definitions."""

    def test_definitions_valid(self):
        """All tool definitions should have required fields."""
        required_fields = ["name", "description", "inputSchema"]

        for tool in TOOL_DEFINITIONS:
            for field in required_fields:
                assert field in tool, f"Tool {tool.get('name')} missing {field}"

    def test_input_schema_valid(self):
        """Input schemas should be valid JSON Schema."""
        for tool in TOOL_DEFINITIONS:
            schema = tool["inputSchema"]
            assert schema["type"] == "object"
            assert "properties" in schema

    def test_required_fields_present(self):
        """Required fields should be in properties."""
        for tool in TOOL_DEFINITIONS:
            schema = tool["inputSchema"]
            required = schema.get("required", [])
            properties = schema["properties"]

            for field in required:
                assert field in properties, (
                    f"Tool {tool['name']} has required field {field} "
                    f"not in properties"
                )

    def test_memory_recall_definition(self):
        """memory_recall should have expected parameters."""
        recall_tool = next(t for t in TOOL_DEFINITIONS if t["name"] == "memory_recall")

        props = recall_tool["inputSchema"]["properties"]
        assert "query" in props
        assert "top_k" in props
        assert "tenant_id" in props
        assert "mode" in props

    def test_tool_count(self):
        """Should have expected number of tools."""
        assert len(TOOL_DEFINITIONS) >= 5  # At least 5 tools


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
