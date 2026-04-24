"""Unit tests for embedding client."""
import pytest
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from manifold.embedding import (
    EmbeddingCache,
    EmbeddingClient,
    EmbeddingResult,
)


class TestEmbeddingCache:
    """Tests for embedding cache."""

    def test_cache_miss(self):
        """Should return None for uncached text."""
        cache = EmbeddingCache()
        result = cache.get("uncached text", "model")
        assert result is None

    def test_cache_hit(self):
        """Should return cached embedding."""
        cache = EmbeddingCache()
        embedding = [0.1, 0.2, 0.3]

        cache.put("test text", "model", embedding)
        result = cache.get("test text", "model")

        assert result == embedding

    def test_cache_different_models(self):
        """Different models should have separate cache entries."""
        cache = EmbeddingCache()

        cache.put("text", "model-a", [0.1])
        cache.put("text", "model-b", [0.2])

        assert cache.get("text", "model-a") == [0.1]
        assert cache.get("text", "model-b") == [0.2]

    def test_cache_eviction(self):
        """Should evict oldest entries when at capacity."""
        cache = EmbeddingCache(max_size=3)

        cache.put("text1", "model", [0.1])
        cache.put("text2", "model", [0.2])
        cache.put("text3", "model", [0.3])
        cache.put("text4", "model", [0.4])  # Should evict text1

        assert cache.get("text1", "model") is None
        assert cache.get("text2", "model") == [0.2]
        assert cache.get("text4", "model") == [0.4]

    def test_cache_lru_order(self):
        """Access should update LRU order."""
        cache = EmbeddingCache(max_size=3)

        cache.put("text1", "model", [0.1])
        cache.put("text2", "model", [0.2])
        cache.put("text3", "model", [0.3])

        # Access text1 to make it most recently used
        cache.get("text1", "model")

        # Add text4 - should evict text2 (least recently used)
        cache.put("text4", "model", [0.4])

        assert cache.get("text1", "model") == [0.1]
        assert cache.get("text2", "model") is None

    def test_cache_clear(self):
        """Clear should remove all entries."""
        cache = EmbeddingCache()

        cache.put("text1", "model", [0.1])
        cache.put("text2", "model", [0.2])

        cache.clear()

        assert cache.get("text1", "model") is None
        assert cache.get("text2", "model") is None


class TestEmbeddingResult:
    """Tests for EmbeddingResult."""

    def test_result_fields(self):
        """Should store all fields."""
        result = EmbeddingResult(
            text="test",
            embedding=[0.1, 0.2],
            model="nomic-embed-text",
            cached=True,
            latency_ms=5.5,
        )

        assert result.text == "test"
        assert result.embedding == [0.1, 0.2]
        assert result.model == "nomic-embed-text"
        assert result.cached is True
        assert result.latency_ms == 5.5


class TestEmbeddingClient:
    """Tests for EmbeddingClient."""

    def test_client_initialization(self):
        """Should initialize with defaults."""
        client = EmbeddingClient()

        assert client.base_url == "http://localhost:11434"
        assert client.model == "nomic-embed-text"
        assert client.batch_size == 32

    def test_client_custom_config(self):
        """Should accept custom configuration."""
        client = EmbeddingClient(
            base_url="http://custom:8080",
            model="custom-model",
            batch_size=64,
            max_retries=5,
        )

        assert client.base_url == "http://custom:8080"
        assert client.model == "custom-model"
        assert client.batch_size == 64
        assert client.max_retries == 5

    def test_client_uses_cache(self):
        """Client should use provided cache."""
        cache = EmbeddingCache()
        client = EmbeddingClient(cache=cache)

        # Pre-populate cache
        cache.put("cached text", "nomic-embed-text", [0.5, 0.5])

        # Check that client sees cached value
        cached_result = client.cache.get("cached text", "nomic-embed-text")
        assert cached_result == [0.5, 0.5]


class TestEmbeddingClientAsync:
    """Async tests for EmbeddingClient."""

    @pytest.mark.asyncio
    async def test_embed_one_cached(self):
        """Should return cached embedding without network call."""
        cache = EmbeddingCache()
        embedding = [0.1] * 768
        cache.put("test text", "nomic-embed-text", embedding)

        client = EmbeddingClient(cache=cache)

        result = await client.embed_one("test text")

        assert result.cached is True
        assert result.embedding == embedding
        assert result.latency_ms == 0.0

    @pytest.mark.asyncio
    async def test_client_close(self):
        """Should close cleanly."""
        client = EmbeddingClient()
        await client.close()
        assert client._client is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
