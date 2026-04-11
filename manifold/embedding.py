"""Embedding client for manifold system.

Provides async embedding generation with:
- Batching for efficiency
- Rate limiting to prevent overload
- Retry logic for resilience
- Caching for repeated texts
"""
import asyncio
import logging
import hashlib
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger("gami.manifold.embedding")


@dataclass
class EmbeddingResult:
    """Result of an embedding operation."""
    text: str
    embedding: List[float]
    model: str
    cached: bool = False
    latency_ms: float = 0.0


class EmbeddingCache:
    """In-memory LRU cache for embeddings."""

    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Tuple[List[float], datetime]] = {}
        self._access_order: List[str] = []

    def _hash_text(self, text: str, model: str) -> str:
        """Generate cache key from text and model."""
        content = f"{model}:{text}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, text: str, model: str) -> Optional[List[float]]:
        """Get cached embedding if available and not expired."""
        key = self._hash_text(text, model)
        if key not in self._cache:
            return None

        embedding, timestamp = self._cache[key]
        if datetime.now() - timestamp > timedelta(seconds=self.ttl_seconds):
            # Expired
            del self._cache[key]
            self._access_order.remove(key)
            return None

        # Move to end of access order
        self._access_order.remove(key)
        self._access_order.append(key)

        return embedding

    def put(self, text: str, model: str, embedding: List[float]):
        """Cache an embedding."""
        key = self._hash_text(text, model)

        # Evict if at capacity
        while len(self._cache) >= self.max_size:
            oldest_key = self._access_order.pop(0)
            del self._cache[oldest_key]

        self._cache[key] = (embedding, datetime.now())
        self._access_order.append(key)

    def clear(self):
        """Clear all cached embeddings."""
        self._cache.clear()
        self._access_order.clear()


class EmbeddingClient:
    """Async client for embedding generation."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
        batch_size: int = 32,
        max_retries: int = 3,
        timeout_seconds: float = 30.0,
        rate_limit_per_second: float = 10.0,
        cache: Optional[EmbeddingCache] = None,
    ):
        """Initialize embedding client.

        Args:
            base_url: Ollama server URL.
            model: Embedding model name.
            batch_size: Max texts per batch request.
            max_retries: Retry count for failed requests.
            timeout_seconds: Request timeout.
            rate_limit_per_second: Max requests per second.
            cache: Optional embedding cache.
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.rate_limit_interval = 1.0 / rate_limit_per_second
        self.cache = cache or EmbeddingCache()

        self._last_request_time = 0.0
        self._client = None

    async def _get_client(self):
        """Get or create HTTP client."""
        if self._client is None:
            try:
                import httpx
                self._client = httpx.AsyncClient(timeout=self.timeout_seconds)
            except ImportError:
                raise RuntimeError("httpx required: pip install httpx")
        return self._client

    async def _rate_limit(self):
        """Apply rate limiting."""
        import time
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self.rate_limit_interval:
            await asyncio.sleep(self.rate_limit_interval - elapsed)
        self._last_request_time = time.time()

    async def embed_one(self, text: str) -> EmbeddingResult:
        """Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            EmbeddingResult with embedding vector.
        """
        # Check cache first
        cached = self.cache.get(text, self.model)
        if cached is not None:
            return EmbeddingResult(
                text=text,
                embedding=cached,
                model=self.model,
                cached=True,
            )

        import time
        start = time.time()

        # Rate limit
        await self._rate_limit()

        # Make request with retries
        client = await self._get_client()
        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = await client.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model, "prompt": text},
                )
                response.raise_for_status()
                data = response.json()
                embedding = data["embedding"]

                # Cache result
                self.cache.put(text, self.model, embedding)

                latency_ms = (time.time() - start) * 1000
                return EmbeddingResult(
                    text=text,
                    embedding=embedding,
                    model=self.model,
                    cached=False,
                    latency_ms=latency_ms,
                )

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Embedding attempt {attempt + 1}/{self.max_retries} "
                    f"failed: {e}"
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff

        raise RuntimeError(f"Embedding failed after {self.max_retries} attempts: {last_error}")

    async def embed_many(
        self,
        texts: List[str],
        show_progress: bool = False,
    ) -> List[EmbeddingResult]:
        """Generate embeddings for multiple texts.

        Args:
            texts: Texts to embed.
            show_progress: Whether to log progress.

        Returns:
            List of EmbeddingResults in same order as input.
        """
        results = []
        total = len(texts)

        for i in range(0, total, self.batch_size):
            batch = texts[i:i + self.batch_size]

            if show_progress:
                logger.info(f"Embedding batch {i // self.batch_size + 1}, "
                           f"items {i + 1}-{min(i + len(batch), total)}/{total}")

            # Process batch (currently sequential, could be parallel)
            batch_results = await asyncio.gather(
                *[self.embed_one(text) for text in batch],
                return_exceptions=True,
            )

            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to embed text {i + j}: {result}")
                    # Return zero vector for failed embeddings
                    results.append(EmbeddingResult(
                        text=batch[j],
                        embedding=[0.0] * 768,  # Assume 768-dim
                        model=self.model,
                        cached=False,
                    ))
                else:
                    results.append(result)

        return results

    async def close(self):
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None


# Convenience function
async def embed_text(
    text: str,
    model: str = "nomic-embed-text",
    base_url: str = "http://localhost:11434",
) -> List[float]:
    """Embed a single text and return the vector.

    Args:
        text: Text to embed.
        model: Embedding model name.
        base_url: Ollama server URL.

    Returns:
        Embedding vector.
    """
    client = EmbeddingClient(base_url=base_url, model=model)
    try:
        result = await client.embed_one(text)
        return result.embedding
    finally:
        await client.close()


async def embed_texts(
    texts: List[str],
    model: str = "nomic-embed-text",
    base_url: str = "http://localhost:11434",
    batch_size: int = 32,
) -> List[List[float]]:
    """Embed multiple texts and return vectors.

    Args:
        texts: Texts to embed.
        model: Embedding model name.
        base_url: Ollama server URL.
        batch_size: Batch size for requests.

    Returns:
        List of embedding vectors.
    """
    client = EmbeddingClient(
        base_url=base_url,
        model=model,
        batch_size=batch_size,
    )
    try:
        results = await client.embed_many(texts)
        return [r.embedding for r in results]
    finally:
        await client.close()
