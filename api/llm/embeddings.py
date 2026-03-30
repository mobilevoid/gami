"""Embedding service for GAMI using Ollama nomic-embed-text."""
import logging
from typing import Optional

import httpx

from api.config import settings

logger = logging.getLogger(__name__)


async def embed_text(text: str, is_query: bool = False) -> list[float]:
    """Get embedding for a single text using Ollama nomic-embed-text."""
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{settings.OLLAMA_URL}/api/embeddings",
            json={"model": settings.EMBEDDING_MODEL, "prompt": ("search_query: " + text if is_query else text)},
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()["embedding"]


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch embed multiple texts.

    Sends one request per text (Ollama doesn't support true batching).
    Uses a single client for connection pooling.
    """
    results: list[list[float]] = []
    async with httpx.AsyncClient(timeout=30.0) as client:
        for text in texts:
            resp = await client.post(
                f"{settings.OLLAMA_URL}/api/embeddings",
                json={"model": settings.EMBEDDING_MODEL, "prompt": ("search_query: " + text if is_query else text)},
            )
            resp.raise_for_status()
            results.append(resp.json()["embedding"])
    return results


def embed_text_sync(text: str, is_query: bool = False) -> list[float]:
    """Sync version for Celery workers."""
    import requests

    resp = requests.post(
        f"{settings.OLLAMA_URL}/api/embeddings",
        json={"model": settings.EMBEDDING_MODEL, "prompt": ("search_query: " + text if is_query else text)},
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def embed_texts_sync(texts: list[str]) -> list[list[float]]:
    """Sync batch embedding for Celery workers."""
    import requests

    results: list[list[float]] = []
    session = requests.Session()
    for text in texts:
        resp = session.post(
            f"{settings.OLLAMA_URL}/api/embeddings",
            json={"model": settings.EMBEDDING_MODEL, "prompt": ("search_query: " + text if is_query else text)},
            timeout=30.0,
        )
        resp.raise_for_status()
        results.append(resp.json()["embedding"])
    session.close()
    return results
