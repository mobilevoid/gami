"""Universal embedding service for GAMI using sentence-transformers.

Produces IDENTICAL embeddings on CPU and GPU — no Ollama dependency.
Uses nomic-ai/nomic-embed-text-v1.5 (768 dims).

For the GAMI API process (system python), shells out to gami-embed conda env.
For batch scripts running in gami-embed env, loads the model directly.
"""
import asyncio
import json
import logging
import os
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)

MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"

# Find the gami-embed python
EMBED_PYTHON = None
for p in [
    "/home/ai/.conda/envs/gami-embed/bin/python",
    os.path.expanduser("~/.conda/envs/gami-embed/bin/python"),
]:
    if os.path.exists(p):
        EMBED_PYTHON = p
        break

# Try to load model directly (works in gami-embed env)
_model = None
_direct_available = False

try:
    from sentence_transformers import SentenceTransformer
    import torch
    _direct_available = True
    logger.info("sentence-transformers available — will load model on first use")
except ImportError:
    logger.info("sentence-transformers not in this env — will use subprocess to gami-embed")


def _get_model():
    """Lazy-load the model (only in envs with sentence-transformers)."""
    global _model
    if _model is None and _direct_available:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = SentenceTransformer(MODEL_NAME, trust_remote_code=True, device=device)
        logger.info(f"Embedding model loaded on {device}")
    return _model


def _embed_direct(texts: list[str]) -> list[list[float]]:
    """Embed using direct model (same process)."""
    model = _get_model()
    truncated = [t[:2000] for t in texts]
    embs = model.encode(truncated, normalize_embeddings=False, show_progress_bar=False)
    return embs.tolist()


def _embed_subprocess(texts: list[str]) -> list[list[float]]:
    """Embed via subprocess to gami-embed env."""
    if not EMBED_PYTHON:
        raise RuntimeError(
            "gami-embed conda env not found. Install with:\n"
            "  conda create -n gami-embed python=3.10 -y\n"
            "  /home/ai/.conda/envs/gami-embed/bin/pip install torch sentence-transformers einops"
        )
    script = (
        "import sys, json\n"
        "from sentence_transformers import SentenceTransformer\n"
        f"model = SentenceTransformer('{MODEL_NAME}', trust_remote_code=True)\n"
        "texts = json.loads(sys.stdin.read())\n"
        "embs = model.encode([t[:2000] for t in texts], normalize_embeddings=False, show_progress_bar=False)\n"
        "print(json.dumps(embs.tolist()))\n"
    )
    result = subprocess.run(
        [EMBED_PYTHON, "-c", script],
        input=json.dumps(texts),
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Embedding subprocess failed: {result.stderr[:500]}")
    return json.loads(result.stdout)


def embed_texts_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts. Uses direct model if available, else subprocess."""
    if _direct_available:
        return _embed_direct(texts)
    return _embed_subprocess(texts)


def embed_text_sync(text: str, is_query: bool = False) -> list[float]:
    """Embed a single text synchronously."""
    results = embed_texts_batch([text])
    return results[0]


def embed_texts_sync(texts: list[str]) -> list[list[float]]:
    """Sync batch embedding."""
    return embed_texts_batch(texts)


async def embed_text(text: str, is_query: bool = False) -> list[float]:
    """Async wrapper for embedding a single text."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, embed_text_sync, text)


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Async wrapper for batch embedding."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, embed_texts_batch, texts)
