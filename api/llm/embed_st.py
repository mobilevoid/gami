"""Universal embedding via sentence-transformers.

Identical results on CPU and GPU. No Ollama dependency.
Use gami-embed conda env: /home/ai/.conda/envs/gami-embed/bin/python

For use inside GAMI API (which runs on system python), this module
shells out to the gami-embed env. For batch scripts, import directly.
"""
import json
import logging
import subprocess
import os

logger = logging.getLogger("gami.embed")

# Path to the gami-embed python
EMBED_PYTHON = None
for p in [
    "/home/ai/.conda/envs/gami-embed/bin/python",
    os.path.expanduser("~/.conda/envs/gami-embed/bin/python"),
]:
    if os.path.exists(p):
        EMBED_PYTHON = p
        break

MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"

# Inline script that the subprocess runs
_EMBED_SCRIPT = '''
import sys, json
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("{model}", trust_remote_code=True)
texts = json.loads(sys.stdin.read())
embs = model.encode(texts, normalize_embeddings=False, show_progress_bar=False)
print(json.dumps(embs.tolist()))
'''


def embed_texts_subprocess(texts: list[str]) -> list[list[float]]:
    """Embed texts via subprocess to gami-embed env."""
    if not EMBED_PYTHON:
        raise RuntimeError("gami-embed conda env not found")
    
    script = _EMBED_SCRIPT.format(model=MODEL_NAME)
    result = subprocess.run(
        [EMBED_PYTHON, "-c", script],
        input=json.dumps(texts),
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Embedding failed: {result.stderr[:500]}")
    return json.loads(result.stdout)


def embed_text_sync(text: str) -> list[float]:
    """Embed a single text synchronously."""
    results = embed_texts_subprocess([text[:2000]])
    return results[0]


async def embed_text(text: str, is_query: bool = False) -> list[float]:
    """Async wrapper for embedding."""
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, embed_text_sync, text[:2000])
