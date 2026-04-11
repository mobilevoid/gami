#!/usr/bin/env python3
"""Embed all unembedded segments for a specific GAMI tenant.

Usage:
    python embed_tenant.py --tenant books --batch-size 50 --workers 4
    python embed_tenant.py --tenant books --gpu   # use sentence-transformers if available

Features:
    - Embeds only segments missing embeddings for the target tenant
    - Progress reporting with ETA
    - Robust error handling (retries on Ollama 500s, skips persistent failures)
    - Resume capability (only processes NULL-embedding segments)
    - Optional GPU batch mode via sentence-transformers
"""
import argparse
import logging
import os
import sys
import time
from typing import Optional

GAMI_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if GAMI_ROOT not in sys.path:
    sys.path.insert(0, GAMI_ROOT)

from api.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("embed_tenant")


def get_engine():
    from sqlalchemy import create_engine
    return create_engine(
        settings.DATABASE_URL_SYNC,
        pool_size=3,
        max_overflow=2,
        pool_pre_ping=True,
    )


def count_unembedded(engine, tenant_id: str) -> int:
    from sqlalchemy import text
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT COUNT(*) FROM segments "
                "WHERE owner_tenant_id = :tid AND embedding IS NULL"
            ),
            {"tid": tenant_id},
        ).fetchone()
    return row[0]


def fetch_batch(engine, tenant_id: str, batch_size: int) -> list[tuple[str, str]]:
    """Fetch a batch of (segment_id, text) for segments missing embeddings."""
    from sqlalchemy import text
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT segment_id, text FROM segments "
                "WHERE owner_tenant_id = :tid AND embedding IS NULL "
                "ORDER BY ordinal "
                "LIMIT :lim"
            ),
            {"tid": tenant_id, "lim": batch_size},
        ).fetchall()
    return [(r[0], r[1]) for r in rows]


def embed_ollama_single(text: str, retries: int = 3) -> Optional[list[float]]:
    """Embed a single text via Ollama with retries."""
    import requests

    for attempt in range(retries):
        try:
            resp = requests.post(
                f"{settings.OLLAMA_URL}/api/embeddings",
                json={"model": settings.EMBEDDING_MODEL, "prompt": text},
                timeout=60.0,
            )
            if resp.status_code == 200:
                return resp.json()["embedding"]
            elif resp.status_code >= 500:
                logger.warning("Ollama 500 error (attempt %d/%d)", attempt + 1, retries)
                time.sleep(2 ** attempt)
            else:
                logger.error("Ollama returned %d: %s", resp.status_code, resp.text[:200])
                return None
        except requests.exceptions.ConnectionError:
            logger.warning("Ollama connection error (attempt %d/%d)", attempt + 1, retries)
            time.sleep(5)
        except requests.exceptions.Timeout:
            logger.warning("Ollama timeout (attempt %d/%d)", attempt + 1, retries)
            time.sleep(2)
        except Exception as e:
            logger.error("Unexpected embedding error: %s", e)
            return None
    return None


def embed_batch_ollama(texts: list[str]) -> list[Optional[list[float]]]:
    """Embed multiple texts via Ollama (sequential, with connection reuse)."""
    import requests

    session = requests.Session()
    results = []

    for text in texts:
        embedding = None
        for attempt in range(3):
            try:
                resp = session.post(
                    f"{settings.OLLAMA_URL}/api/embeddings",
                    json={"model": settings.EMBEDDING_MODEL, "prompt": text},
                    timeout=60.0,
                )
                if resp.status_code == 200:
                    embedding = resp.json()["embedding"]
                    break
                elif resp.status_code >= 500:
                    time.sleep(2 ** attempt)
                else:
                    break
            except Exception:
                time.sleep(2)

        results.append(embedding)

    session.close()
    return results


def try_gpu_embed(texts: list[str]) -> Optional[list[list[float]]]:
    """Try to use sentence-transformers GPU batch embedding."""
    try:
        from sentence_transformers import SentenceTransformer
        import torch

        if not torch.cuda.is_available():
            return None

        model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
        model = model.to("cuda")
        embeddings = model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
        return [emb.tolist() for emb in embeddings]
    except ImportError:
        return None
    except Exception as e:
        logger.warning("GPU embedding failed: %s", e)
        return None


def store_embeddings(engine, segment_embeddings: list[tuple[str, list[float]]]):
    """Store embeddings in the database."""
    from sqlalchemy import text

    with engine.connect() as conn:
        for seg_id, embedding in segment_embeddings:
            vec_str = "[" + ",".join(str(v) for v in embedding) + "]"
            conn.execute(
                text(
                    "UPDATE segments SET embedding = CAST(:vec AS vector) "
                    "WHERE segment_id = :sid"
                ),
                {"vec": vec_str, "sid": seg_id},
            )
        conn.commit()


def main():
    parser = argparse.ArgumentParser(description="Embed unembedded segments for a GAMI tenant")
    parser.add_argument("--tenant", required=True, help="Tenant ID")
    parser.add_argument("--batch-size", type=int, default=50, help="Segments per batch")
    parser.add_argument("--gpu", action="store_true", help="Try GPU batch mode via sentence-transformers")
    parser.add_argument("--limit", type=int, default=0, help="Max segments to embed (0 = all)")
    args = parser.parse_args()

    engine = get_engine()
    tenant_id = args.tenant

    # Check tenant exists
    from sqlalchemy import text as sql_text
    with engine.connect() as conn:
        row = conn.execute(
            sql_text("SELECT tenant_id FROM tenants WHERE tenant_id = :tid"),
            {"tid": tenant_id},
        ).fetchone()
        if not row:
            logger.error("Tenant '%s' does not exist", tenant_id)
            engine.dispose()
            return

    total_unembedded = count_unembedded(engine, tenant_id)
    if total_unembedded == 0:
        logger.info("All segments for tenant '%s' are already embedded", tenant_id)
        engine.dispose()
        return

    target = args.limit if args.limit > 0 else total_unembedded
    target = min(target, total_unembedded)
    logger.info("Embedding %d segments for tenant '%s' (batch_size=%d)", target, tenant_id, args.batch_size)

    # Check if GPU mode is viable
    use_gpu = False
    if args.gpu:
        try:
            import torch
            if torch.cuda.is_available():
                from sentence_transformers import SentenceTransformer
                use_gpu = True
                logger.info("GPU mode enabled — using sentence-transformers")
            else:
                logger.info("No GPU available, falling back to Ollama CPU")
        except ImportError:
            logger.info("sentence-transformers not installed, using Ollama CPU")

    embedded_count = 0
    failed_count = 0
    start_time = time.time()

    while embedded_count < target:
        batch = fetch_batch(engine, tenant_id, args.batch_size)
        if not batch:
            break

        seg_ids = [b[0] for b in batch]
        texts = [b[1] for b in batch]

        # Embed
        if use_gpu:
            embeddings = try_gpu_embed(texts)
            if embeddings is None:
                # Fallback to Ollama
                logger.warning("GPU embedding returned None, falling back to Ollama")
                embeddings = embed_batch_ollama(texts)
                use_gpu = False
        else:
            embeddings = embed_batch_ollama(texts)

        # Store successful embeddings
        pairs = []
        for seg_id, embedding in zip(seg_ids, embeddings):
            if embedding is not None:
                pairs.append((seg_id, embedding))
            else:
                failed_count += 1

        if pairs:
            store_embeddings(engine, pairs)
            embedded_count += len(pairs)

        elapsed = time.time() - start_time
        rate = embedded_count / elapsed if elapsed > 0 else 0
        remaining_count = target - embedded_count
        eta = remaining_count / rate if rate > 0 else 0

        logger.info(
            "Progress: %d/%d embedded (%d failed) — %.1f seg/s, ETA %.0fs",
            embedded_count, target, failed_count, rate, eta,
        )

    elapsed = time.time() - start_time
    engine.dispose()

    logger.info("=" * 60)
    logger.info("EMBEDDING COMPLETE")
    logger.info("  Tenant:    %s", tenant_id)
    logger.info("  Embedded:  %d", embedded_count)
    logger.info("  Failed:    %d", failed_count)
    logger.info("  Time:      %.1fs (%.1f seg/s)", elapsed, embedded_count / elapsed if elapsed > 0 else 0)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
