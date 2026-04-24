#!/usr/bin/env python3
"""Generate specialized manifold embeddings for promoted objects.

This script processes objects flagged for promotion and generates:
- Claim embeddings (SPO-aware vectors)
- Procedure embeddings (step-sequence vectors)
- Topic embeddings (if missing)

Usage:
    python embed_promoted.py [--batch-size 50] [--manifold claim]
"""
import argparse
import asyncio
import logging
import sys
import os
from datetime import datetime
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import asyncpg
    import httpx
    from manifold.config import get_config
except ImportError:
    asyncpg = None
    httpx = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("embed_promoted")


MANIFOLD_TYPES = ["topic", "claim", "procedure"]


async def get_db_connection():
    """Get database connection."""
    from manifold.config import get_config
    config = get_config()
    db_url = os.environ.get("DATABASE_URL", config.database_url)
    return await asyncpg.connect(db_url)


async def get_embedding(
    client: "httpx.AsyncClient",
    text: str,
    model: str = "nomic-embed-text",
) -> Optional[List[float]]:
    """Get embedding from Ollama."""
    config = get_config()

    try:
        response = await client.post(
            f"{config.ollama_url}/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("embedding")
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        return None


async def get_promoted_objects(
    conn,
    manifold_type: str,
    offset: int,
    limit: int,
) -> List[dict]:
    """Get promoted objects that need embeddings for this manifold."""

    # Map manifold type to object types
    object_types = {
        "topic": ["segments", "claims", "entities", "summaries"],
        "claim": ["claims"],
        "procedure": ["segments"],  # Procedures extracted from segments
    }

    types = object_types.get(manifold_type, ["segments"])
    type_list = ",".join(f"'{t}'" for t in types)

    rows = await conn.fetch(
        f"""
        SELECT
            ps.object_id,
            ps.object_type,
            ps.score
        FROM promotion_scores ps
        LEFT JOIN manifold_embeddings me
            ON ps.object_id = me.object_id
            AND me.manifold_type = $1
        WHERE ps.should_promote = true
        AND ps.object_type IN ({type_list})
        AND me.object_id IS NULL  -- No embedding yet
        ORDER BY ps.score DESC
        OFFSET $2 LIMIT $3
        """,
        manifold_type,
        offset,
        limit,
    )

    return [dict(r) for r in rows]


async def get_object_text(conn, object_id: str, object_type: str) -> Optional[str]:
    """Get text content for an object."""
    table_map = {
        "segments": ("segments", "text"),
        "claims": ("claims", "text"),
        "entities": ("entities", "name"),
        "summaries": ("summaries", "summary_text"),
    }

    if object_type not in table_map:
        return None

    table, column = table_map[object_type]
    result = await conn.fetchval(
        f"SELECT {column} FROM {table} WHERE id = $1",
        object_id,
    )
    return result


async def get_canonical_claim_text(conn, claim_id: str) -> Optional[str]:
    """Get canonical SPO text for a claim."""
    row = await conn.fetchrow(
        """
        SELECT subject, predicate, object, canonical_text
        FROM canonical_claims
        WHERE claim_id = $1
        """,
        claim_id,
    )

    if row:
        # Use canonical text if available, else construct from SPO
        if row["canonical_text"]:
            return row["canonical_text"]
        return f"{row['subject']} {row['predicate']} {row['object']}"

    return None


async def get_procedure_text(conn, segment_id: str) -> Optional[str]:
    """Get ordered procedure steps for embedding."""
    rows = await conn.fetch(
        """
        SELECT step_number, step_text
        FROM canonical_procedures
        WHERE segment_id = $1
        ORDER BY step_number
        """,
        segment_id,
    )

    if rows:
        steps = [f"{r['step_number']}. {r['step_text']}" for r in rows]
        return "\n".join(steps)

    return None


async def store_embedding(
    conn,
    object_id: str,
    object_type: str,
    manifold_type: str,
    embedding: List[float],
    text_used: str,
):
    """Store manifold embedding in database."""
    await conn.execute(
        """
        INSERT INTO manifold_embeddings (
            object_id, object_type, manifold_type,
            embedding, text_used, created_at
        ) VALUES ($1, $2, $3, $4, $5, $6)
        ON CONFLICT (object_id, manifold_type) DO UPDATE SET
            embedding = EXCLUDED.embedding,
            text_used = EXCLUDED.text_used,
            updated_at = NOW()
        """,
        object_id,
        object_type,
        manifold_type,
        embedding,
        text_used[:1000],  # Truncate for storage
        datetime.utcnow(),
    )


async def process_batch(
    conn,
    client: "httpx.AsyncClient",
    manifold_type: str,
    objects: List[dict],
) -> tuple:
    """Process batch of objects. Returns (success, errors)."""
    success = 0
    errors = 0
    config = get_config()

    for obj in objects:
        try:
            object_id = obj["object_id"]
            object_type = obj["object_type"]

            # Get appropriate text for this manifold type
            if manifold_type == "claim" and object_type == "claims":
                text = await get_canonical_claim_text(conn, object_id)
                if not text:
                    # Fall back to raw claim text
                    text = await get_object_text(conn, object_id, object_type)
            elif manifold_type == "procedure":
                text = await get_procedure_text(conn, object_id)
                if not text:
                    # No procedure structure found
                    continue
            else:
                text = await get_object_text(conn, object_id, object_type)

            if not text:
                logger.warning(f"No text for {object_type}/{object_id}")
                errors += 1
                continue

            # Generate embedding
            embedding = await get_embedding(client, text, config.embedding_model)

            if embedding is None:
                errors += 1
                continue

            # Store embedding
            await store_embedding(
                conn, object_id, object_type, manifold_type, embedding, text
            )
            success += 1

        except Exception as e:
            logger.error(f"Error processing {obj}: {e}")
            errors += 1

    return success, errors


async def main(
    batch_size: int = 50,
    manifold_type: str = None,
):
    """Main embedding loop."""
    if asyncpg is None or httpx is None:
        logger.error("Required packages not installed. Run: pip install asyncpg httpx")
        return

    manifolds_to_process = [manifold_type] if manifold_type else MANIFOLD_TYPES

    conn = await get_db_connection()

    async with httpx.AsyncClient() as client:
        try:
            for manifold in manifolds_to_process:
                logger.info(f"Generating {manifold} embeddings for promoted objects")

                total_success = 0
                total_errors = 0
                offset = 0

                while True:
                    objects = await get_promoted_objects(conn, manifold, offset, batch_size)

                    if not objects:
                        break

                    logger.info(
                        f"Processing {manifold} batch at offset {offset} ({len(objects)} objects)"
                    )

                    success, errors = await process_batch(conn, client, manifold, objects)

                    total_success += success
                    total_errors += errors
                    offset += batch_size

                logger.info(
                    f"{manifold} embeddings complete: {total_success} generated, {total_errors} errors"
                )

        finally:
            await conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate manifold embeddings for promoted objects")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of objects per batch",
    )
    parser.add_argument(
        "--manifold",
        choices=MANIFOLD_TYPES,
        help="Process only this manifold type",
    )
    args = parser.parse_args()

    asyncio.run(main(batch_size=args.batch_size, manifold_type=args.manifold))
