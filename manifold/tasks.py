"""Celery tasks for manifold background processing.

These tasks handle:
- Batch embedding generation
- Canonical form extraction
- Promotion score computation
- Shadow mode analysis
"""
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger("gami.manifold.tasks")

# Celery app import - would be from workers.celery_app in production
# For isolated module, we define task stubs
try:
    from celery import shared_task
except ImportError:
    # Stub decorator for isolated module
    def shared_task(func=None, **kwargs):
        def decorator(f):
            f.delay = lambda *a, **kw: None
            f.apply_async = lambda *a, **kw: None
            return f
        if func:
            return decorator(func)
        return decorator


@shared_task(bind=True, max_retries=3)
def embed_objects_batch(
    self,
    object_ids: List[str],
    object_type: str,
    manifold_type: str,
    tenant_id: str = "shared",
    db_url: str = None,
    ollama_url: str = None,
) -> Dict[str, Any]:
    """Generate embeddings for a batch of objects.

    Args:
        object_ids: List of object IDs to embed.
        object_type: Type of objects (segment, claim, entity).
        manifold_type: Which manifold embedding to generate.
        tenant_id: Tenant context.
        db_url: Database connection URL.
        ollama_url: Ollama API URL.

    Returns:
        Dict with success/failure counts.
    """
    import asyncio
    from .config import get_config

    config = get_config()
    db_url = db_url or f"postgresql://gami:gami@localhost:5433/gami"
    ollama_url = ollama_url or config.ollama_url

    logger.info(
        f"Embedding batch: {len(object_ids)} {object_type}s "
        f"for {manifold_type} manifold"
    )

    async def _run():
        import asyncpg
        from .embedding import EmbeddingClient
        from .repository import ManifoldRepository

        # Connect to database
        pool = await asyncpg.create_pool(db_url, min_size=1, max_size=5)
        repo = ManifoldRepository(pool)
        embed_client = EmbeddingClient(base_url=ollama_url, model=config.embedding_model)

        success = 0
        errors = 0

        try:
            # Table mapping for object types
            table_map = {
                "segment": ("segments", "text"),
                "claim": ("claims", "text"),
                "entity": ("entities", "name"),
                "summary": ("summaries", "summary_text"),
            }

            if object_type not in table_map:
                raise ValueError(f"Unknown object type: {object_type}")

            table, text_column = table_map[object_type]

            conn = await pool.acquire()
            try:
                for object_id in object_ids:
                    try:
                        # Fetch object text
                        row = await conn.fetchrow(
                            f"SELECT {text_column} as text FROM {table} WHERE id = $1",
                            object_id,
                        )

                        if not row or not row["text"]:
                            logger.warning(f"No text found for {object_type}/{object_id}")
                            errors += 1
                            continue

                        text = row["text"]

                        # For claim manifold, use canonical form if available
                        if manifold_type == "claim":
                            canonical = await repo.get_canonical_claim(object_id)
                            if canonical:
                                text = canonical.canonical_text

                        # Generate embedding
                        result = await embed_client.embed_one(text)

                        # Store embedding
                        await repo.upsert_embedding(
                            object_id=object_id,
                            object_type=object_type,
                            manifold_type=manifold_type,
                            embedding=result.embedding,
                            text_used=text[:1000],
                        )

                        success += 1

                    except Exception as e:
                        logger.error(f"Error embedding {object_id}: {e}")
                        errors += 1

            finally:
                await pool.release(conn)

        finally:
            await embed_client.close()
            await pool.close()

        return success, errors

    success, errors = asyncio.get_event_loop().run_until_complete(_run())

    return {
        "object_type": object_type,
        "manifold_type": manifold_type,
        "total": len(object_ids),
        "success": success,
        "errors": errors,
    }


@shared_task(bind=True, max_retries=3)
def extract_canonical_claims(
    self,
    segment_ids: List[str],
    tenant_id: str = "shared",
    db_url: str = None,
) -> Dict[str, Any]:
    """Extract canonical SPO claims from segments.

    Args:
        segment_ids: Segments to extract claims from.
        tenant_id: Tenant context.
        db_url: Database connection URL.

    Returns:
        Dict with extraction counts.
    """
    import asyncio
    import uuid
    from .config import get_config
    from .canonical.claim_normalizer import ClaimNormalizer

    config = get_config()
    db_url = db_url or f"postgresql://gami:gami@localhost:5433/gami"

    logger.info(f"Extracting claims from {len(segment_ids)} segments")

    async def _run():
        import asyncpg
        from .repository import ManifoldRepository

        pool = await asyncpg.create_pool(db_url, min_size=1, max_size=5)
        repo = ManifoldRepository(pool)
        normalizer = ClaimNormalizer()

        extracted = 0
        errors = 0
        claims_created = []

        try:
            conn = await pool.acquire()
            try:
                for segment_id in segment_ids:
                    try:
                        # Fetch segment text
                        row = await conn.fetchrow(
                            "SELECT text, source_id FROM segments WHERE id = $1",
                            segment_id,
                        )

                        if not row or not row["text"]:
                            logger.warning(f"No text found for segment {segment_id}")
                            errors += 1
                            continue

                        text = row["text"]
                        source_id = row["source_id"]

                        # Extract claims using normalizer
                        claims = normalizer.extract_claims(text)

                        for claim in claims:
                            # Normalize to SPO canonical form
                            canonical = normalizer.normalize(claim)

                            if not canonical or not canonical.canonical_text:
                                continue

                            # Generate claim ID
                            claim_id = str(uuid.uuid4())

                            # Store canonical claim
                            await conn.execute(
                                """
                                INSERT INTO canonical_claims (
                                    id, segment_id, source_id, tenant_id,
                                    subject, predicate, object,
                                    canonical_text, original_text,
                                    confidence, modality, temporal_scope,
                                    created_at
                                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, NOW())
                                ON CONFLICT (id) DO NOTHING
                                """,
                                claim_id,
                                segment_id,
                                source_id,
                                tenant_id,
                                canonical.subject,
                                canonical.predicate,
                                canonical.object,
                                canonical.canonical_text,
                                claim.text if hasattr(claim, 'text') else text[:500],
                                canonical.confidence,
                                canonical.modality or "assertion",
                                canonical.temporal_scope,
                            )

                            claims_created.append(claim_id)
                            extracted += 1

                    except Exception as e:
                        logger.error(f"Error extracting claims from {segment_id}: {e}")
                        errors += 1

            finally:
                await pool.release(conn)

        finally:
            await pool.close()

        return extracted, errors, claims_created

    extracted, errors, claims_created = asyncio.get_event_loop().run_until_complete(_run())

    return {
        "segments_processed": len(segment_ids),
        "claims_extracted": extracted,
        "claim_ids": claims_created[:100],  # Limit response size
        "errors": errors,
    }


@shared_task(bind=True, max_retries=3)
def extract_canonical_procedures(
    self,
    segment_ids: List[str],
    tenant_id: str = "shared",
    db_url: str = None,
) -> Dict[str, Any]:
    """Extract canonical procedures from segments.

    Args:
        segment_ids: Segments to extract procedures from.
        tenant_id: Tenant context.
        db_url: Database connection URL.

    Returns:
        Dict with extraction counts.
    """
    import asyncio
    import uuid
    from .config import get_config
    from .canonical.procedure_normalizer import ProcedureNormalizer

    config = get_config()
    db_url = db_url or f"postgresql://gami:gami@localhost:5433/gami"

    logger.info(f"Extracting procedures from {len(segment_ids)} segments")

    async def _run():
        import asyncpg
        from .repository import ManifoldRepository

        pool = await asyncpg.create_pool(db_url, min_size=1, max_size=5)
        repo = ManifoldRepository(pool)
        normalizer = ProcedureNormalizer()

        extracted = 0
        errors = 0
        procedures_created = []

        try:
            conn = await pool.acquire()
            try:
                for segment_id in segment_ids:
                    try:
                        # Fetch segment text
                        row = await conn.fetchrow(
                            "SELECT text, source_id FROM segments WHERE id = $1",
                            segment_id,
                        )

                        if not row or not row["text"]:
                            logger.warning(f"No text found for segment {segment_id}")
                            errors += 1
                            continue

                        text = row["text"]
                        source_id = row["source_id"]

                        # Extract procedures using normalizer
                        procedures = normalizer.extract_procedures(text)

                        for procedure in procedures:
                            # Normalize to canonical step form
                            canonical = normalizer.normalize(procedure)

                            if not canonical or not canonical.steps:
                                continue

                            # Generate procedure ID
                            procedure_id = str(uuid.uuid4())

                            # Serialize steps to JSON
                            import json
                            steps_json = json.dumps([
                                {
                                    "order": step.order,
                                    "action": step.action,
                                    "target": step.target,
                                    "conditions": step.conditions,
                                    "parameters": step.parameters,
                                }
                                for step in canonical.steps
                            ])

                            # Store canonical procedure
                            await conn.execute(
                                """
                                INSERT INTO canonical_procedures (
                                    id, segment_id, source_id, tenant_id,
                                    title, description, steps,
                                    canonical_text, original_text,
                                    confidence, domain, prerequisites,
                                    created_at
                                ) VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9, $10, $11, $12, NOW())
                                ON CONFLICT (id) DO NOTHING
                                """,
                                procedure_id,
                                segment_id,
                                source_id,
                                tenant_id,
                                canonical.title or "Untitled Procedure",
                                canonical.description,
                                steps_json,
                                canonical.canonical_text,
                                procedure.text if hasattr(procedure, 'text') else text[:500],
                                canonical.confidence,
                                canonical.domain,
                                json.dumps(canonical.prerequisites) if canonical.prerequisites else None,
                            )

                            procedures_created.append(procedure_id)
                            extracted += 1

                    except Exception as e:
                        logger.error(f"Error extracting procedures from {segment_id}: {e}")
                        errors += 1

            finally:
                await pool.release(conn)

        finally:
            await pool.close()

        return extracted, errors, procedures_created

    extracted, errors, procedures_created = asyncio.get_event_loop().run_until_complete(_run())

    return {
        "segments_processed": len(segment_ids),
        "procedures_extracted": extracted,
        "procedure_ids": procedures_created[:100],  # Limit response size
        "errors": errors,
    }


@shared_task(bind=True, max_retries=3)
def compute_promotion_scores_batch(
    self,
    object_ids: List[str],
    object_type: str,
    db_url: str = None,
    promotion_threshold: float = 0.65,
) -> Dict[str, Any]:
    """Compute promotion scores for a batch of objects.

    Args:
        object_ids: Objects to score.
        object_type: Type of objects.
        db_url: Database connection URL.
        promotion_threshold: Threshold above which objects are promoted.

    Returns:
        Dict with scoring results.
    """
    import asyncio
    from .config import get_config
    from .scoring.promotion import compute_promotion_score, PromotionFactors

    config = get_config()
    db_url = db_url or f"postgresql://gami:gami@localhost:5433/gami"
    promotion_threshold = promotion_threshold or config.promotion_threshold

    logger.info(f"Computing promotion scores for {len(object_ids)} {object_type}s")

    async def _run():
        import asyncpg

        pool = await asyncpg.create_pool(db_url, min_size=1, max_size=5)

        scored = 0
        promoted = 0
        demoted = 0
        errors = 0
        score_updates = []

        try:
            conn = await pool.acquire()
            try:
                for object_id in object_ids:
                    try:
                        # Fetch metrics for promotion factors
                        metrics_row = await conn.fetchrow(
                            """
                            SELECT
                                COALESCE(m.retrieval_count, 0) as retrieval_count,
                                COALESCE(m.citation_count, 0) as citation_count,
                                COALESCE(m.confirmation_count, 0) as confirmation_count,
                                COALESCE(m.contradiction_count, 0) as contradiction_count,
                                COALESCE(m.last_retrieved_at, created_at) as last_retrieved_at,
                                created_at,
                                COALESCE(e.embedding IS NOT NULL, false) as has_embedding
                            FROM (
                                SELECT id, created_at FROM segments WHERE id = $1
                                UNION ALL
                                SELECT id, created_at FROM claims WHERE id = $1
                                UNION ALL
                                SELECT id, created_at FROM summaries WHERE id = $1
                            ) obj
                            LEFT JOIN object_metrics m ON m.object_id = $1
                            LEFT JOIN manifold_embeddings e ON e.object_id = $1
                            LIMIT 1
                            """,
                            object_id,
                        )

                        if not metrics_row:
                            logger.warning(f"No object found for {object_id}")
                            errors += 1
                            continue

                        # Calculate age in days
                        from datetime import datetime, timezone
                        now = datetime.now(timezone.utc)
                        created_at = metrics_row["created_at"]
                        if created_at.tzinfo is None:
                            created_at = created_at.replace(tzinfo=timezone.utc)
                        age_days = (now - created_at).days

                        # Build promotion factors
                        factors = PromotionFactors(
                            importance=min(1.0, metrics_row["citation_count"] / 10.0),
                            retrieval_frequency=min(1.0, metrics_row["retrieval_count"] / 50.0),
                            diversity=0.5,  # Would compute from cluster analysis
                            confidence=max(0.0, 1.0 - (metrics_row["contradiction_count"] * 0.2)),
                            novelty=max(0.0, 1.0 - (age_days / 365.0)),
                            centrality=0.5,  # Would compute from graph
                            relevance=min(1.0, metrics_row["confirmation_count"] / 5.0),
                        )

                        # Compute score
                        score = compute_promotion_score(factors)

                        # Determine tier
                        if score >= promotion_threshold:
                            tier = "promoted"
                            promoted += 1
                        elif score < config.demotion_threshold:
                            tier = "demoted"
                            demoted += 1
                        else:
                            tier = "standard"

                        # Update or insert score
                        await conn.execute(
                            """
                            INSERT INTO promotion_scores (
                                object_id, object_type, score, tier,
                                factors, computed_at
                            ) VALUES ($1, $2, $3, $4, $5::jsonb, NOW())
                            ON CONFLICT (object_id) DO UPDATE SET
                                score = EXCLUDED.score,
                                tier = EXCLUDED.tier,
                                factors = EXCLUDED.factors,
                                computed_at = NOW()
                            """,
                            object_id,
                            object_type,
                            score,
                            tier,
                            factors.to_json() if hasattr(factors, 'to_json') else str(factors.__dict__),
                        )

                        score_updates.append({
                            "object_id": object_id,
                            "score": score,
                            "tier": tier,
                        })
                        scored += 1

                    except Exception as e:
                        logger.error(f"Error scoring {object_id}: {e}")
                        errors += 1

            finally:
                await pool.release(conn)

        finally:
            await pool.close()

        return scored, promoted, demoted, errors, score_updates

    scored, promoted, demoted, errors, score_updates = asyncio.get_event_loop().run_until_complete(_run())

    return {
        "object_type": object_type,
        "scored": scored,
        "promoted": promoted,
        "demoted": demoted,
        "errors": errors,
        "threshold": promotion_threshold,
        "updates": score_updates[:50],  # Limit response size
    }


@shared_task(bind=True)
def extract_temporal_features_batch(
    self,
    object_ids: List[str],
    object_type: str,
    db_url: str = None,
) -> Dict[str, Any]:
    """Extract temporal features for a batch of objects.

    Args:
        object_ids: Objects to extract features for.
        object_type: Type of objects.
        db_url: Database connection URL.

    Returns:
        Dict with extraction results.
    """
    import asyncio
    import json
    from .config import get_config
    from .temporal.feature_extractor import TemporalExtractor

    config = get_config()
    db_url = db_url or f"postgresql://gami:gami@localhost:5433/gami"

    logger.info(f"Extracting temporal features for {len(object_ids)} {object_type}s")

    async def _run():
        import asyncpg

        pool = await asyncpg.create_pool(db_url, min_size=1, max_size=5)
        extractor = TemporalExtractor()

        extracted = 0
        errors = 0
        features_stored = []

        try:
            conn = await pool.acquire()
            try:
                # Table mapping for object types
                table_map = {
                    "segment": ("segments", "text", "created_at"),
                    "claim": ("claims", "text", "created_at"),
                    "event": ("events", "description", "event_time"),
                    "summary": ("summaries", "summary_text", "created_at"),
                }

                if object_type not in table_map:
                    logger.error(f"Unknown object type for temporal extraction: {object_type}")
                    return 0, len(object_ids), []

                table, text_col, time_col = table_map[object_type]

                for object_id in object_ids:
                    try:
                        # Fetch object with text and timestamp
                        row = await conn.fetchrow(
                            f"""
                            SELECT {text_col} as text, {time_col} as timestamp,
                                   source_id, tenant_id
                            FROM {table} WHERE id = $1
                            """,
                            object_id,
                        )

                        if not row:
                            logger.warning(f"No object found for {object_id}")
                            errors += 1
                            continue

                        text = row["text"] or ""
                        timestamp = row["timestamp"]

                        # Extract temporal features (12-dimensional)
                        features = extractor.extract(
                            text=text,
                            reference_time=timestamp,
                        )

                        if not features:
                            errors += 1
                            continue

                        # Convert features to storable format
                        feature_vector = features.to_vector()
                        feature_dict = features.to_dict()

                        # Store temporal features
                        await conn.execute(
                            """
                            INSERT INTO temporal_features (
                                object_id, object_type, tenant_id,
                                feature_vector, features_json,
                                reference_time, extracted_times,
                                time_range_start, time_range_end,
                                temporal_type, confidence,
                                created_at
                            ) VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7::jsonb, $8, $9, $10, $11, NOW())
                            ON CONFLICT (object_id) DO UPDATE SET
                                feature_vector = EXCLUDED.feature_vector,
                                features_json = EXCLUDED.features_json,
                                reference_time = EXCLUDED.reference_time,
                                extracted_times = EXCLUDED.extracted_times,
                                time_range_start = EXCLUDED.time_range_start,
                                time_range_end = EXCLUDED.time_range_end,
                                temporal_type = EXCLUDED.temporal_type,
                                confidence = EXCLUDED.confidence,
                                created_at = NOW()
                            """,
                            object_id,
                            object_type,
                            row["tenant_id"],
                            feature_vector,
                            json.dumps(feature_dict),
                            timestamp,
                            json.dumps([t.isoformat() if hasattr(t, 'isoformat') else str(t) for t in features.extracted_times]) if features.extracted_times else None,
                            features.time_range_start,
                            features.time_range_end,
                            features.temporal_type,
                            features.confidence,
                        )

                        features_stored.append({
                            "object_id": object_id,
                            "temporal_type": features.temporal_type,
                            "confidence": features.confidence,
                        })
                        extracted += 1

                    except Exception as e:
                        logger.error(f"Error extracting temporal features for {object_id}: {e}")
                        errors += 1

            finally:
                await pool.release(conn)

        finally:
            await pool.close()

        return extracted, errors, features_stored

    extracted, errors, features_stored = asyncio.get_event_loop().run_until_complete(_run())

    return {
        "object_type": object_type,
        "extracted": extracted,
        "errors": errors,
        "features": features_stored[:50],  # Limit response size
    }


@shared_task(bind=True)
def analyze_shadow_comparisons(
    self,
    since_hours: int = 24,
    db_url: str = None,
) -> Dict[str, Any]:
    """Analyze recent shadow mode comparisons.

    Args:
        since_hours: Look back this many hours.
        db_url: Database connection URL.

    Returns:
        Dict with analysis results.
    """
    import asyncio
    from .config import get_config
    from .retrieval.shadow_mode import ShadowAnalyzer, ShadowStats

    config = get_config()
    db_url = db_url or f"postgresql://gami:gami@localhost:5433/gami"

    logger.info(f"Analyzing shadow comparisons from last {since_hours} hours")

    async def _run():
        import asyncpg
        from datetime import datetime, timezone, timedelta

        pool = await asyncpg.create_pool(db_url, min_size=1, max_size=3)

        try:
            conn = await pool.acquire()
            try:
                # Calculate time boundary
                since_time = datetime.now(timezone.utc) - timedelta(hours=since_hours)

                # Fetch aggregated comparison statistics
                stats_row = await conn.fetchrow(
                    """
                    SELECT
                        COUNT(*) as total_comparisons,
                        AVG(CASE WHEN legacy_better THEN 0 ELSE 1 END) as manifold_win_rate,
                        AVG(overlap_ratio) as avg_overlap_ratio,
                        AVG(manifold_latency_ms) as avg_manifold_latency,
                        AVG(legacy_latency_ms) as avg_legacy_latency,
                        SUM(CASE WHEN manifold_better THEN 1 ELSE 0 END) as manifold_wins,
                        SUM(CASE WHEN legacy_better THEN 1 ELSE 0 END) as legacy_wins,
                        SUM(CASE WHEN NOT manifold_better AND NOT legacy_better THEN 1 ELSE 0 END) as ties
                    FROM shadow_comparisons
                    WHERE created_at >= $1
                    """,
                    since_time,
                )

                # Fetch per-query-mode breakdown
                mode_breakdown = await conn.fetch(
                    """
                    SELECT
                        query_mode,
                        COUNT(*) as count,
                        AVG(overlap_ratio) as avg_overlap,
                        AVG(CASE WHEN manifold_better THEN 1 ELSE 0 END) as manifold_win_rate
                    FROM shadow_comparisons
                    WHERE created_at >= $1
                    GROUP BY query_mode
                    ORDER BY count DESC
                    """,
                    since_time,
                )

                # Fetch recent problematic comparisons (low overlap, manifold lost)
                problem_queries = await conn.fetch(
                    """
                    SELECT query_text, query_mode, overlap_ratio,
                           manifold_latency_ms, legacy_latency_ms
                    FROM shadow_comparisons
                    WHERE created_at >= $1
                      AND legacy_better = true
                      AND overlap_ratio < 0.5
                    ORDER BY overlap_ratio ASC
                    LIMIT 10
                    """,
                    since_time,
                )

                total = stats_row["total_comparisons"] or 0
                manifold_wins = stats_row["manifold_wins"] or 0
                legacy_wins = stats_row["legacy_wins"] or 0

                # Build stats object
                stats = ShadowStats(
                    total_comparisons=total,
                    manifold_wins=manifold_wins,
                    legacy_wins=legacy_wins,
                    ties=stats_row["ties"] or 0,
                    match_rate=float(stats_row["manifold_win_rate"] or 0),
                    improvement_rate=(manifold_wins - legacy_wins) / max(1, total),
                    avg_overlap_ratio=float(stats_row["avg_overlap_ratio"] or 0),
                    avg_manifold_latency_ms=float(stats_row["avg_manifold_latency"] or 0),
                    avg_legacy_latency_ms=float(stats_row["avg_legacy_latency"] or 0),
                )

                mode_stats = [
                    {
                        "mode": row["query_mode"],
                        "count": row["count"],
                        "avg_overlap": float(row["avg_overlap"] or 0),
                        "manifold_win_rate": float(row["manifold_win_rate"] or 0),
                    }
                    for row in mode_breakdown
                ]

                problems = [
                    {
                        "query": row["query_text"][:100],
                        "mode": row["query_mode"],
                        "overlap": float(row["overlap_ratio"] or 0),
                    }
                    for row in problem_queries
                ]

                return stats, mode_stats, problems

            finally:
                await pool.release(conn)

        finally:
            await pool.close()

    stats, mode_stats, problems = asyncio.get_event_loop().run_until_complete(_run())

    return {
        "period_hours": since_hours,
        "total_comparisons": stats.total_comparisons,
        "manifold_wins": stats.manifold_wins,
        "legacy_wins": stats.legacy_wins,
        "ties": stats.ties,
        "match_rate": stats.match_rate,
        "improvement_rate": stats.improvement_rate,
        "avg_overlap_ratio": stats.avg_overlap_ratio,
        "avg_manifold_latency_ms": stats.avg_manifold_latency_ms,
        "avg_legacy_latency_ms": stats.avg_legacy_latency_ms,
        "latency_improvement": (
            (stats.avg_legacy_latency_ms - stats.avg_manifold_latency_ms) /
            max(1, stats.avg_legacy_latency_ms)
        ) if stats.avg_legacy_latency_ms else 0,
        "mode_breakdown": mode_stats,
        "problem_queries": problems,
        "recommendation": (
            "ready_for_promotion" if stats.improvement_rate > 0.1 and stats.avg_overlap_ratio > 0.7
            else "needs_tuning" if stats.total_comparisons > 100
            else "insufficient_data"
        ),
    }


@shared_task(bind=True)
def rebuild_graph_fingerprints(
    self,
    entity_type: Optional[str] = None,
    batch_size: int = 100,
    db_url: str = None,
) -> Dict[str, Any]:
    """Rebuild graph fingerprints for entities.

    Computes structural fingerprints based on entity neighborhood
    in the Apache AGE knowledge graph.

    Args:
        entity_type: Specific type to rebuild, or all if None.
        batch_size: Process in batches of this size.
        db_url: Database connection URL.

    Returns:
        Dict with rebuild results.
    """
    import asyncio
    import json
    import hashlib
    from .config import get_config

    config = get_config()
    db_url = db_url or f"postgresql://gami:gami@localhost:5433/gami"

    logger.info(f"Rebuilding graph fingerprints (type={entity_type})")

    async def _run():
        import asyncpg

        pool = await asyncpg.create_pool(db_url, min_size=1, max_size=5)

        computed = 0
        errors = 0
        offset = 0

        try:
            conn = await pool.acquire()
            try:
                # Ensure AGE extension is loaded
                await conn.execute("LOAD 'age'")
                await conn.execute("SET search_path = ag_catalog, '$user', public")

                while True:
                    # Fetch batch of entities
                    type_filter = "AND entity_type = $3" if entity_type else ""
                    query = f"""
                        SELECT id, entity_type, name, aliases
                        FROM entities
                        WHERE 1=1 {type_filter}
                        ORDER BY id
                        LIMIT $1 OFFSET $2
                    """
                    params = [batch_size, offset]
                    if entity_type:
                        params.append(entity_type)

                    entities = await conn.fetch(query, *params)

                    if not entities:
                        break

                    for entity in entities:
                        try:
                            entity_id = entity["id"]

                            # Query AGE graph for entity neighborhood
                            # Using Cypher to get 2-hop neighborhood structure
                            cypher_result = await conn.fetch(
                                """
                                SELECT * FROM cypher('gami_graph', $$
                                    MATCH (e:Entity {id: $entity_id})
                                    OPTIONAL MATCH (e)-[r1]->(n1)
                                    OPTIONAL MATCH (e)<-[r2]-(n2)
                                    OPTIONAL MATCH (e)-[r3]->(m1)-[r4]->(n3)
                                    RETURN
                                        count(DISTINCT n1) as out_degree_1,
                                        count(DISTINCT n2) as in_degree_1,
                                        count(DISTINCT n3) as out_degree_2,
                                        collect(DISTINCT type(r1)) as out_edge_types,
                                        collect(DISTINCT type(r2)) as in_edge_types,
                                        collect(DISTINCT labels(n1)) as neighbor_types_1
                                $$, $1) AS (
                                    out_degree_1 agtype,
                                    in_degree_1 agtype,
                                    out_degree_2 agtype,
                                    out_edge_types agtype,
                                    in_edge_types agtype,
                                    neighbor_types_1 agtype
                                )
                                """,
                                json.dumps({"entity_id": entity_id}),
                            )

                            # Default fingerprint if no graph data
                            out_degree_1 = 0
                            in_degree_1 = 0
                            out_degree_2 = 0
                            out_edge_types = []
                            in_edge_types = []
                            neighbor_types = []

                            if cypher_result:
                                row = cypher_result[0]
                                out_degree_1 = int(row["out_degree_1"]) if row["out_degree_1"] else 0
                                in_degree_1 = int(row["in_degree_1"]) if row["in_degree_1"] else 0
                                out_degree_2 = int(row["out_degree_2"]) if row["out_degree_2"] else 0
                                out_edge_types = row["out_edge_types"] or []
                                in_edge_types = row["in_edge_types"] or []
                                neighbor_types = row["neighbor_types_1"] or []

                            # Compute structural fingerprint
                            fingerprint_data = {
                                "out_degree_1": out_degree_1,
                                "in_degree_1": in_degree_1,
                                "out_degree_2": out_degree_2,
                                "total_degree": out_degree_1 + in_degree_1,
                                "out_edge_types": sorted(set(out_edge_types)),
                                "in_edge_types": sorted(set(in_edge_types)),
                                "neighbor_type_diversity": len(set(neighbor_types)),
                                "hub_score": min(1.0, (out_degree_1 + in_degree_1) / 50.0),
                            }

                            # Create hash fingerprint for similarity comparison
                            fingerprint_hash = hashlib.sha256(
                                json.dumps(fingerprint_data, sort_keys=True).encode()
                            ).hexdigest()[:32]

                            # Store fingerprint
                            await conn.execute(
                                """
                                INSERT INTO graph_fingerprints (
                                    entity_id, entity_type, fingerprint_hash,
                                    fingerprint_data, out_degree, in_degree,
                                    hub_score, computed_at
                                ) VALUES ($1, $2, $3, $4::jsonb, $5, $6, $7, NOW())
                                ON CONFLICT (entity_id) DO UPDATE SET
                                    fingerprint_hash = EXCLUDED.fingerprint_hash,
                                    fingerprint_data = EXCLUDED.fingerprint_data,
                                    out_degree = EXCLUDED.out_degree,
                                    in_degree = EXCLUDED.in_degree,
                                    hub_score = EXCLUDED.hub_score,
                                    computed_at = NOW()
                                """,
                                entity_id,
                                entity["entity_type"],
                                fingerprint_hash,
                                json.dumps(fingerprint_data),
                                out_degree_1,
                                in_degree_1,
                                fingerprint_data["hub_score"],
                            )

                            computed += 1

                        except Exception as e:
                            logger.error(f"Error computing fingerprint for entity {entity['id']}: {e}")
                            errors += 1

                    offset += batch_size
                    logger.info(f"Processed {offset} entities, {computed} fingerprints computed")

            finally:
                await pool.release(conn)

        finally:
            await pool.close()

        return computed, errors

    computed, errors = asyncio.get_event_loop().run_until_complete(_run())

    return {
        "entity_type": entity_type or "all",
        "fingerprints_computed": computed,
        "errors": errors,
        "batch_size": batch_size,
    }


@shared_task(bind=True)
def warm_query_cache(
    self,
    tenant_id: str,
    query_count: int = 100,
    db_url: str = None,
    redis_url: str = None,
) -> Dict[str, Any]:
    """Warm the query cache with common queries.

    Fetches the most frequent queries from logs and pre-executes them
    to populate the Redis cache, improving response times.

    Args:
        tenant_id: Tenant to warm cache for.
        query_count: Number of top queries to cache.
        db_url: Database connection URL.
        redis_url: Redis connection URL.

    Returns:
        Dict with warming results.
    """
    import asyncio
    from .config import get_config

    config = get_config()
    db_url = db_url or f"postgresql://gami:gami@localhost:5433/gami"
    redis_url = redis_url or config.redis_url

    logger.info(f"Warming cache for tenant {tenant_id} with top {query_count} queries")

    async def _run():
        import asyncpg
        import redis.asyncio as redis

        pool = await asyncpg.create_pool(db_url, min_size=1, max_size=5)
        redis_client = redis.from_url(redis_url, decode_responses=True)

        warmed = 0
        skipped = 0
        errors = 0
        queries_processed = []

        try:
            conn = await pool.acquire()
            try:
                # Fetch top queries by frequency from query logs
                top_queries = await conn.fetch(
                    """
                    SELECT
                        query_text,
                        query_mode,
                        COUNT(*) as frequency,
                        AVG(latency_ms) as avg_latency,
                        MAX(executed_at) as last_executed
                    FROM query_logs
                    WHERE tenant_id = $1
                      AND executed_at > NOW() - INTERVAL '7 days'
                      AND success = true
                    GROUP BY query_text, query_mode
                    ORDER BY frequency DESC, avg_latency DESC
                    LIMIT $2
                    """,
                    tenant_id,
                    query_count,
                )

                if not top_queries:
                    logger.info(f"No queries found for tenant {tenant_id}")
                    return 0, 0, 0, []

                # Import retrieval orchestrator for query execution
                from .retrieval.orchestrator import RetrievalOrchestrator
                from .repository import ManifoldRepository

                repo = ManifoldRepository(pool)
                orchestrator = RetrievalOrchestrator(
                    config=config,
                    repository=repo,
                )

                for query_row in top_queries:
                    query_text = query_row["query_text"]
                    query_mode = query_row["query_mode"]

                    try:
                        # Check if already cached
                        import hashlib
                        cache_key = f"manifold:recall:{tenant_id}:{hashlib.sha256(query_text.encode()).hexdigest()[:16]}"

                        cached = await redis_client.get(cache_key)
                        if cached:
                            skipped += 1
                            continue

                        # Execute query to warm cache
                        result = await orchestrator.recall(
                            query=query_text,
                            top_k=20,
                            tenant_id=tenant_id,
                            mode_hint=query_mode,
                        )

                        # Cache the result
                        import json
                        cache_data = {
                            "candidates": [
                                {
                                    "object_id": c.object_id,
                                    "object_type": c.object_type,
                                    "text": c.text[:500],
                                    "fused_score": c.fused_score,
                                }
                                for c in (result.candidates if result else [])[:20]
                            ],
                            "query_mode": result.query_mode.value if result else None,
                            "cached_at": datetime.now().isoformat(),
                        }

                        # Cache for 1 hour
                        await redis_client.setex(
                            cache_key,
                            3600,
                            json.dumps(cache_data),
                        )

                        queries_processed.append({
                            "query": query_text[:50],
                            "mode": query_mode,
                            "frequency": query_row["frequency"],
                            "results": len(result.candidates) if result else 0,
                        })

                        warmed += 1

                    except Exception as e:
                        logger.error(f"Error warming cache for query '{query_text[:50]}': {e}")
                        errors += 1

                # Also warm embedding cache for common query terms
                common_terms = await conn.fetch(
                    """
                    SELECT DISTINCT unnest(string_to_array(lower(query_text), ' ')) as term
                    FROM query_logs
                    WHERE tenant_id = $1
                      AND executed_at > NOW() - INTERVAL '7 days'
                    GROUP BY term
                    HAVING COUNT(*) > 5
                    LIMIT 200
                    """,
                    tenant_id,
                )

                if common_terms:
                    from .embedding import EmbeddingClient
                    embed_client = EmbeddingClient(
                        base_url=config.ollama_url,
                        model=config.embedding_model,
                    )

                    try:
                        terms = [row["term"] for row in common_terms if len(row["term"]) > 2]
                        if terms:
                            # Batch embed common terms (this populates the embedding cache)
                            await embed_client.embed_batch(terms[:100])
                            logger.info(f"Warmed embedding cache with {len(terms[:100])} common terms")
                    finally:
                        await embed_client.close()

            finally:
                await pool.release(conn)

        finally:
            await redis_client.close()
            await pool.close()

        return warmed, skipped, errors, queries_processed

    warmed, skipped, errors, queries_processed = asyncio.get_event_loop().run_until_complete(_run())

    return {
        "tenant_id": tenant_id,
        "queries_warmed": warmed,
        "queries_skipped": skipped,
        "errors": errors,
        "total_processed": warmed + skipped + errors,
        "warmed_queries": queries_processed[:20],  # Sample
    }


# Periodic task schedules (for Celery Beat)
CELERY_BEAT_SCHEDULE = {
    "compute-promotion-scores-daily": {
        "task": "manifold.tasks.compute_promotion_scores_batch",
        "schedule": 86400,  # Daily
        "kwargs": {"object_ids": [], "object_type": "segment"},
    },
    "analyze-shadow-comparisons-hourly": {
        "task": "manifold.tasks.analyze_shadow_comparisons",
        "schedule": 3600,  # Hourly
        "kwargs": {"since_hours": 1},
    },
    "rebuild-graph-fingerprints-weekly": {
        "task": "manifold.tasks.rebuild_graph_fingerprints",
        "schedule": 604800,  # Weekly
    },
}
