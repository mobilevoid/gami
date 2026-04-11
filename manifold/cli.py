#!/usr/bin/env python3
"""Command-line interface for manifold system administration.

Usage:
    python -m manifold.cli <command> [options]

Commands:
    migrate     Run migration scripts
    embed       Generate embeddings for objects
    promote     Compute promotion scores
    stats       Show system statistics
    config      Show or validate configuration
    cache       Cache management
    shadow      Shadow mode analysis
"""
import argparse
import asyncio
import json
import sys
import logging
from typing import Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("manifold.cli")


def cmd_migrate(args):
    """Run migration scripts."""
    from .config import get_config

    config = get_config()
    db_url = f"postgresql://gami:gami@localhost:5433/gami"

    async def run_migrations():
        import asyncpg

        pool = await asyncpg.create_pool(db_url, min_size=1, max_size=3)
        migrated = 0

        try:
            conn = await pool.acquire()
            try:
                if args.claims or args.all:
                    logger.info("Running canonical claims migration...")
                    # Fetch segments without canonical claims
                    segments = await conn.fetch(
                        """
                        SELECT s.id FROM segments s
                        LEFT JOIN canonical_claims c ON c.segment_id = s.id
                        WHERE c.id IS NULL
                        LIMIT 1000
                        """
                    )
                    if segments:
                        from .tasks import extract_canonical_claims
                        segment_ids = [row["id"] for row in segments]
                        result = extract_canonical_claims.delay(segment_ids)
                        print(f"Queued {len(segment_ids)} segments for claim extraction")
                        migrated += len(segment_ids)
                    else:
                        print("No segments need claim migration")

                if args.procedures or args.all:
                    logger.info("Running canonical procedures migration...")
                    # Fetch segments without canonical procedures
                    segments = await conn.fetch(
                        """
                        SELECT s.id FROM segments s
                        LEFT JOIN canonical_procedures p ON p.segment_id = s.id
                        WHERE p.id IS NULL
                        AND s.text LIKE '%step%' OR s.text LIKE '%how to%'
                        LIMIT 1000
                        """
                    )
                    if segments:
                        from .tasks import extract_canonical_procedures
                        segment_ids = [row["id"] for row in segments]
                        result = extract_canonical_procedures.delay(segment_ids)
                        print(f"Queued {len(segment_ids)} segments for procedure extraction")
                        migrated += len(segment_ids)
                    else:
                        print("No segments need procedure migration")

            finally:
                await pool.release(conn)
        finally:
            await pool.close()

        return migrated

    try:
        migrated = asyncio.get_event_loop().run_until_complete(run_migrations())
        print(f"\nMigration complete: {migrated} objects queued for processing")
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        print(f"Migration failed: {e}")
        return 1


def cmd_embed(args):
    """Generate embeddings."""
    from .config import get_config

    config = get_config()
    db_url = f"postgresql://gami:gami@localhost:5433/gami"
    batch_size = args.batch_size
    tenant_id = args.tenant

    logger.info(f"Embedding {args.type} objects (batch_size={batch_size})")

    async def run_embedding():
        import asyncpg

        pool = await asyncpg.create_pool(db_url, min_size=1, max_size=3)
        total_queued = 0

        try:
            conn = await pool.acquire()
            try:
                object_types = []
                if args.type == "all":
                    object_types = ["segment", "claim", "entity", "summary"]
                else:
                    # Map plural to singular
                    type_map = {
                        "segments": "segment",
                        "claims": "claim",
                        "entities": "entity",
                    }
                    object_types = [type_map.get(args.type, args.type)]

                for object_type in object_types:
                    # Table mapping
                    table_map = {
                        "segment": "segments",
                        "claim": "claims",
                        "entity": "entities",
                        "summary": "summaries",
                    }
                    table = table_map.get(object_type, f"{object_type}s")

                    # Find objects without embeddings (promoted tier first)
                    objects = await conn.fetch(
                        f"""
                        SELECT o.id FROM {table} o
                        LEFT JOIN manifold_embeddings e
                            ON e.object_id = o.id AND e.manifold_type = 'topic'
                        LEFT JOIN promotion_scores p ON p.object_id = o.id
                        WHERE e.id IS NULL
                        ORDER BY COALESCE(p.score, 0) DESC
                        LIMIT $1
                        """,
                        batch_size * 10,  # Get more to batch
                    )

                    if not objects:
                        print(f"No {object_type}s need embedding")
                        continue

                    # Queue in batches
                    from .tasks import embed_objects_batch
                    object_ids = [row["id"] for row in objects]

                    for i in range(0, len(object_ids), batch_size):
                        batch = object_ids[i:i + batch_size]
                        embed_objects_batch.delay(
                            object_ids=batch,
                            object_type=object_type,
                            manifold_type="topic",
                            tenant_id=tenant_id,
                        )
                        total_queued += len(batch)

                    print(f"Queued {len(object_ids)} {object_type}s for embedding")

            finally:
                await pool.release(conn)
        finally:
            await pool.close()

        return total_queued

    try:
        total = asyncio.get_event_loop().run_until_complete(run_embedding())
        print(f"\nEmbedding jobs queued: {total} objects")
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        print(f"Embedding failed: {e}")
        return 1


def cmd_promote(args):
    """Compute promotion scores."""
    from .config import get_config

    config = get_config()
    db_url = f"postgresql://gami:gami@localhost:5433/gami"
    batch_size = args.batch_size

    logger.info(f"Computing promotion scores for {args.type}")

    async def run_promotion():
        import asyncpg

        pool = await asyncpg.create_pool(db_url, min_size=1, max_size=3)
        total_queued = 0

        try:
            conn = await pool.acquire()
            try:
                object_types = []
                if args.type == "all":
                    object_types = ["segment", "claim", "entity"]
                else:
                    type_map = {
                        "segments": "segment",
                        "claims": "claim",
                        "entities": "entity",
                    }
                    object_types = [type_map.get(args.type, args.type)]

                for object_type in object_types:
                    table_map = {
                        "segment": "segments",
                        "claim": "claims",
                        "entity": "entities",
                    }
                    table = table_map.get(object_type, f"{object_type}s")

                    # Find objects needing score computation
                    # (no score or score older than 7 days)
                    objects = await conn.fetch(
                        f"""
                        SELECT o.id FROM {table} o
                        LEFT JOIN promotion_scores p ON p.object_id = o.id
                        WHERE p.id IS NULL
                           OR p.computed_at < NOW() - INTERVAL '7 days'
                        ORDER BY o.created_at DESC
                        LIMIT $1
                        """,
                        batch_size * 10,
                    )

                    if not objects:
                        print(f"No {object_type}s need scoring")
                        continue

                    from .tasks import compute_promotion_scores_batch
                    object_ids = [row["id"] for row in objects]

                    for i in range(0, len(object_ids), batch_size):
                        batch = object_ids[i:i + batch_size]
                        compute_promotion_scores_batch.delay(
                            object_ids=batch,
                            object_type=object_type,
                        )
                        total_queued += len(batch)

                    print(f"Queued {len(object_ids)} {object_type}s for promotion scoring")

            finally:
                await pool.release(conn)
        finally:
            await pool.close()

        return total_queued

    try:
        total = asyncio.get_event_loop().run_until_complete(run_promotion())
        print(f"\nPromotion scoring jobs queued: {total} objects")
    except Exception as e:
        logger.error(f"Promotion scoring failed: {e}")
        print(f"Promotion scoring failed: {e}")
        return 1


def cmd_stats(args):
    """Show system statistics."""
    from .config import get_config

    config = get_config()
    db_url = f"postgresql://gami:gami@localhost:5433/gami"

    async def get_stats():
        import asyncpg
        import redis.asyncio as redis

        db_stats = {"status": "disconnected", "tables": {}}
        cache_stats = {"status": "disconnected"}

        # Database stats
        try:
            pool = await asyncpg.create_pool(db_url, min_size=1, max_size=2, timeout=5)
            try:
                conn = await pool.acquire()
                try:
                    db_stats["status"] = "connected"

                    # Get table counts
                    tables = ["segments", "claims", "entities", "summaries",
                              "canonical_claims", "canonical_procedures",
                              "manifold_embeddings", "promotion_scores",
                              "shadow_comparisons", "query_logs"]

                    for table in tables:
                        try:
                            row = await conn.fetchrow(f"SELECT COUNT(*) as cnt FROM {table}")
                            db_stats["tables"][table] = row["cnt"]
                        except:
                            db_stats["tables"][table] = "N/A"

                    # Get promotion tier breakdown
                    tier_rows = await conn.fetch(
                        """
                        SELECT tier, COUNT(*) as cnt
                        FROM promotion_scores
                        GROUP BY tier
                        """
                    )
                    db_stats["promotion_tiers"] = {row["tier"]: row["cnt"] for row in tier_rows}

                    # Get embedding coverage
                    embed_rows = await conn.fetch(
                        """
                        SELECT manifold_type, COUNT(*) as cnt
                        FROM manifold_embeddings
                        GROUP BY manifold_type
                        """
                    )
                    db_stats["embeddings_by_manifold"] = {row["manifold_type"]: row["cnt"] for row in embed_rows}

                finally:
                    await pool.release(conn)
            finally:
                await pool.close()
        except Exception as e:
            db_stats["status"] = f"error: {e}"

        # Redis stats
        try:
            redis_client = redis.from_url(config.redis_url, decode_responses=True)
            try:
                info = await redis_client.info("memory")
                cache_stats["status"] = "connected"
                cache_stats["used_memory_human"] = info.get("used_memory_human", "N/A")
                cache_stats["keys"] = await redis_client.dbsize()
            finally:
                await redis_client.close()
        except Exception as e:
            cache_stats["status"] = f"error: {e}"

        return db_stats, cache_stats

    db_stats, cache_stats = asyncio.get_event_loop().run_until_complete(get_stats())

    stats = {
        "config": {
            "embedding_model": config.embedding_model,
            "embedding_dim": config.embedding_dim,
            "promotion_threshold": config.promotion_threshold,
            "demotion_threshold": config.demotion_threshold,
            "shadow_mode_enabled": config.shadow_mode_enabled,
        },
        "database": db_stats,
        "cache": cache_stats,
    }

    if args.json:
        print(json.dumps(stats, indent=2))
    else:
        print("\n=== Manifold System Statistics ===\n")
        print("Configuration:")
        for key, value in stats["config"].items():
            print(f"  {key}: {value}")

        print(f"\nDatabase: {db_stats['status']}")
        if db_stats.get("tables"):
            print("  Table counts:")
            for table, count in db_stats["tables"].items():
                print(f"    {table}: {count}")

        if db_stats.get("promotion_tiers"):
            print("  Promotion tiers:")
            for tier, count in db_stats["promotion_tiers"].items():
                print(f"    {tier}: {count}")

        if db_stats.get("embeddings_by_manifold"):
            print("  Embeddings by manifold:")
            for manifold, count in db_stats["embeddings_by_manifold"].items():
                print(f"    {manifold}: {count}")

        print(f"\nCache: {cache_stats['status']}")
        if cache_stats.get("keys"):
            print(f"  Keys: {cache_stats['keys']}")
        if cache_stats.get("used_memory_human"):
            print(f"  Memory: {cache_stats['used_memory_human']}")


def cmd_config(args):
    """Show or validate configuration."""
    from .config import ManifoldConfig, get_config
    from .validation import validate_config

    if args.validate:
        config = get_config()
        result = validate_config(config.to_dict())

        if result.valid:
            print("Configuration is valid")
            if result.warnings:
                print("\nWarnings:")
                for w in result.warnings:
                    print(f"  - {w}")
            return 0
        else:
            print("Configuration is INVALID")
            print("\nErrors:")
            for e in result.errors:
                print(f"  - {e}")
            return 1

    if args.show:
        config = get_config()
        if args.json:
            print(json.dumps(config.to_dict(), indent=2))
        else:
            print("\n=== Manifold Configuration ===\n")
            for key, value in sorted(config.to_dict().items()):
                print(f"{key}: {value}")

    if args.save:
        config = get_config()
        config.save(args.save)
        print(f"Configuration saved to {args.save}")


def cmd_cache(args):
    """Cache management commands."""
    from .config import get_config

    config = get_config()

    async def run_cache_command():
        import redis.asyncio as redis

        redis_client = redis.from_url(config.redis_url, decode_responses=True)

        try:
            if args.clear:
                # Clear manifold-related keys only
                cursor = 0
                cleared = 0
                while True:
                    cursor, keys = await redis_client.scan(
                        cursor=cursor,
                        match="manifold:*",
                        count=100,
                    )
                    if keys:
                        await redis_client.delete(*keys)
                        cleared += len(keys)
                    if cursor == 0:
                        break
                print(f"Cleared {cleared} cache entries")

            elif args.warm:
                from .tasks import warm_query_cache
                result = warm_query_cache.delay(
                    tenant_id=args.tenant,
                    query_count=100,
                )
                print(f"Cache warming job queued for tenant '{args.tenant}'")

            elif args.stats:
                info = await redis_client.info("memory")
                keyspace = await redis_client.info("keyspace")

                # Count manifold keys
                manifold_keys = 0
                cursor = 0
                while True:
                    cursor, keys = await redis_client.scan(
                        cursor=cursor,
                        match="manifold:*",
                        count=100,
                    )
                    manifold_keys += len(keys)
                    if cursor == 0:
                        break

                print("\n=== Cache Statistics ===\n")
                print(f"Total keys: {await redis_client.dbsize()}")
                print(f"Manifold keys: {manifold_keys}")
                print(f"Used memory: {info.get('used_memory_human', 'N/A')}")
                print(f"Peak memory: {info.get('used_memory_peak_human', 'N/A')}")
                print(f"Evicted keys: {info.get('evicted_keys', 0)}")
                print(f"Hit rate: {info.get('keyspace_hits', 0)}/{info.get('keyspace_misses', 0) + info.get('keyspace_hits', 0)}")

        finally:
            await redis_client.close()

    try:
        asyncio.get_event_loop().run_until_complete(run_cache_command())
    except Exception as e:
        logger.error(f"Cache command failed: {e}")
        print(f"Cache command failed: {e}")
        return 1


def cmd_shadow(args):
    """Shadow mode analysis."""
    from .config import get_config

    config = get_config()
    db_url = f"postgresql://gami:gami@localhost:5433/gami"

    async def run_shadow_command():
        import asyncpg

        pool = await asyncpg.create_pool(db_url, min_size=1, max_size=2, timeout=5)

        try:
            conn = await pool.acquire()
            try:
                if args.stats:
                    # Get shadow comparison statistics
                    stats = await conn.fetchrow(
                        """
                        SELECT
                            COUNT(*) as total,
                            SUM(CASE WHEN manifold_better THEN 1 ELSE 0 END) as manifold_wins,
                            SUM(CASE WHEN legacy_better THEN 1 ELSE 0 END) as legacy_wins,
                            AVG(overlap_ratio) as avg_overlap,
                            AVG(manifold_latency_ms) as avg_manifold_latency,
                            AVG(legacy_latency_ms) as avg_legacy_latency
                        FROM shadow_comparisons
                        WHERE created_at > NOW() - INTERVAL '24 hours'
                        """
                    )

                    # Recent trends
                    hourly = await conn.fetch(
                        """
                        SELECT
                            date_trunc('hour', created_at) as hour,
                            COUNT(*) as count,
                            AVG(overlap_ratio) as overlap
                        FROM shadow_comparisons
                        WHERE created_at > NOW() - INTERVAL '24 hours'
                        GROUP BY hour
                        ORDER BY hour DESC
                        LIMIT 12
                        """
                    )

                    print("\n=== Shadow Mode Statistics (last 24h) ===\n")
                    print(f"Shadow mode: {'ENABLED' if config.shadow_mode_enabled else 'DISABLED'}")
                    print(f"\nTotal comparisons: {stats['total'] or 0}")
                    print(f"Manifold wins: {stats['manifold_wins'] or 0}")
                    print(f"Legacy wins: {stats['legacy_wins'] or 0}")
                    print(f"Average overlap: {(stats['avg_overlap'] or 0):.2%}")
                    print(f"Avg manifold latency: {(stats['avg_manifold_latency'] or 0):.1f}ms")
                    print(f"Avg legacy latency: {(stats['avg_legacy_latency'] or 0):.1f}ms")

                    if stats['total'] and stats['total'] > 0:
                        win_rate = (stats['manifold_wins'] or 0) / stats['total']
                        print(f"\nManifold win rate: {win_rate:.1%}")

                        if win_rate > 0.6 and (stats['avg_overlap'] or 0) > 0.7:
                            print("\nRecommendation: Ready for production switch")
                        elif stats['total'] < 100:
                            print("\nRecommendation: Insufficient data (need 100+ comparisons)")
                        else:
                            print("\nRecommendation: Continue tuning")

                    if hourly:
                        print("\nHourly trends:")
                        for row in hourly[:6]:
                            print(f"  {row['hour'].strftime('%H:%M')}: {row['count']} comparisons, {row['overlap']:.1%} overlap")

                elif args.enable:
                    # Update configuration
                    await conn.execute(
                        """
                        INSERT INTO manifold_config (key, value, updated_at)
                        VALUES ('shadow_mode_enabled', 'true', NOW())
                        ON CONFLICT (key) DO UPDATE SET value = 'true', updated_at = NOW()
                        """
                    )
                    print("Shadow mode ENABLED")
                    print("New queries will run both manifold and legacy retrieval for comparison")

                elif args.disable:
                    await conn.execute(
                        """
                        INSERT INTO manifold_config (key, value, updated_at)
                        VALUES ('shadow_mode_enabled', 'false', NOW())
                        ON CONFLICT (key) DO UPDATE SET value = 'false', updated_at = NOW()
                        """
                    )
                    print("Shadow mode DISABLED")

            finally:
                await pool.release(conn)
        finally:
            await pool.close()

    try:
        asyncio.get_event_loop().run_until_complete(run_shadow_command())
    except Exception as e:
        logger.error(f"Shadow command failed: {e}")
        print(f"Shadow command failed: {e}")
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Manifold system administration CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # migrate command
    migrate_parser = subparsers.add_parser("migrate", help="Run migrations")
    migrate_parser.add_argument("--claims", action="store_true", help="Migrate claims to canonical form")
    migrate_parser.add_argument("--procedures", action="store_true", help="Migrate procedures")
    migrate_parser.add_argument("--all", action="store_true", help="Run all migrations")
    migrate_parser.set_defaults(func=cmd_migrate)

    # embed command
    embed_parser = subparsers.add_parser("embed", help="Generate embeddings")
    embed_parser.add_argument("--type", choices=["segments", "claims", "entities", "all"], default="all")
    embed_parser.add_argument("--batch-size", type=int, default=100)
    embed_parser.add_argument("--tenant", default="shared")
    embed_parser.set_defaults(func=cmd_embed)

    # promote command
    promote_parser = subparsers.add_parser("promote", help="Compute promotion scores")
    promote_parser.add_argument("--type", choices=["segments", "claims", "entities", "all"], default="all")
    promote_parser.add_argument("--batch-size", type=int, default=500)
    promote_parser.set_defaults(func=cmd_promote)

    # stats command
    stats_parser = subparsers.add_parser("stats", help="Show statistics")
    stats_parser.add_argument("--json", action="store_true", help="Output as JSON")
    stats_parser.set_defaults(func=cmd_stats)

    # config command
    config_parser = subparsers.add_parser("config", help="Configuration management")
    config_parser.add_argument("--show", action="store_true", help="Show current config")
    config_parser.add_argument("--validate", action="store_true", help="Validate config")
    config_parser.add_argument("--save", metavar="FILE", help="Save config to file")
    config_parser.add_argument("--json", action="store_true", help="Output as JSON")
    config_parser.set_defaults(func=cmd_config)

    # cache command
    cache_parser = subparsers.add_parser("cache", help="Cache management")
    cache_parser.add_argument("--clear", action="store_true", help="Clear all caches")
    cache_parser.add_argument("--warm", action="store_true", help="Warm caches")
    cache_parser.add_argument("--stats", action="store_true", help="Show cache stats")
    cache_parser.add_argument("--tenant", default="shared")
    cache_parser.set_defaults(func=cmd_cache)

    # shadow command
    shadow_parser = subparsers.add_parser("shadow", help="Shadow mode management")
    shadow_parser.add_argument("--stats", action="store_true", help="Show shadow stats")
    shadow_parser.add_argument("--enable", action="store_true", help="Enable shadow mode")
    shadow_parser.add_argument("--disable", action="store_true", help="Disable shadow mode")
    shadow_parser.set_defaults(func=cmd_shadow)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    try:
        result = args.func(args)
        return result if result is not None else 0
    except Exception as e:
        logger.error(f"Command failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
