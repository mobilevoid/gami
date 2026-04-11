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

    if args.claims:
        logger.info("Running canonical claims migration...")
        # Would call migrate_canonical_claims
        print("Canonical claims migration: stub (run scripts/migrate_canonical_claims.py)")

    if args.procedures:
        logger.info("Running canonical procedures migration...")
        print("Canonical procedures migration: stub")

    if args.all:
        logger.info("Running all migrations...")
        print("All migrations: stub")


def cmd_embed(args):
    """Generate embeddings."""
    from .config import get_config

    logger.info(f"Embedding {args.type} objects (batch_size={args.batch_size})")

    if args.type == "all":
        print("Embedding all object types: stub")
    else:
        print(f"Embedding {args.type}: stub (run scripts/embed_promoted.py)")


def cmd_promote(args):
    """Compute promotion scores."""
    logger.info(f"Computing promotion scores for {args.type}")

    print("Promotion scoring: stub (run scripts/compute_promotion_scores.py)")


def cmd_stats(args):
    """Show system statistics."""
    from .config import get_config

    config = get_config()

    stats = {
        "config": {
            "embedding_model": config.embedding_model,
            "embedding_dim": config.embedding_dim,
            "promotion_threshold": config.promotion_threshold,
            "demotion_threshold": config.demotion_threshold,
            "shadow_mode_enabled": config.shadow_mode_enabled,
        },
        "database": {
            "status": "not connected (isolated module)",
        },
        "cache": {
            "status": "not connected (isolated module)",
        },
    }

    if args.json:
        print(json.dumps(stats, indent=2))
    else:
        print("\n=== Manifold System Statistics ===\n")
        print("Configuration:")
        for key, value in stats["config"].items():
            print(f"  {key}: {value}")
        print("\nDatabase:", stats["database"]["status"])
        print("Cache:", stats["cache"]["status"])


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
    if args.clear:
        print("Clearing cache: stub (would clear Redis)")
    elif args.warm:
        print(f"Warming cache for tenant {args.tenant}: stub")
    elif args.stats:
        print("Cache stats: stub (not connected)")


def cmd_shadow(args):
    """Shadow mode analysis."""
    if args.stats:
        print("\n=== Shadow Mode Statistics ===\n")
        print("Status: stub (not connected to database)")
        print("Total comparisons: 0")
        print("Match rate: N/A")
    elif args.enable:
        print("Shadow mode enabled")
    elif args.disable:
        print("Shadow mode disabled")


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
