"""Entry point for the GAMI Subconscious Daemon.

Usage:
    python -m daemon.subconscious
"""
import asyncio
import logging
import signal
import sys

from .subconscious import SubconsciousDaemon

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("gami.daemon")


async def main():
    """Main entry point."""
    from manifold.config_v2 import get_config

    config = get_config()
    daemon = SubconsciousDaemon(config)

    # Handle shutdown signals
    loop = asyncio.get_event_loop()

    def shutdown_handler():
        logger.info("Shutdown signal received")
        asyncio.create_task(daemon.stop())

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, shutdown_handler)

    logger.info("Starting GAMI Subconscious Daemon...")

    try:
        await daemon.start()
    except Exception as e:
        logger.error(f"Daemon failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
