"""
Hot-reloadable configuration loader for Manifold.

Supports:
- File-based config watching
- Database-based config with tenant/agent overrides
- Hot-reload without restart
- Config caching with TTL
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import asdict

from .config_v2 import ManifoldConfigV2, ScoringWeights

logger = logging.getLogger("manifold.config_loader")


class ConfigLoader:
    """Hot-reloadable configuration manager.

    Manages configuration from multiple sources with caching:
    - Environment variables (base)
    - Config file (optional)
    - Database manifold_config table
    - Per-tenant overrides
    - Per-agent overrides

    Usage:
        loader = ConfigLoader()
        config = await loader.get_config(tenant_id="my-tenant", agent_id="my-agent")
    """

    def __init__(
        self,
        config_file: Optional[Path] = None,
        db_url: Optional[str] = None,
        cache_ttl_seconds: int = 60,
    ):
        """Initialize the config loader.

        Args:
            config_file: Optional path to JSON config file
            db_url: Database URL for fetching overrides
            cache_ttl_seconds: How long to cache configs before refresh
        """
        self.config_file = config_file
        self.db_url = db_url
        self.cache_ttl = timedelta(seconds=cache_ttl_seconds)

        # Cache: {cache_key: (config, loaded_at)}
        self._cache: Dict[str, Tuple[ManifoldConfigV2, datetime]] = {}

        # Base config (from env + file)
        self._base_config: Optional[ManifoldConfigV2] = None
        self._base_loaded_at: Optional[datetime] = None

        # Database config cache
        self._db_config: Dict[str, Dict[str, Any]] = {}
        self._db_loaded_at: Optional[datetime] = None

    async def get_config(
        self,
        tenant_id: str = "*",
        agent_id: Optional[str] = None,
    ) -> ManifoldConfigV2:
        """Get effective config with tenant/agent overrides.

        Args:
            tenant_id: Tenant ID for tenant-specific overrides
            agent_id: Agent ID for agent-specific overrides

        Returns:
            ManifoldConfigV2 with all overrides applied
        """
        cache_key = f"{tenant_id}:{agent_id or '*'}"

        # Check cache
        if cache_key in self._cache:
            config, loaded_at = self._cache[cache_key]
            if datetime.now() - loaded_at < self.cache_ttl:
                return config

        # Build layered config
        config = await self._build_config(tenant_id, agent_id)

        # Cache it
        self._cache[cache_key] = (config, datetime.now())

        return config

    async def _build_config(
        self,
        tenant_id: str,
        agent_id: Optional[str],
    ) -> ManifoldConfigV2:
        """Build config by layering sources."""

        # 1. Start with base config (env + file)
        base = await self._get_base_config()
        config_dict = asdict(base)

        # 2. Apply database global defaults
        db_overrides = await self._get_db_config("*", None)
        if db_overrides:
            _deep_merge(config_dict, db_overrides)

        # 3. Apply tenant overrides (if not global)
        if tenant_id != "*":
            tenant_overrides = await self._get_db_config(tenant_id, None)
            if tenant_overrides:
                _deep_merge(config_dict, tenant_overrides)

        # 4. Apply agent overrides
        if agent_id:
            agent_overrides = await self._get_db_config(tenant_id, agent_id)
            if agent_overrides:
                _deep_merge(config_dict, agent_overrides)

            # Also check for agent config in agent_configs table
            agent_config = await self._get_agent_config(agent_id)
            if agent_config:
                _deep_merge(config_dict, agent_config)

        # Reconstruct config object
        return ManifoldConfigV2._from_dict(config_dict)

    async def _get_base_config(self) -> ManifoldConfigV2:
        """Get base config from env and optional file."""
        if self._base_config is not None:
            # Check if file changed
            if self.config_file and self._file_changed():
                self._base_config = None

        if self._base_config is None:
            # Load from env
            self._base_config = ManifoldConfigV2.from_env()

            # Overlay file config if present
            if self.config_file and self.config_file.exists():
                try:
                    file_config = ManifoldConfigV2.from_file(self.config_file)
                    # Merge file over env
                    base_dict = asdict(self._base_config)
                    file_dict = asdict(file_config)
                    _deep_merge(base_dict, file_dict)
                    self._base_config = ManifoldConfigV2._from_dict(base_dict)
                except Exception as e:
                    logger.error(f"Failed to load config file: {e}")

            self._base_loaded_at = datetime.now()

        return self._base_config

    def _file_changed(self) -> bool:
        """Check if config file was modified since last load."""
        if not self.config_file or not self._base_loaded_at:
            return False

        try:
            mtime = datetime.fromtimestamp(self.config_file.stat().st_mtime)
            return mtime > self._base_loaded_at
        except Exception:
            return False

    async def _get_db_config(
        self,
        tenant_id: str,
        agent_id: Optional[str],
    ) -> Dict[str, Any]:
        """Get config overrides from database."""
        if not self.db_url:
            return {}

        # Refresh DB cache if stale
        if (
            self._db_loaded_at is None
            or datetime.now() - self._db_loaded_at > self.cache_ttl
        ):
            await self._refresh_db_cache()

        # Build lookup key
        key = f"{tenant_id}:{agent_id or '*'}"
        return self._db_config.get(key, {})

    async def _refresh_db_cache(self) -> None:
        """Refresh database config cache."""
        try:
            import asyncpg

            conn = await asyncpg.connect(self.db_url)
            try:
                rows = await conn.fetch("""
                    SELECT tenant_id, config_key, config_value
                    FROM manifold_config
                    WHERE config_value IS NOT NULL
                """)

                self._db_config.clear()

                for row in rows:
                    tenant = row["tenant_id"]
                    key = row["config_key"]
                    value = row["config_value"]

                    # Parse JSON value
                    if isinstance(value, str):
                        try:
                            value = json.loads(value)
                        except json.JSONDecodeError:
                            pass

                    # Store in nested structure
                    cache_key = f"{tenant}:*"
                    if cache_key not in self._db_config:
                        self._db_config[cache_key] = {}

                    # Handle dotted keys (e.g., "scoring.evidence_authority")
                    _set_nested(self._db_config[cache_key], key, value)

                self._db_loaded_at = datetime.now()

            finally:
                await conn.close()

        except ImportError:
            logger.warning("asyncpg not available, skipping DB config")
        except Exception as e:
            logger.error(f"Failed to load DB config: {e}")

    async def _get_agent_config(self, agent_id: str) -> Dict[str, Any]:
        """Get agent-specific config from agent_configs table."""
        if not self.db_url:
            return {}

        try:
            import asyncpg

            conn = await asyncpg.connect(self.db_url)
            try:
                row = await conn.fetchrow("""
                    SELECT
                        default_temperature,
                        default_max_tokens,
                        scoring_overrides,
                        extraction_prompt_overrides
                    FROM agent_configs
                    WHERE agent_id = $1 AND status = 'active'
                """, agent_id)

                if not row:
                    return {}

                result = {}

                # Map agent config to manifold config structure
                if row["scoring_overrides"]:
                    scoring = row["scoring_overrides"]
                    if isinstance(scoring, str):
                        scoring = json.loads(scoring)
                    result["scoring"] = scoring

                return result

            finally:
                await conn.close()

        except Exception as e:
            logger.error(f"Failed to get agent config: {e}")
            return {}

    def invalidate(
        self,
        tenant_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> int:
        """Invalidate cached configs.

        Args:
            tenant_id: If provided, only invalidate this tenant
            agent_id: If provided, only invalidate this agent

        Returns:
            Number of cache entries invalidated
        """
        if tenant_id is None and agent_id is None:
            # Invalidate everything
            count = len(self._cache)
            self._cache.clear()
            self._db_config.clear()
            self._db_loaded_at = None
            return count

        # Selective invalidation
        keys_to_remove = []
        for key in self._cache:
            t, a = key.split(":", 1)
            if tenant_id and t != tenant_id:
                continue
            if agent_id and a != agent_id and a != "*":
                continue
            keys_to_remove.append(key)

        for key in keys_to_remove:
            del self._cache[key]

        return len(keys_to_remove)

    async def set_config(
        self,
        tenant_id: str,
        config_key: str,
        config_value: Any,
        agent_id: Optional[str] = None,
    ) -> bool:
        """Set a config value in the database.

        Args:
            tenant_id: Tenant to set config for ('*' for global)
            config_key: Config key (e.g., 'scoring.evidence_authority')
            config_value: Value to set
            agent_id: Optional agent to set config for

        Returns:
            True if successful
        """
        if not self.db_url:
            logger.error("No database URL configured")
            return False

        try:
            import asyncpg

            conn = await asyncpg.connect(self.db_url)
            try:
                value_json = json.dumps(config_value)

                await conn.execute("""
                    INSERT INTO manifold_config (tenant_id, config_key, config_value)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (tenant_id, config_key)
                    DO UPDATE SET config_value = $3, updated_at = NOW()
                """, tenant_id, config_key, value_json)

                # Invalidate cache
                self.invalidate(tenant_id=tenant_id, agent_id=agent_id)

                return True

            finally:
                await conn.close()

        except Exception as e:
            logger.error(f"Failed to set config: {e}")
            return False


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _deep_merge(base: Dict, override: Dict) -> None:
    """Deep merge override into base dict (mutates base)."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def _set_nested(d: Dict, key: str, value: Any) -> None:
    """Set a nested dict value using dotted key."""
    parts = key.split(".")
    for part in parts[:-1]:
        if part not in d:
            d[part] = {}
        d = d[part]
    d[parts[-1]] = value


# =============================================================================
# GLOBAL LOADER SINGLETON
# =============================================================================

_loader: Optional[ConfigLoader] = None


def get_loader() -> ConfigLoader:
    """Get the global config loader."""
    global _loader
    if _loader is None:
        from .config_v2 import get_config
        base = get_config()
        _loader = ConfigLoader(db_url=base.database_url)
    return _loader


async def get_config_for_context(
    tenant_id: str = "*",
    agent_id: Optional[str] = None,
) -> ManifoldConfigV2:
    """Get config for a specific tenant/agent context.

    This is the main entry point for getting config in request handlers.
    """
    loader = get_loader()
    return await loader.get_config(tenant_id=tenant_id, agent_id=agent_id)


def invalidate_config(
    tenant_id: Optional[str] = None,
    agent_id: Optional[str] = None,
) -> int:
    """Invalidate cached configs globally."""
    return get_loader().invalidate(tenant_id=tenant_id, agent_id=agent_id)
