"""
Agent configuration service for multi-agent support.

Manages:
- Agent credentials with AES-256-GCM encryption
- Per-agent LLM configuration
- Personality and scoring overrides
- Trust metrics and history
- Rate limiting and token budgets
"""

import os
import json
import secrets
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from base64 import b64encode, b64decode

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger("gami.agent_service")


# =============================================================================
# CREDENTIAL ENCRYPTION
# =============================================================================

def _get_encryption_key() -> bytes:
    """Get or derive encryption key from environment.

    Uses GAMI_AGENT_KEY env var, or derives from GAMI_SECRET_KEY.
    """
    agent_key = os.environ.get("GAMI_AGENT_KEY")
    if agent_key:
        # Direct key (must be 32 bytes base64-encoded)
        return b64decode(agent_key)

    secret = os.environ.get("GAMI_SECRET_KEY", "gami-default-secret-change-me")

    # Derive key using PBKDF2
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=b"gami-agent-credentials",
        iterations=100000,
    )
    return kdf.derive(secret.encode())


def encrypt_credentials(credentials: Dict[str, str]) -> Tuple[bytes, str]:
    """Encrypt agent credentials using AES-256-GCM.

    Args:
        credentials: Dict of credential key-value pairs

    Returns:
        Tuple of (encrypted_bytes, key_id)
    """
    key = _get_encryption_key()
    key_id = "env:GAMI_AGENT_KEY" if os.environ.get("GAMI_AGENT_KEY") else "derived"

    # Generate nonce (12 bytes for GCM)
    nonce = secrets.token_bytes(12)

    # Encrypt
    aesgcm = AESGCM(key)
    plaintext = json.dumps(credentials).encode()
    ciphertext = aesgcm.encrypt(nonce, plaintext, None)

    # Prepend nonce to ciphertext
    encrypted = nonce + ciphertext

    return encrypted, key_id


def decrypt_credentials(encrypted: bytes, key_id: str) -> Dict[str, str]:
    """Decrypt agent credentials.

    Args:
        encrypted: Encrypted bytes (nonce + ciphertext)
        key_id: Key identifier (for future key rotation)

    Returns:
        Decrypted credentials dict
    """
    key = _get_encryption_key()

    # Extract nonce and ciphertext
    nonce = encrypted[:12]
    ciphertext = encrypted[12:]

    # Decrypt
    aesgcm = AESGCM(key)
    plaintext = aesgcm.decrypt(nonce, ciphertext, None)

    return json.loads(plaintext.decode())


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class AgentConfig:
    """Agent configuration data."""
    agent_id: str
    owner_tenant_id: str
    agent_name: str
    agent_type: str = "assistant"

    # LLM Configuration
    default_model: Optional[str] = None
    endpoint_url: Optional[str] = None
    default_temperature: float = 0.7
    default_max_tokens: int = 2048
    context_window_size: int = 8192

    # Provider Configuration (NEW)
    llm_provider: Optional[str] = None  # vllm, ollama, openai, anthropic
    embedding_provider: Optional[str] = None  # sentence_transformers, ollama, openai
    embedding_model: Optional[str] = None
    embedding_device: Optional[str] = None  # auto, cpu, cuda, mps

    # Personality
    personality_json: Dict[str, Any] = None
    system_prompt_override: Optional[str] = None
    extraction_prompt_overrides: Dict[str, Any] = None
    scoring_overrides: Dict[str, Any] = None

    # Rate Limits
    rate_limit_rpm: int = 60
    token_budget_daily: int = 1000000
    tokens_used_today: int = 0

    # Trust Metrics
    accuracy_score: float = 0.5
    verified_claims: int = 0
    disputed_claims: int = 0
    total_claims: int = 0

    # Status
    status: str = "active"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def __post_init__(self):
        if self.personality_json is None:
            self.personality_json = {}
        if self.extraction_prompt_overrides is None:
            self.extraction_prompt_overrides = {}
        if self.scoring_overrides is None:
            self.scoring_overrides = {}

    def get_llm_provider_config(self) -> Dict[str, Any]:
        """Get LLM provider configuration for this agent."""
        return {
            "provider": self.llm_provider,
            "model": self.default_model,
            "base_url": self.endpoint_url,
            "temperature": self.default_temperature,
            "max_tokens": self.default_max_tokens,
        }

    def get_embedding_provider_config(self) -> Dict[str, Any]:
        """Get embedding provider configuration for this agent."""
        return {
            "provider": self.embedding_provider,
            "model": self.embedding_model,
            "device": self.embedding_device,
        }


@dataclass
class AgentCredentials:
    """Decrypted agent credentials (transient, never persisted)."""
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    bearer_token: Optional[str] = None
    custom: Dict[str, str] = None

    def __post_init__(self):
        if self.custom is None:
            self.custom = {}


@dataclass
class AgentTrustMetrics:
    """Agent trust scoring metrics."""
    agent_id: str
    accuracy_score: float
    verified_claims: int
    disputed_claims: int
    total_claims: int
    trust_trend: List[Dict[str, Any]]  # Historical data


# =============================================================================
# AGENT SERVICE
# =============================================================================

class AgentService:
    """Service for managing agent configurations."""

    def __init__(self):
        """Initialize agent service."""
        self._cache: Dict[str, Tuple[AgentConfig, datetime]] = {}
        self._cache_ttl = timedelta(minutes=5)

    async def create_agent(
        self,
        db: AsyncSession,
        agent_id: str,
        agent_name: str,
        owner_tenant_id: str,
        agent_type: str = "assistant",
        credentials: Optional[Dict[str, str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> AgentConfig:
        """Create a new agent configuration.

        Args:
            db: Database session
            agent_id: Unique agent identifier
            agent_name: Human-readable name
            owner_tenant_id: Owning tenant
            agent_type: Type (assistant, tool, background, human)
            credentials: Optional API credentials to encrypt
            config: Additional configuration options

        Returns:
            Created AgentConfig
        """
        config = config or {}

        # Encrypt credentials if provided
        encrypted_creds = None
        key_id = None
        if credentials:
            encrypted_creds, key_id = encrypt_credentials(credentials)

        # Insert agent config
        await db.execute(text("""
            INSERT INTO agent_configs (
                agent_id, owner_tenant_id, agent_name, agent_type,
                default_model, endpoint_url, credentials_encrypted, credential_key_id,
                default_temperature, default_max_tokens, context_window_size,
                system_prompt_override, personality_json,
                extraction_prompt_overrides, scoring_overrides,
                rate_limit_rpm, token_budget_daily
            ) VALUES (
                :agent_id, :tenant_id, :name, :type,
                :model, :endpoint, :creds, :key_id,
                :temp, :tokens, :context,
                :system_prompt, :personality,
                :extraction_overrides, :scoring_overrides,
                :rate_limit, :token_budget
            )
        """), {
            "agent_id": agent_id,
            "tenant_id": owner_tenant_id,
            "name": agent_name,
            "type": agent_type,
            "model": config.get("default_model"),
            "endpoint": config.get("endpoint_url"),
            "creds": encrypted_creds,
            "key_id": key_id,
            "temp": config.get("default_temperature", 0.7),
            "tokens": config.get("default_max_tokens", 2048),
            "context": config.get("context_window_size", 8192),
            "system_prompt": config.get("system_prompt_override"),
            "personality": json.dumps(config.get("personality_json", {})),
            "extraction_overrides": json.dumps(config.get("extraction_prompt_overrides", {})),
            "scoring_overrides": json.dumps(config.get("scoring_overrides", {})),
            "rate_limit": config.get("rate_limit_rpm", 60),
            "token_budget": config.get("token_budget_daily", 1000000),
        })

        await db.commit()

        logger.info(f"Created agent: {agent_id} for tenant {owner_tenant_id}")

        return AgentConfig(
            agent_id=agent_id,
            owner_tenant_id=owner_tenant_id,
            agent_name=agent_name,
            agent_type=agent_type,
            default_model=config.get("default_model"),
            endpoint_url=config.get("endpoint_url"),
            default_temperature=config.get("default_temperature", 0.7),
            default_max_tokens=config.get("default_max_tokens", 2048),
            created_at=datetime.now(),
        )

    async def get_agent(
        self,
        db: AsyncSession,
        agent_id: str,
        include_credentials: bool = False,
    ) -> Optional[Tuple[AgentConfig, Optional[AgentCredentials]]]:
        """Get agent configuration by ID.

        Args:
            db: Database session
            agent_id: Agent ID to lookup
            include_credentials: Whether to decrypt and include credentials

        Returns:
            Tuple of (AgentConfig, AgentCredentials) or None
        """
        # Check cache first (without credentials)
        if agent_id in self._cache and not include_credentials:
            config, loaded_at = self._cache[agent_id]
            if datetime.now() - loaded_at < self._cache_ttl:
                return config, None

        result = await db.execute(text("""
            SELECT
                agent_id, owner_tenant_id, agent_name, agent_type,
                default_model, endpoint_url, credentials_encrypted, credential_key_id,
                default_temperature, default_max_tokens, context_window_size,
                system_prompt_override, personality_json,
                extraction_prompt_overrides, scoring_overrides,
                rate_limit_rpm, token_budget_daily, tokens_used_today,
                accuracy_score, verified_claims, disputed_claims, total_claims,
                status, created_at, updated_at
            FROM agent_configs
            WHERE agent_id = :agent_id
        """), {"agent_id": agent_id})

        row = result.fetchone()
        if row is None:
            return None

        config = AgentConfig(
            agent_id=row.agent_id,
            owner_tenant_id=row.owner_tenant_id,
            agent_name=row.agent_name,
            agent_type=row.agent_type,
            default_model=row.default_model,
            endpoint_url=row.endpoint_url,
            default_temperature=row.default_temperature,
            default_max_tokens=row.default_max_tokens,
            context_window_size=row.context_window_size,
            system_prompt_override=row.system_prompt_override,
            personality_json=row.personality_json or {},
            extraction_prompt_overrides=row.extraction_prompt_overrides or {},
            scoring_overrides=row.scoring_overrides or {},
            rate_limit_rpm=row.rate_limit_rpm,
            token_budget_daily=row.token_budget_daily,
            tokens_used_today=row.tokens_used_today,
            accuracy_score=row.accuracy_score,
            verified_claims=row.verified_claims,
            disputed_claims=row.disputed_claims,
            total_claims=row.total_claims,
            status=row.status,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

        # Cache without credentials
        self._cache[agent_id] = (config, datetime.now())

        # Decrypt credentials if requested
        credentials = None
        if include_credentials and row.credentials_encrypted:
            try:
                creds_dict = decrypt_credentials(
                    row.credentials_encrypted,
                    row.credential_key_id,
                )
                credentials = AgentCredentials(
                    api_key=creds_dict.get("api_key"),
                    api_secret=creds_dict.get("api_secret"),
                    bearer_token=creds_dict.get("bearer_token"),
                    custom={k: v for k, v in creds_dict.items()
                           if k not in ("api_key", "api_secret", "bearer_token")},
                )
            except Exception as e:
                logger.error(f"Failed to decrypt credentials for {agent_id}: {e}")

        return config, credentials

    async def update_agent(
        self,
        db: AsyncSession,
        agent_id: str,
        updates: Dict[str, Any],
        new_credentials: Optional[Dict[str, str]] = None,
    ) -> bool:
        """Update agent configuration.

        Args:
            db: Database session
            agent_id: Agent ID to update
            updates: Fields to update
            new_credentials: New credentials to encrypt

        Returns:
            True if updated successfully
        """
        # Build dynamic update query
        set_clauses = ["updated_at = NOW()"]
        params = {"agent_id": agent_id}

        field_mapping = {
            "agent_name": "agent_name",
            "agent_type": "agent_type",
            "default_model": "default_model",
            "endpoint_url": "endpoint_url",
            "default_temperature": "default_temperature",
            "default_max_tokens": "default_max_tokens",
            "context_window_size": "context_window_size",
            "system_prompt_override": "system_prompt_override",
            "rate_limit_rpm": "rate_limit_rpm",
            "token_budget_daily": "token_budget_daily",
            "status": "status",
        }

        for key, column in field_mapping.items():
            if key in updates:
                set_clauses.append(f"{column} = :{key}")
                params[key] = updates[key]

        # Handle JSON fields
        json_fields = ["personality_json", "extraction_prompt_overrides", "scoring_overrides"]
        for field in json_fields:
            if field in updates:
                set_clauses.append(f"{field} = :{field}")
                params[field] = json.dumps(updates[field])

        # Handle new credentials
        if new_credentials:
            encrypted, key_id = encrypt_credentials(new_credentials)
            set_clauses.append("credentials_encrypted = :creds")
            set_clauses.append("credential_key_id = :key_id")
            params["creds"] = encrypted
            params["key_id"] = key_id

        query = text(f"""
            UPDATE agent_configs
            SET {', '.join(set_clauses)}
            WHERE agent_id = :agent_id
            RETURNING agent_id
        """)

        result = await db.execute(query, params)
        updated = result.fetchone() is not None

        if updated:
            await db.commit()
            # Invalidate cache
            self._cache.pop(agent_id, None)
            logger.info(f"Updated agent: {agent_id}")

        return updated

    async def delete_agent(
        self,
        db: AsyncSession,
        agent_id: str,
        hard_delete: bool = False,
    ) -> bool:
        """Delete or deactivate an agent.

        Args:
            db: Database session
            agent_id: Agent ID to delete
            hard_delete: If True, permanently delete; otherwise soft delete

        Returns:
            True if deleted
        """
        if hard_delete:
            query = text("DELETE FROM agent_configs WHERE agent_id = :agent_id RETURNING agent_id")
        else:
            query = text("""
                UPDATE agent_configs
                SET status = 'deleted', updated_at = NOW()
                WHERE agent_id = :agent_id
                RETURNING agent_id
            """)

        result = await db.execute(query, {"agent_id": agent_id})
        deleted = result.fetchone() is not None

        if deleted:
            await db.commit()
            self._cache.pop(agent_id, None)
            logger.info(f"{'Deleted' if hard_delete else 'Deactivated'} agent: {agent_id}")

        return deleted

    async def list_agents(
        self,
        db: AsyncSession,
        tenant_id: Optional[str] = None,
        agent_type: Optional[str] = None,
        status: str = "active",
    ) -> List[AgentConfig]:
        """List agents with optional filtering.

        Args:
            db: Database session
            tenant_id: Filter by owner tenant
            agent_type: Filter by agent type
            status: Filter by status

        Returns:
            List of AgentConfig objects
        """
        conditions = ["status = :status"]
        params = {"status": status}

        if tenant_id:
            conditions.append("owner_tenant_id = :tenant_id")
            params["tenant_id"] = tenant_id

        if agent_type:
            conditions.append("agent_type = :agent_type")
            params["agent_type"] = agent_type

        query = text(f"""
            SELECT
                agent_id, owner_tenant_id, agent_name, agent_type,
                default_model, endpoint_url,
                default_temperature, default_max_tokens, context_window_size,
                rate_limit_rpm, token_budget_daily,
                accuracy_score, verified_claims, disputed_claims, total_claims,
                status, created_at, updated_at
            FROM agent_configs
            WHERE {' AND '.join(conditions)}
            ORDER BY created_at DESC
        """)

        result = await db.execute(query, params)

        return [
            AgentConfig(
                agent_id=row.agent_id,
                owner_tenant_id=row.owner_tenant_id,
                agent_name=row.agent_name,
                agent_type=row.agent_type,
                default_model=row.default_model,
                endpoint_url=row.endpoint_url,
                default_temperature=row.default_temperature,
                default_max_tokens=row.default_max_tokens,
                context_window_size=row.context_window_size,
                rate_limit_rpm=row.rate_limit_rpm,
                token_budget_daily=row.token_budget_daily,
                accuracy_score=row.accuracy_score,
                verified_claims=row.verified_claims,
                disputed_claims=row.disputed_claims,
                total_claims=row.total_claims,
                status=row.status,
                created_at=row.created_at,
                updated_at=row.updated_at,
            )
            for row in result.fetchall()
        ]


# =============================================================================
# TRUST SCORING
# =============================================================================

class AgentTrustService:
    """Service for managing agent trust scores."""

    async def get_trust_metrics(
        self,
        db: AsyncSession,
        agent_id: str,
        history_days: int = 30,
    ) -> Optional[AgentTrustMetrics]:
        """Get trust metrics for an agent with history.

        Args:
            db: Database session
            agent_id: Agent ID
            history_days: Days of history to include

        Returns:
            AgentTrustMetrics or None
        """
        # Get current metrics
        result = await db.execute(text("""
            SELECT accuracy_score, verified_claims, disputed_claims, total_claims
            FROM agent_configs
            WHERE agent_id = :agent_id
        """), {"agent_id": agent_id})

        row = result.fetchone()
        if row is None:
            return None

        # Get historical trend
        history_result = await db.execute(text("""
            SELECT accuracy_score, verified_claims, disputed_claims, recorded_at
            FROM agent_trust_history
            WHERE agent_id = :agent_id
            AND recorded_at > NOW() - INTERVAL ':days days'
            ORDER BY recorded_at DESC
            LIMIT 30
        """.replace(":days", str(history_days))), {"agent_id": agent_id})

        trend = [
            {
                "accuracy_score": h.accuracy_score,
                "verified_claims": h.verified_claims,
                "disputed_claims": h.disputed_claims,
                "recorded_at": h.recorded_at.isoformat(),
            }
            for h in history_result.fetchall()
        ]

        return AgentTrustMetrics(
            agent_id=agent_id,
            accuracy_score=row.accuracy_score,
            verified_claims=row.verified_claims,
            disputed_claims=row.disputed_claims,
            total_claims=row.total_claims,
            trust_trend=trend,
        )

    async def update_trust_from_claims(
        self,
        db: AsyncSession,
        agent_id: str,
    ) -> Optional[float]:
        """Update agent trust score based on claim verification.

        Called during dream cycle to recalculate trust metrics.

        Args:
            db: Database session
            agent_id: Agent ID to update

        Returns:
            New accuracy score or None
        """
        # Count verified vs disputed claims from last 30 days
        result = await db.execute(text("""
            SELECT
                COUNT(*) FILTER (
                    WHERE status = 'active'
                    AND NOT EXISTS (
                        SELECT 1 FROM claims c2
                        WHERE c2.superseded_by_id = c.claim_id
                    )
                ) as verified,
                COUNT(*) FILTER (
                    WHERE contradiction_group_id IS NOT NULL
                ) as disputed,
                COUNT(*) as total
            FROM claims c
            WHERE created_by_agent_id = :agent_id
            AND created_at > NOW() - INTERVAL '30 days'
        """), {"agent_id": agent_id})

        row = result.fetchone()
        if row is None or row.total == 0:
            return None

        verified = row.verified or 0
        disputed = row.disputed or 0
        total = row.total

        # Calculate accuracy: verified claims minus half disputed, centered at 0.5
        raw_accuracy = (verified - disputed * 0.5) / total
        accuracy = max(0.0, min(1.0, raw_accuracy * 0.5 + 0.5))

        # Update agent config
        await db.execute(text("""
            UPDATE agent_configs SET
                accuracy_score = :accuracy,
                verified_claims = :verified,
                disputed_claims = :disputed,
                total_claims = :total,
                updated_at = NOW()
            WHERE agent_id = :agent_id
        """), {
            "accuracy": accuracy,
            "verified": verified,
            "disputed": disputed,
            "total": total,
            "agent_id": agent_id,
        })

        # Record history
        await db.execute(text("""
            INSERT INTO agent_trust_history
            (agent_id, accuracy_score, verified_claims, disputed_claims)
            VALUES (:agent_id, :accuracy, :verified, :disputed)
        """), {
            "agent_id": agent_id,
            "accuracy": accuracy,
            "verified": verified,
            "disputed": disputed,
        })

        await db.commit()

        logger.info(f"Updated trust for agent {agent_id}: accuracy={accuracy:.3f}")
        return accuracy

    async def update_all_agents_trust(self, db: AsyncSession) -> int:
        """Update trust scores for all agents (called in dream cycle).

        Returns:
            Number of agents updated
        """
        # Get all active agents with claims
        result = await db.execute(text("""
            SELECT DISTINCT created_by_agent_id
            FROM claims
            WHERE created_by_agent_id IS NOT NULL
            AND created_at > NOW() - INTERVAL '30 days'
        """))

        updated = 0
        for row in result.fetchall():
            try:
                await self.update_trust_from_claims(db, row.created_by_agent_id)
                updated += 1
            except Exception as e:
                logger.error(f"Failed to update trust for {row.created_by_agent_id}: {e}")

        return updated


# =============================================================================
# TOKEN BUDGET MANAGEMENT
# =============================================================================

class TokenBudgetService:
    """Service for managing agent token budgets."""

    async def check_budget(
        self,
        db: AsyncSession,
        agent_id: str,
        requested_tokens: int,
    ) -> Tuple[bool, int]:
        """Check if agent has budget for requested tokens.

        Args:
            db: Database session
            agent_id: Agent ID
            requested_tokens: Tokens to use

        Returns:
            Tuple of (allowed, remaining_budget)
        """
        result = await db.execute(text("""
            SELECT token_budget_daily, tokens_used_today, budget_reset_at
            FROM agent_configs
            WHERE agent_id = :agent_id AND status = 'active'
        """), {"agent_id": agent_id})

        row = result.fetchone()
        if row is None:
            return False, 0

        # Check if budget needs reset
        if row.budget_reset_at is None or row.budget_reset_at < datetime.now():
            # Reset budget
            await db.execute(text("""
                UPDATE agent_configs SET
                    tokens_used_today = 0,
                    budget_reset_at = NOW() + INTERVAL '1 day'
                WHERE agent_id = :agent_id
            """), {"agent_id": agent_id})
            await db.commit()
            remaining = row.token_budget_daily
        else:
            remaining = row.token_budget_daily - row.tokens_used_today

        allowed = remaining >= requested_tokens
        return allowed, remaining

    async def consume_tokens(
        self,
        db: AsyncSession,
        agent_id: str,
        tokens_used: int,
    ) -> int:
        """Record token usage.

        Args:
            db: Database session
            agent_id: Agent ID
            tokens_used: Tokens consumed

        Returns:
            New total tokens used today
        """
        result = await db.execute(text("""
            UPDATE agent_configs SET
                tokens_used_today = tokens_used_today + :tokens
            WHERE agent_id = :agent_id
            RETURNING tokens_used_today
        """), {"agent_id": agent_id, "tokens": tokens_used})

        row = result.fetchone()
        await db.commit()

        return row.tokens_used_today if row else 0


# =============================================================================
# GLOBAL SERVICE INSTANCES
# =============================================================================

_agent_service: Optional[AgentService] = None
_trust_service: Optional[AgentTrustService] = None
_budget_service: Optional[TokenBudgetService] = None


def get_agent_service() -> AgentService:
    """Get global agent service instance."""
    global _agent_service
    if _agent_service is None:
        _agent_service = AgentService()
    return _agent_service


def get_trust_service() -> AgentTrustService:
    """Get global trust service instance."""
    global _trust_service
    if _trust_service is None:
        _trust_service = AgentTrustService()
    return _trust_service


def get_budget_service() -> TokenBudgetService:
    """Get global budget service instance."""
    global _budget_service
    if _budget_service is None:
        _budget_service = TokenBudgetService()
    return _budget_service


# =============================================================================
# PROVIDER INTEGRATION
# =============================================================================

async def get_llm_for_agent(
    db: AsyncSession,
    agent_id: Optional[str] = None,
    fallback_provider: Optional[str] = None,
    fallback_model: Optional[str] = None,
):
    """
    Get an LLM provider configured for a specific agent.

    Args:
        db: Database session
        agent_id: Agent ID to get config for (optional)
        fallback_provider: Provider to use if agent has none configured
        fallback_model: Model to use if agent has none configured

    Returns:
        Configured LLM provider instance

    Example:
        # Get provider for specific agent
        llm = await get_llm_for_agent(db, "agent_123")
        response = await llm.complete(LLMRequest(prompt="Hello"))

        # Use default provider
        llm = await get_llm_for_agent(db, fallback_provider="ollama")
    """
    from api.llm.providers import (
        LLMProvider,
        ProviderConfig,
        ProviderRegistry,
    )

    agent_config = None
    credentials = None

    if agent_id:
        service = get_agent_service()
        result = await service.get_agent(db, agent_id, include_credentials=True)
        if result:
            agent_config, credentials = result

    # Determine provider
    if agent_config and agent_config.llm_provider:
        provider = LLMProvider(agent_config.llm_provider)
    elif fallback_provider:
        provider = LLMProvider(fallback_provider)
    else:
        provider = None  # Use default

    # Build config
    config = ProviderConfig(provider=provider or LLMProvider.VLLM)

    if agent_config:
        config.base_url = agent_config.endpoint_url
        config.model = agent_config.default_model or fallback_model

    if credentials:
        config.api_key = credentials.api_key or credentials.bearer_token

    return ProviderRegistry.get_llm_provider(provider, config)


async def get_embedder_for_agent(
    db: AsyncSession,
    agent_id: Optional[str] = None,
    fallback_provider: Optional[str] = None,
    fallback_model: Optional[str] = None,
    fallback_device: Optional[str] = None,
):
    """
    Get an embedding provider configured for a specific agent.

    Args:
        db: Database session
        agent_id: Agent ID to get config for (optional)
        fallback_provider: Provider to use if agent has none configured
        fallback_model: Model to use if agent has none configured
        fallback_device: Device to use (auto, cpu, cuda)

    Returns:
        Configured embedding provider instance

    Example:
        # Get embedder for agent (uses their config)
        embedder = await get_embedder_for_agent(db, "agent_123")
        response = await embedder.embed(EmbeddingRequest(texts=["Hello"]))

        # Force CPU embedding
        embedder = await get_embedder_for_agent(db, fallback_device="cpu")
    """
    from api.llm.providers import (
        EmbeddingProvider,
        DeviceType,
        ProviderConfig,
        ProviderRegistry,
    )

    agent_config = None

    if agent_id:
        service = get_agent_service()
        result = await service.get_agent(db, agent_id)
        if result:
            agent_config = result[0] if isinstance(result, tuple) else result

    # Determine provider
    if agent_config and agent_config.embedding_provider:
        provider = EmbeddingProvider(agent_config.embedding_provider)
    elif fallback_provider:
        provider = EmbeddingProvider(fallback_provider)
    else:
        provider = EmbeddingProvider.SENTENCE_TRANSFORMERS

    # Determine device
    device = DeviceType.AUTO
    if agent_config and agent_config.embedding_device:
        device = DeviceType(agent_config.embedding_device)
    elif fallback_device:
        device = DeviceType(fallback_device)

    # Build config
    config = ProviderConfig(
        provider=provider,
        model=agent_config.embedding_model if agent_config else fallback_model,
        device=device,
    )

    return ProviderRegistry.get_embedding_provider(provider, config)


async def complete_for_agent(
    db: AsyncSession,
    prompt: str,
    agent_id: Optional[str] = None,
    system_prompt: Optional[str] = None,
    **kwargs,
):
    """
    Run LLM completion using agent's configured provider.

    Convenience function that handles provider lookup and completion.

    Args:
        db: Database session
        prompt: User prompt
        agent_id: Agent to use (optional, uses default if not provided)
        system_prompt: System prompt (overrides agent's default if provided)
        **kwargs: Additional LLM parameters

    Returns:
        LLMResponse with completion

    Example:
        response = await complete_for_agent(
            db,
            "Explain quantum computing",
            agent_id="agent_123",
        )
        print(response.content)
    """
    from api.llm.providers import LLMRequest

    llm = await get_llm_for_agent(db, agent_id)

    # Get agent's system prompt if not overridden
    if system_prompt is None and agent_id:
        service = get_agent_service()
        result = await service.get_agent(db, agent_id)
        if result:
            config = result[0] if isinstance(result, tuple) else result
            system_prompt = config.system_prompt_override

    request = LLMRequest(
        prompt=prompt,
        system_prompt=system_prompt,
        agent_id=agent_id,
        **kwargs,
    )

    return await llm.complete(request)


async def embed_for_agent(
    db: AsyncSession,
    texts: List[str],
    agent_id: Optional[str] = None,
    **kwargs,
):
    """
    Generate embeddings using agent's configured provider.

    Convenience function that handles provider lookup and embedding.

    Args:
        db: Database session
        texts: Texts to embed
        agent_id: Agent to use (optional, uses default if not provided)
        **kwargs: Additional embedding parameters

    Returns:
        EmbeddingResponse with vectors

    Example:
        response = await embed_for_agent(
            db,
            ["Hello world", "Goodbye world"],
            agent_id="agent_123",
        )
        vectors = response.embeddings
    """
    from api.llm.providers import EmbeddingRequest

    embedder = await get_embedder_for_agent(db, agent_id, **kwargs)

    request = EmbeddingRequest(
        texts=texts,
        agent_id=agent_id,
    )

    return await embedder.embed(request)
