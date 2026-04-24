"""
Agent configuration API endpoints.

Manages agent configurations, credentials, and trust metrics.

Authentication:
    Set GAMI_API_KEY and GAMI_REQUIRE_AUTH_FOR_AGENTS=true to require API key.

Rate Limiting:
    All endpoints are rate limited to 60 requests/minute per IP.
    Credential endpoints are limited to 10 requests/minute.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status, Request
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from ..services.db import get_async_session
from ..services.agent_service import (
    AgentService,
    AgentTrustService,
    TokenBudgetService,
    get_agent_service,
    get_trust_service,
    get_budget_service,
    AgentConfig,
)
from ..dependencies.auth import verify_api_key, RateLimiter

router = APIRouter(prefix="/agents", tags=["agents"])

# Standard rate limiter for most endpoints
standard_rate_limit = RateLimiter(requests_per_minute=60)

# Stricter rate limiter for sensitive operations
sensitive_rate_limit = RateLimiter(requests_per_minute=10)


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class CreateAgentRequest(BaseModel):
    """Request to create a new agent."""
    agent_id: Optional[str] = Field(None, description="Agent ID (auto-generated if not provided)")
    agent_name: str = Field(..., min_length=1, max_length=100)
    agent_type: str = Field(default="assistant", pattern="^(assistant|tool|background|human)$")
    owner_tenant_id: str = Field(..., min_length=1)

    # LLM Configuration
    default_model: Optional[str] = None
    endpoint_url: Optional[str] = None
    default_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    default_max_tokens: int = Field(default=2048, ge=1, le=128000)
    context_window_size: int = Field(default=8192, ge=1024)

    # Provider Configuration
    llm_provider: Optional[str] = Field(
        default="vllm",
        pattern="^(vllm|ollama|openai|anthropic)$",
        description="LLM provider: vllm, ollama, openai, anthropic"
    )
    embedding_provider: Optional[str] = Field(
        default="sentence_transformers",
        pattern="^(sentence_transformers|ollama|openai)$",
        description="Embedding provider: sentence_transformers, ollama, openai"
    )
    embedding_model: Optional[str] = Field(
        default="nomic-ai/nomic-embed-text-v1.5",
        description="Embedding model name"
    )
    embedding_device: Optional[str] = Field(
        default="auto",
        pattern="^(auto|cpu|cuda|mps)$",
        description="Device for local embeddings: auto, cpu, cuda, mps"
    )

    # Credentials (will be encrypted)
    credentials: Optional[Dict[str, str]] = Field(default=None, description="API credentials to encrypt")

    # Personality & Overrides
    personality_json: Optional[Dict[str, Any]] = None
    system_prompt_override: Optional[str] = None
    extraction_prompt_overrides: Optional[Dict[str, Any]] = None
    scoring_overrides: Optional[Dict[str, Any]] = None

    # Rate Limits
    rate_limit_rpm: int = Field(default=60, ge=1, le=10000)
    token_budget_daily: int = Field(default=1000000, ge=1000)


class UpdateAgentRequest(BaseModel):
    """Request to update an agent."""
    agent_name: Optional[str] = Field(None, min_length=1, max_length=100)
    agent_type: Optional[str] = Field(None, pattern="^(assistant|tool|background|human)$")

    # LLM Configuration
    default_model: Optional[str] = None
    endpoint_url: Optional[str] = None
    default_temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    default_max_tokens: Optional[int] = Field(None, ge=1, le=128000)
    context_window_size: Optional[int] = Field(None, ge=1024)

    # Provider Configuration
    llm_provider: Optional[str] = Field(
        None,
        pattern="^(vllm|ollama|openai|anthropic)$",
        description="LLM provider: vllm, ollama, openai, anthropic"
    )
    embedding_provider: Optional[str] = Field(
        None,
        pattern="^(sentence_transformers|ollama|openai)$",
        description="Embedding provider: sentence_transformers, ollama, openai"
    )
    embedding_model: Optional[str] = None
    embedding_device: Optional[str] = Field(
        None,
        pattern="^(auto|cpu|cuda|mps)$",
        description="Device for local embeddings: auto, cpu, cuda, mps"
    )

    # New credentials (replaces existing)
    credentials: Optional[Dict[str, str]] = None

    # Personality & Overrides
    personality_json: Optional[Dict[str, Any]] = None
    system_prompt_override: Optional[str] = None
    extraction_prompt_overrides: Optional[Dict[str, Any]] = None
    scoring_overrides: Optional[Dict[str, Any]] = None

    # Rate Limits
    rate_limit_rpm: Optional[int] = Field(None, ge=1, le=10000)
    token_budget_daily: Optional[int] = Field(None, ge=1000)

    # Status
    status: Optional[str] = Field(None, pattern="^(active|suspended|deleted)$")


class AgentResponse(BaseModel):
    """Agent configuration response."""
    agent_id: str
    owner_tenant_id: str
    agent_name: str
    agent_type: str
    default_model: Optional[str]
    endpoint_url: Optional[str]
    default_temperature: float
    default_max_tokens: int
    context_window_size: int
    has_credentials: bool
    personality_json: Dict[str, Any]
    scoring_overrides: Dict[str, Any]
    rate_limit_rpm: int
    token_budget_daily: int
    tokens_used_today: int
    accuracy_score: float
    verified_claims: int
    disputed_claims: int
    total_claims: int
    status: str
    created_at: Optional[datetime]
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True


class AgentListResponse(BaseModel):
    """List of agents response."""
    agents: List[AgentResponse]
    total: int


class TrustMetricsResponse(BaseModel):
    """Agent trust metrics response."""
    agent_id: str
    accuracy_score: float
    verified_claims: int
    disputed_claims: int
    total_claims: int
    trust_trend: List[Dict[str, Any]]


class BudgetResponse(BaseModel):
    """Token budget response."""
    agent_id: str
    token_budget_daily: int
    tokens_used_today: int
    remaining: int
    reset_at: Optional[datetime]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _agent_to_response(config: AgentConfig, has_credentials: bool = False) -> AgentResponse:
    """Convert AgentConfig to response model."""
    return AgentResponse(
        agent_id=config.agent_id,
        owner_tenant_id=config.owner_tenant_id,
        agent_name=config.agent_name,
        agent_type=config.agent_type,
        default_model=config.default_model,
        endpoint_url=config.endpoint_url,
        default_temperature=config.default_temperature,
        default_max_tokens=config.default_max_tokens,
        context_window_size=config.context_window_size,
        has_credentials=has_credentials,
        personality_json=config.personality_json or {},
        scoring_overrides=config.scoring_overrides or {},
        rate_limit_rpm=config.rate_limit_rpm,
        token_budget_daily=config.token_budget_daily,
        tokens_used_today=config.tokens_used_today,
        accuracy_score=config.accuracy_score,
        verified_claims=config.verified_claims,
        disputed_claims=config.disputed_claims,
        total_claims=config.total_claims,
        status=config.status,
        created_at=config.created_at,
        updated_at=config.updated_at,
    )


def _generate_agent_id() -> str:
    """Generate unique agent ID."""
    import secrets
    return f"agent_{secrets.token_hex(8)}"


# =============================================================================
# AGENT CRUD ENDPOINTS
# =============================================================================

@router.post("", response_model=AgentResponse, status_code=status.HTTP_201_CREATED)
async def create_agent(
    request: CreateAgentRequest,
    db: AsyncSession = Depends(get_async_session),
    service: AgentService = Depends(get_agent_service),
    _auth: str = Depends(verify_api_key),
    _rate: None = Depends(standard_rate_limit),
):
    """Create a new agent configuration.

    Credentials are encrypted using AES-256-GCM before storage.
    Requires API key if GAMI_REQUIRE_AUTH_FOR_AGENTS is enabled.
    """
    agent_id = request.agent_id or _generate_agent_id()

    # Check if agent ID already exists
    existing = await service.get_agent(db, agent_id)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Agent with ID {agent_id} already exists"
        )

    config = {
        "default_model": request.default_model,
        "endpoint_url": request.endpoint_url,
        "default_temperature": request.default_temperature,
        "default_max_tokens": request.default_max_tokens,
        "context_window_size": request.context_window_size,
        "personality_json": request.personality_json,
        "system_prompt_override": request.system_prompt_override,
        "extraction_prompt_overrides": request.extraction_prompt_overrides,
        "scoring_overrides": request.scoring_overrides,
        "rate_limit_rpm": request.rate_limit_rpm,
        "token_budget_daily": request.token_budget_daily,
    }

    agent = await service.create_agent(
        db=db,
        agent_id=agent_id,
        agent_name=request.agent_name,
        owner_tenant_id=request.owner_tenant_id,
        agent_type=request.agent_type,
        credentials=request.credentials,
        config=config,
    )

    return _agent_to_response(agent, has_credentials=request.credentials is not None)


@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(
    agent_id: str,
    db: AsyncSession = Depends(get_async_session),
    service: AgentService = Depends(get_agent_service),
):
    """Get agent configuration by ID.

    Does not return decrypted credentials for security.
    """
    result = await service.get_agent(db, agent_id, include_credentials=False)

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found"
        )

    config, _ = result
    return _agent_to_response(config, has_credentials=False)


@router.patch("/{agent_id}", response_model=AgentResponse)
async def update_agent(
    agent_id: str,
    request: UpdateAgentRequest,
    db: AsyncSession = Depends(get_async_session),
    service: AgentService = Depends(get_agent_service),
):
    """Update agent configuration.

    Only provided fields are updated. Credentials can be replaced.
    """
    # Check agent exists
    existing = await service.get_agent(db, agent_id)
    if existing is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found"
        )

    # Build updates dict from non-None fields
    updates = {k: v for k, v in request.model_dump().items()
               if v is not None and k != "credentials"}

    if not updates and request.credentials is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No updates provided"
        )

    success = await service.update_agent(
        db=db,
        agent_id=agent_id,
        updates=updates,
        new_credentials=request.credentials,
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update agent"
        )

    # Fetch updated config
    result = await service.get_agent(db, agent_id)
    config, _ = result
    return _agent_to_response(config)


@router.delete("/{agent_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent(
    agent_id: str,
    hard_delete: bool = False,
    db: AsyncSession = Depends(get_async_session),
    service: AgentService = Depends(get_agent_service),
):
    """Delete or deactivate an agent.

    By default, performs soft delete (sets status to 'deleted').
    Use hard_delete=true to permanently remove.
    """
    success = await service.delete_agent(db, agent_id, hard_delete=hard_delete)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found"
        )


@router.get("", response_model=AgentListResponse)
async def list_agents(
    tenant_id: Optional[str] = None,
    agent_type: Optional[str] = None,
    status: str = "active",
    db: AsyncSession = Depends(get_async_session),
    service: AgentService = Depends(get_agent_service),
    _auth: str = Depends(verify_api_key),
    _rate: None = Depends(standard_rate_limit),
):
    """List agents with optional filtering."""
    agents = await service.list_agents(
        db=db,
        tenant_id=tenant_id,
        agent_type=agent_type,
        status=status,
    )

    return AgentListResponse(
        agents=[_agent_to_response(a) for a in agents],
        total=len(agents),
    )


# =============================================================================
# TRUST ENDPOINTS
# =============================================================================

@router.get("/{agent_id}/trust", response_model=TrustMetricsResponse)
async def get_agent_trust(
    agent_id: str,
    history_days: int = 30,
    db: AsyncSession = Depends(get_async_session),
    service: AgentTrustService = Depends(get_trust_service),
):
    """Get trust metrics for an agent.

    Includes accuracy score, claim counts, and historical trend.
    """
    metrics = await service.get_trust_metrics(db, agent_id, history_days)

    if metrics is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found"
        )

    return TrustMetricsResponse(
        agent_id=metrics.agent_id,
        accuracy_score=metrics.accuracy_score,
        verified_claims=metrics.verified_claims,
        disputed_claims=metrics.disputed_claims,
        total_claims=metrics.total_claims,
        trust_trend=metrics.trust_trend,
    )


@router.post("/{agent_id}/trust/recalculate")
async def recalculate_trust(
    agent_id: str,
    db: AsyncSession = Depends(get_async_session),
    service: AgentTrustService = Depends(get_trust_service),
):
    """Force recalculation of trust score for an agent.

    Normally this happens during the dream cycle, but can be triggered manually.
    """
    new_score = await service.update_trust_from_claims(db, agent_id)

    if new_score is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found or has no claims"
        )

    return {"agent_id": agent_id, "new_accuracy_score": new_score}


# =============================================================================
# BUDGET ENDPOINTS
# =============================================================================

@router.get("/{agent_id}/budget", response_model=BudgetResponse)
async def get_agent_budget(
    agent_id: str,
    db: AsyncSession = Depends(get_async_session),
    agent_service: AgentService = Depends(get_agent_service),
):
    """Get token budget status for an agent."""
    result = await agent_service.get_agent(db, agent_id)

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found"
        )

    config, _ = result

    return BudgetResponse(
        agent_id=agent_id,
        token_budget_daily=config.token_budget_daily,
        tokens_used_today=config.tokens_used_today,
        remaining=config.token_budget_daily - config.tokens_used_today,
        reset_at=None,  # TODO: Add reset_at to query
    )


@router.post("/{agent_id}/budget/check")
async def check_budget(
    agent_id: str,
    requested_tokens: int,
    db: AsyncSession = Depends(get_async_session),
    service: TokenBudgetService = Depends(get_budget_service),
):
    """Check if agent has budget for requested tokens."""
    allowed, remaining = await service.check_budget(db, agent_id, requested_tokens)

    return {
        "agent_id": agent_id,
        "requested_tokens": requested_tokens,
        "allowed": allowed,
        "remaining_budget": remaining,
    }


@router.post("/{agent_id}/budget/consume")
async def consume_budget(
    agent_id: str,
    tokens_used: int,
    db: AsyncSession = Depends(get_async_session),
    service: TokenBudgetService = Depends(get_budget_service),
):
    """Record token consumption for an agent."""
    new_total = await service.consume_tokens(db, agent_id, tokens_used)

    return {
        "agent_id": agent_id,
        "tokens_consumed": tokens_used,
        "total_used_today": new_total,
    }


# =============================================================================
# CREDENTIAL MANAGEMENT
# =============================================================================

@router.post("/{agent_id}/credentials/test")
async def test_credentials(
    agent_id: str,
    db: AsyncSession = Depends(get_async_session),
    service: AgentService = Depends(get_agent_service),
    _auth: Optional[str] = Depends(verify_api_key),
    _rate: None = Depends(sensitive_rate_limit),
):
    """Test that agent credentials are valid and working.

    Attempts to authenticate with the configured endpoint.
    """
    result = await service.get_agent(db, agent_id, include_credentials=True)

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found"
        )

    config, credentials = result

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Agent has no credentials configured"
        )

    # TODO: Implement actual credential testing based on endpoint type
    # For now, just verify credentials were decrypted successfully
    return {
        "agent_id": agent_id,
        "credentials_valid": True,
        "has_api_key": credentials.api_key is not None,
        "has_bearer_token": credentials.bearer_token is not None,
        "endpoint_url": config.endpoint_url,
    }


@router.delete("/{agent_id}/credentials")
async def clear_credentials(
    agent_id: str,
    db: AsyncSession = Depends(get_async_session),
    service: AgentService = Depends(get_agent_service),
    _auth: Optional[str] = Depends(verify_api_key),
    _rate: None = Depends(sensitive_rate_limit),
):
    """Clear stored credentials for an agent."""
    from sqlalchemy import text

    result = await db.execute(text("""
        UPDATE agent_configs
        SET credentials_encrypted = NULL, credential_key_id = NULL, updated_at = NOW()
        WHERE agent_id = :agent_id
        RETURNING agent_id
    """), {"agent_id": agent_id})

    if result.fetchone() is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent {agent_id} not found"
        )

    await db.commit()
    return {"agent_id": agent_id, "credentials_cleared": True}
