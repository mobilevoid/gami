"""Authentication dependencies for GAMI API.

Provides optional API key authentication that can be enabled via environment variables:
- GAMI_API_KEY: The API key to validate against
- GAMI_REQUIRE_AUTH_FOR_AGENTS: Set to "true" to require auth for agent endpoints
"""

from typing import Optional
from fastapi import Header, HTTPException, status, Request
from functools import lru_cache
import secrets
import time
import logging

from api.config import settings

logger = logging.getLogger("gami.auth")

# Simple in-memory rate limiter
_rate_limit_cache: dict = {}
RATE_LIMIT_WINDOW = 60  # seconds
DEFAULT_RATE_LIMIT = 60  # requests per window


def _check_rate_limit(identifier: str, limit: int = DEFAULT_RATE_LIMIT) -> bool:
    """Check if identifier is within rate limit.

    Returns True if allowed, False if rate limited.
    """
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW

    # Clean old entries
    if identifier in _rate_limit_cache:
        _rate_limit_cache[identifier] = [
            t for t in _rate_limit_cache[identifier] if t > window_start
        ]
    else:
        _rate_limit_cache[identifier] = []

    # Check limit
    if len(_rate_limit_cache[identifier]) >= limit:
        return False

    # Record request
    _rate_limit_cache[identifier].append(now)
    return True


async def verify_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None),
) -> Optional[str]:
    """Verify API key if authentication is required.

    Accepts either:
    - X-API-Key header
    - Authorization: Bearer <key> header

    Returns the verified API key or None if auth is disabled.
    Raises HTTPException if auth is required but key is invalid.
    """
    # If no API key configured, auth is disabled
    if not settings.API_KEY:
        return None

    # If auth not required for agents, skip
    if not settings.REQUIRE_AUTH_FOR_AGENTS:
        return None

    # Extract key from headers
    api_key = None
    if x_api_key:
        api_key = x_api_key
    elif authorization and authorization.startswith("Bearer "):
        api_key = authorization[7:]

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required. Provide X-API-Key header or Authorization: Bearer <key>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Constant-time comparison to prevent timing attacks
    if not secrets.compare_digest(api_key, settings.API_KEY):
        logger.warning(f"Invalid API key attempt")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return api_key


async def rate_limit_check(
    request: Request,
    limit: int = DEFAULT_RATE_LIMIT,
) -> None:
    """Check rate limit for the request.

    Uses client IP as identifier. Raises HTTPException if rate limited.
    """
    # Get client IP
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        client_ip = forwarded.split(",")[0].strip()
    else:
        client_ip = request.client.host if request.client else "unknown"

    if not _check_rate_limit(client_ip, limit):
        logger.warning(f"Rate limit exceeded for {client_ip}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Max {limit} requests per minute.",
            headers={"Retry-After": str(RATE_LIMIT_WINDOW)},
        )


class RateLimiter:
    """Configurable rate limiter dependency.

    Usage:
        @router.post("/endpoint")
        async def endpoint(
            _: None = Depends(RateLimiter(requests_per_minute=30))
        ):
            ...
    """

    def __init__(self, requests_per_minute: int = DEFAULT_RATE_LIMIT):
        self.limit = requests_per_minute

    async def __call__(self, request: Request) -> None:
        await rate_limit_check(request, self.limit)
