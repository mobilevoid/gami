"""
Unified LLM and Embedding Provider System for GAMI.

Supports multiple backends:
- vLLM (local GPU inference)
- Ollama (local CPU/GPU)
- OpenAI API
- Anthropic API (Claude)
- Local sentence-transformers (CPU/GPU embeddings)

Each provider can be selected per-call or configured per-agent.
"""

import asyncio
import json
import logging
import os
import re
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

logger = logging.getLogger("gami.llm.providers")


# =============================================================================
# ENUMS AND CONFIGURATION
# =============================================================================

class LLMProvider(str, Enum):
    """Available LLM providers."""
    VLLM = "vllm"
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"  # For future local model support


class EmbeddingProvider(str, Enum):
    """Available embedding providers."""
    SENTENCE_TRANSFORMERS = "sentence_transformers"  # Local CPU/GPU
    OLLAMA = "ollama"
    OPENAI = "openai"
    VLLM = "vllm"


class DeviceType(str, Enum):
    """Compute device for local models."""
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon


@dataclass
class ProviderConfig:
    """Configuration for a provider."""
    provider: Union[LLMProvider, EmbeddingProvider]
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None
    device: DeviceType = DeviceType.AUTO
    timeout_seconds: float = 120.0
    max_retries: int = 3
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMRequest:
    """Request for LLM completion."""
    prompt: str
    system_prompt: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.1
    model: Optional[str] = None
    provider: Optional[LLMProvider] = None
    # For agent-specific config
    agent_id: Optional[str] = None
    # Additional provider-specific params
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """Response from LLM completion."""
    content: str
    model: str
    provider: LLMProvider
    usage: Optional[Dict[str, int]] = None
    latency_ms: float = 0.0
    raw_response: Optional[Dict] = None


@dataclass
class EmbeddingRequest:
    """Request for embedding generation."""
    texts: List[str]
    model: Optional[str] = None
    provider: Optional[EmbeddingProvider] = None
    device: Optional[DeviceType] = None
    # For agent-specific config
    agent_id: Optional[str] = None


@dataclass
class EmbeddingResponse:
    """Response from embedding generation."""
    embeddings: List[List[float]]
    model: str
    provider: EmbeddingProvider
    dimension: int
    latency_ms: float = 0.0
    device_used: Optional[str] = None


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

class ProviderDefaults:
    """Default configuration loaded from environment."""

    # LLM Providers
    VLLM_URL: str = os.getenv("GAMI_VLLM_URL", "http://localhost:8000/v1")
    OLLAMA_URL: str = os.getenv("GAMI_OLLAMA_URL", "http://localhost:11434")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_BASE_URL: str = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")

    # Default models per provider
    VLLM_MODEL: str = os.getenv("GAMI_VLLM_MODEL", "qwen35-27b-unredacted")
    OLLAMA_MODEL: str = os.getenv("GAMI_OLLAMA_MODEL", "qwen3:8b")
    OPENAI_MODEL: str = os.getenv("GAMI_OPENAI_MODEL", "gpt-4o")
    ANTHROPIC_MODEL: str = os.getenv("GAMI_ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

    # Embedding
    EMBEDDING_PROVIDER: str = os.getenv("GAMI_EMBEDDING_PROVIDER", "sentence_transformers")
    EMBEDDING_MODEL: str = os.getenv("GAMI_EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5")
    EMBEDDING_DEVICE: str = os.getenv("GAMI_EMBEDDING_DEVICE", "auto")
    EMBEDDING_DIM: int = int(os.getenv("GAMI_EMBEDDING_DIM", "768"))

    # Default provider
    DEFAULT_LLM_PROVIDER: str = os.getenv("GAMI_DEFAULT_LLM_PROVIDER", "vllm")


# =============================================================================
# BASE PROVIDER CLASSES
# =============================================================================

class BaseLLMProvider(ABC):
    """Base class for LLM providers."""

    provider_type: LLMProvider

    def __init__(self, config: ProviderConfig):
        self.config = config
        self.client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self.client is None:
            self.client = httpx.AsyncClient(timeout=self.config.timeout_seconds)
        return self.client

    async def close(self):
        if self.client:
            await self.client.aclose()
            self.client = None

    @abstractmethod
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Generate completion."""
        pass

    def _strip_thinking(self, text: str) -> str:
        """Remove Qwen3 <think>...</think> blocks."""
        if not text:
            return text
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        text = re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()
        return text


class BaseEmbeddingProvider(ABC):
    """Base class for embedding providers."""

    provider_type: EmbeddingProvider

    def __init__(self, config: ProviderConfig):
        self.config = config

    @abstractmethod
    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embeddings."""
        pass

    async def close(self):
        """Cleanup resources."""
        pass


# =============================================================================
# VLLM PROVIDER
# =============================================================================

class VLLMProvider(BaseLLMProvider):
    """vLLM provider (OpenAI-compatible API)."""

    provider_type = LLMProvider.VLLM

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException)),
    )
    async def complete(self, request: LLMRequest) -> LLMResponse:
        import time
        start = time.time()

        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        payload = {
            "model": request.model or self.config.model or ProviderDefaults.VLLM_MODEL,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            # Disable Qwen3 thinking mode for structured extraction
            "chat_template_kwargs": {"enable_thinking": False},
            **request.extra_params,
        }

        client = await self._get_client()
        base_url = self.config.base_url or ProviderDefaults.VLLM_URL

        resp = await client.post(
            f"{base_url}/chat/completions",
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

        content = self._strip_thinking(data["choices"][0]["message"]["content"])
        latency_ms = (time.time() - start) * 1000

        return LLMResponse(
            content=content,
            model=payload["model"],
            provider=self.provider_type,
            usage=data.get("usage"),
            latency_ms=latency_ms,
            raw_response=data,
        )


# =============================================================================
# OLLAMA PROVIDER
# =============================================================================

class OllamaLLMProvider(BaseLLMProvider):
    """Ollama provider for local models."""

    provider_type = LLMProvider.OLLAMA

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException)),
    )
    async def complete(self, request: LLMRequest) -> LLMResponse:
        import time
        start = time.time()

        payload = {
            "model": request.model or self.config.model or ProviderDefaults.OLLAMA_MODEL,
            "prompt": request.prompt,
            "stream": False,
            "options": {
                "num_predict": request.max_tokens,
                "temperature": request.temperature,
            },
            **request.extra_params,
        }
        if request.system_prompt:
            payload["system"] = request.system_prompt

        client = await self._get_client()
        base_url = self.config.base_url or ProviderDefaults.OLLAMA_URL

        resp = await client.post(
            f"{base_url}/api/generate",
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

        content = self._strip_thinking(data["response"])
        latency_ms = (time.time() - start) * 1000

        return LLMResponse(
            content=content,
            model=payload["model"],
            provider=self.provider_type,
            usage={
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
            },
            latency_ms=latency_ms,
            raw_response=data,
        )


class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    """Ollama embedding provider."""

    provider_type = EmbeddingProvider.OLLAMA

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self.client is None:
            self.client = httpx.AsyncClient(timeout=self.config.timeout_seconds)
        return self.client

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        import time
        start = time.time()

        model = request.model or self.config.model or "nomic-embed-text"
        base_url = self.config.base_url or ProviderDefaults.OLLAMA_URL
        client = await self._get_client()

        embeddings = []
        for text in request.texts:
            resp = await client.post(
                f"{base_url}/api/embeddings",
                json={"model": model, "prompt": text[:2000]},  # Truncate
            )
            resp.raise_for_status()
            data = resp.json()
            embeddings.append(data["embedding"])

        latency_ms = (time.time() - start) * 1000

        return EmbeddingResponse(
            embeddings=embeddings,
            model=model,
            provider=self.provider_type,
            dimension=len(embeddings[0]) if embeddings else 768,
            latency_ms=latency_ms,
        )

    async def close(self):
        if self.client:
            await self.client.aclose()
            self.client = None


# =============================================================================
# OPENAI PROVIDER
# =============================================================================

class OpenAILLMProvider(BaseLLMProvider):
    """OpenAI API provider."""

    provider_type = LLMProvider.OPENAI

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException)),
    )
    async def complete(self, request: LLMRequest) -> LLMResponse:
        import time
        start = time.time()

        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        payload = {
            "model": request.model or self.config.model or ProviderDefaults.OPENAI_MODEL,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            **request.extra_params,
        }

        client = await self._get_client()
        base_url = self.config.base_url or ProviderDefaults.OPENAI_BASE_URL
        api_key = self.config.api_key or ProviderDefaults.OPENAI_API_KEY

        if not api_key:
            raise ValueError("OpenAI API key not configured. Set OPENAI_API_KEY env var.")

        resp = await client.post(
            f"{base_url}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        resp.raise_for_status()
        data = resp.json()

        content = data["choices"][0]["message"]["content"]
        latency_ms = (time.time() - start) * 1000

        return LLMResponse(
            content=content,
            model=payload["model"],
            provider=self.provider_type,
            usage=data.get("usage"),
            latency_ms=latency_ms,
            raw_response=data,
        )


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI embedding provider."""

    provider_type = EmbeddingProvider.OPENAI

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self.client is None:
            self.client = httpx.AsyncClient(timeout=self.config.timeout_seconds)
        return self.client

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        import time
        start = time.time()

        model = request.model or self.config.model or "text-embedding-3-small"
        base_url = self.config.base_url or ProviderDefaults.OPENAI_BASE_URL
        api_key = self.config.api_key or ProviderDefaults.OPENAI_API_KEY

        if not api_key:
            raise ValueError("OpenAI API key not configured. Set OPENAI_API_KEY env var.")

        client = await self._get_client()

        resp = await client.post(
            f"{base_url}/embeddings",
            json={
                "model": model,
                "input": [t[:8000] for t in request.texts],  # OpenAI limit
            },
            headers={"Authorization": f"Bearer {api_key}"},
        )
        resp.raise_for_status()
        data = resp.json()

        embeddings = [d["embedding"] for d in data["data"]]
        latency_ms = (time.time() - start) * 1000

        return EmbeddingResponse(
            embeddings=embeddings,
            model=model,
            provider=self.provider_type,
            dimension=len(embeddings[0]) if embeddings else 1536,
            latency_ms=latency_ms,
        )

    async def close(self):
        if self.client:
            await self.client.aclose()
            self.client = None


# =============================================================================
# ANTHROPIC PROVIDER
# =============================================================================

class AnthropicProvider(BaseLLMProvider):
    """Anthropic API provider (Claude)."""

    provider_type = LLMProvider.ANTHROPIC

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException)),
    )
    async def complete(self, request: LLMRequest) -> LLMResponse:
        import time
        start = time.time()

        messages = [{"role": "user", "content": request.prompt}]

        payload = {
            "model": request.model or self.config.model or ProviderDefaults.ANTHROPIC_MODEL,
            "max_tokens": request.max_tokens,
            "messages": messages,
            **request.extra_params,
        }
        if request.system_prompt:
            payload["system"] = request.system_prompt

        client = await self._get_client()
        base_url = self.config.base_url or ProviderDefaults.ANTHROPIC_BASE_URL
        api_key = self.config.api_key or ProviderDefaults.ANTHROPIC_API_KEY

        if not api_key:
            raise ValueError("Anthropic API key not configured. Set ANTHROPIC_API_KEY env var.")

        resp = await client.post(
            f"{base_url}/v1/messages",
            json=payload,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
        )
        resp.raise_for_status()
        data = resp.json()

        # Extract text from content blocks
        content = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                content += block.get("text", "")

        latency_ms = (time.time() - start) * 1000

        return LLMResponse(
            content=content,
            model=payload["model"],
            provider=self.provider_type,
            usage=data.get("usage"),
            latency_ms=latency_ms,
            raw_response=data,
        )


# =============================================================================
# SENTENCE TRANSFORMERS PROVIDER (LOCAL CPU/GPU)
# =============================================================================

class SentenceTransformersProvider(BaseEmbeddingProvider):
    """Local sentence-transformers provider with CPU/GPU support."""

    provider_type = EmbeddingProvider.SENTENCE_TRANSFORMERS

    # Class-level model cache
    _models: Dict[Tuple[str, str], Any] = {}
    _torch_available: bool = False
    _st_available: bool = False

    @classmethod
    def _check_availability(cls):
        """Check if sentence-transformers is available."""
        if not hasattr(cls, "_checked"):
            try:
                import torch
                from sentence_transformers import SentenceTransformer
                cls._torch_available = True
                cls._st_available = True
            except ImportError:
                cls._torch_available = False
                cls._st_available = False
            cls._checked = True

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._check_availability()
        self._subprocess_python = self._find_embed_python()

    def _find_embed_python(self) -> Optional[str]:
        """Find the gami-embed conda environment python."""
        candidates = [
            "python3",
            os.path.expanduser("~/.conda/envs/gami-embed/bin/python"),
            "/opt/gami/.venv/bin/python",
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        return None

    def _get_device(self, request: EmbeddingRequest) -> str:
        """Determine compute device."""
        device_pref = request.device or self.config.device

        if device_pref == DeviceType.AUTO:
            if self._torch_available:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    return "mps"
            return "cpu"

        return device_pref.value

    def _get_model(self, model_name: str, device: str):
        """Get or load model (cached)."""
        cache_key = (model_name, device)

        if cache_key not in self._models:
            if not self._st_available:
                raise RuntimeError(
                    "sentence-transformers not available. Install with:\n"
                    "  pip install sentence-transformers torch"
                )

            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model {model_name} on {device}")
            self._models[cache_key] = SentenceTransformer(
                model_name,
                trust_remote_code=True,
                device=device,
            )

        return self._models[cache_key]

    def _embed_direct(self, texts: List[str], model_name: str, device: str) -> List[List[float]]:
        """Embed using direct model loading."""
        model = self._get_model(model_name, device)
        truncated = [t[:2000] for t in texts]
        embs = model.encode(truncated, normalize_embeddings=False, show_progress_bar=False)
        return embs.tolist()

    def _embed_subprocess(self, texts: List[str], model_name: str) -> List[List[float]]:
        """Embed via subprocess to gami-embed environment."""
        if not self._subprocess_python:
            raise RuntimeError(
                "No Python environment with sentence-transformers found.\n"
                "Install with: pip install sentence-transformers torch"
            )

        script = f"""
import sys, json
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('{model_name}', trust_remote_code=True)
texts = json.loads(sys.stdin.read())
embs = model.encode([t[:2000] for t in texts], normalize_embeddings=False, show_progress_bar=False)
print(json.dumps(embs.tolist()))
"""

        result = subprocess.run(
            [self._subprocess_python, "-c", script],
            input=json.dumps(texts),
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Embedding subprocess failed: {result.stderr[:500]}")

        return json.loads(result.stdout)

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        import time
        start = time.time()

        model_name = request.model or self.config.model or ProviderDefaults.EMBEDDING_MODEL
        device = self._get_device(request)

        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()

        if self._st_available:
            embeddings = await loop.run_in_executor(
                None, self._embed_direct, request.texts, model_name, device
            )
        else:
            embeddings = await loop.run_in_executor(
                None, self._embed_subprocess, request.texts, model_name
            )

        latency_ms = (time.time() - start) * 1000

        return EmbeddingResponse(
            embeddings=embeddings,
            model=model_name,
            provider=self.provider_type,
            dimension=len(embeddings[0]) if embeddings else 768,
            latency_ms=latency_ms,
            device_used=device,
        )


# =============================================================================
# PROVIDER REGISTRY
# =============================================================================

class ProviderRegistry:
    """Registry for managing providers."""

    _llm_providers: Dict[LLMProvider, type] = {
        LLMProvider.VLLM: VLLMProvider,
        LLMProvider.OLLAMA: OllamaLLMProvider,
        LLMProvider.OPENAI: OpenAILLMProvider,
        LLMProvider.ANTHROPIC: AnthropicProvider,
    }

    _embedding_providers: Dict[EmbeddingProvider, type] = {
        EmbeddingProvider.SENTENCE_TRANSFORMERS: SentenceTransformersProvider,
        EmbeddingProvider.OLLAMA: OllamaEmbeddingProvider,
        EmbeddingProvider.OPENAI: OpenAIEmbeddingProvider,
    }

    # Instance cache
    _llm_instances: Dict[str, BaseLLMProvider] = {}
    _embedding_instances: Dict[str, BaseEmbeddingProvider] = {}

    @classmethod
    def get_llm_provider(
        cls,
        provider: Optional[LLMProvider] = None,
        config: Optional[ProviderConfig] = None,
    ) -> BaseLLMProvider:
        """Get or create LLM provider instance."""
        if provider is None:
            provider = LLMProvider(ProviderDefaults.DEFAULT_LLM_PROVIDER)

        cache_key = f"{provider.value}:{config.base_url if config else 'default'}"

        if cache_key not in cls._llm_instances:
            provider_class = cls._llm_providers.get(provider)
            if provider_class is None:
                raise ValueError(f"Unknown LLM provider: {provider}")

            if config is None:
                config = ProviderConfig(provider=provider)

            cls._llm_instances[cache_key] = provider_class(config)

        return cls._llm_instances[cache_key]

    @classmethod
    def get_embedding_provider(
        cls,
        provider: Optional[EmbeddingProvider] = None,
        config: Optional[ProviderConfig] = None,
    ) -> BaseEmbeddingProvider:
        """Get or create embedding provider instance."""
        if provider is None:
            provider = EmbeddingProvider(ProviderDefaults.EMBEDDING_PROVIDER)

        cache_key = f"{provider.value}:{config.model if config else 'default'}"

        if cache_key not in cls._embedding_instances:
            provider_class = cls._embedding_providers.get(provider)
            if provider_class is None:
                raise ValueError(f"Unknown embedding provider: {provider}")

            if config is None:
                config = ProviderConfig(provider=provider)

            cls._embedding_instances[cache_key] = provider_class(config)

        return cls._embedding_instances[cache_key]

    @classmethod
    async def close_all(cls):
        """Close all provider instances."""
        for instance in cls._llm_instances.values():
            await instance.close()
        for instance in cls._embedding_instances.values():
            await instance.close()
        cls._llm_instances.clear()
        cls._embedding_instances.clear()


# =============================================================================
# UNIFIED API FUNCTIONS
# =============================================================================

async def complete(
    prompt: str,
    system_prompt: Optional[str] = None,
    max_tokens: int = 4096,
    temperature: float = 0.1,
    model: Optional[str] = None,
    provider: Optional[Union[LLMProvider, str]] = None,
    agent_id: Optional[str] = None,
    **kwargs,
) -> LLMResponse:
    """
    Generate LLM completion with configurable provider and model.

    Args:
        prompt: User prompt
        system_prompt: Optional system prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        model: Model name (provider-specific)
        provider: LLM provider (vllm, ollama, openai, anthropic)
        agent_id: Agent ID for per-agent config lookup
        **kwargs: Additional provider-specific parameters

    Returns:
        LLMResponse with generated content

    Examples:
        # Use default provider (vLLM)
        response = await complete("What is 2+2?")

        # Use specific provider and model
        response = await complete(
            "Explain quantum computing",
            provider="anthropic",
            model="claude-sonnet-4-20250514",
        )

        # Use local Ollama
        response = await complete(
            "List prime numbers",
            provider="ollama",
            model="llama3:8b",
        )
    """
    if isinstance(provider, str):
        provider = LLMProvider(provider)

    # For agent-specific config with DB lookup, use complete_for_agent() from agent_service
    llm_provider = ProviderRegistry.get_llm_provider(provider)

    request = LLMRequest(
        prompt=prompt,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        model=model,
        provider=provider,
        agent_id=agent_id,
        extra_params=kwargs,
    )

    return await llm_provider.complete(request)


async def embed(
    texts: Union[str, List[str]],
    model: Optional[str] = None,
    provider: Optional[Union[EmbeddingProvider, str]] = None,
    device: Optional[Union[DeviceType, str]] = None,
    agent_id: Optional[str] = None,
) -> EmbeddingResponse:
    """
    Generate embeddings with configurable provider, model, and device.

    Args:
        texts: Single text or list of texts to embed
        model: Model name (provider-specific)
        provider: Embedding provider (sentence_transformers, ollama, openai)
        device: Compute device (auto, cpu, cuda, mps)
        agent_id: Agent ID for per-agent config lookup

    Returns:
        EmbeddingResponse with embedding vectors

    Examples:
        # Use default (sentence-transformers on auto device)
        response = await embed("Hello world")

        # Force CPU
        response = await embed(
            ["Text 1", "Text 2"],
            device="cpu",
        )

        # Use OpenAI embeddings
        response = await embed(
            "Important document",
            provider="openai",
            model="text-embedding-3-large",
        )
    """
    if isinstance(texts, str):
        texts = [texts]

    if isinstance(provider, str):
        provider = EmbeddingProvider(provider)

    if isinstance(device, str):
        device = DeviceType(device)

    # For agent-specific config with DB lookup, use embed_for_agent() from agent_service
    embed_provider = ProviderRegistry.get_embedding_provider(provider)

    request = EmbeddingRequest(
        texts=texts,
        model=model,
        provider=provider,
        device=device,
        agent_id=agent_id,
    )

    return await embed_provider.embed(request)


# Convenience functions for backward compatibility
async def embed_text(text: str, **kwargs) -> List[float]:
    """Embed single text, return vector."""
    response = await embed(text, **kwargs)
    return response.embeddings[0]


async def embed_texts(texts: List[str], **kwargs) -> List[List[float]]:
    """Embed multiple texts, return vectors."""
    response = await embed(texts, **kwargs)
    return response.embeddings


# =============================================================================
# SYNC WRAPPERS (for Celery workers)
# =============================================================================

def complete_sync(
    prompt: str,
    system_prompt: Optional[str] = None,
    max_tokens: int = 4096,
    temperature: float = 0.1,
    model: Optional[str] = None,
    provider: Optional[Union[LLMProvider, str]] = None,
    **kwargs,
) -> LLMResponse:
    """Synchronous wrapper for complete()."""
    return asyncio.run(complete(
        prompt=prompt,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        model=model,
        provider=provider,
        **kwargs,
    ))


def embed_sync(
    texts: Union[str, List[str]],
    model: Optional[str] = None,
    provider: Optional[Union[EmbeddingProvider, str]] = None,
    device: Optional[Union[DeviceType, str]] = None,
) -> EmbeddingResponse:
    """Synchronous wrapper for embed()."""
    return asyncio.run(embed(
        texts=texts,
        model=model,
        provider=provider,
        device=device,
    ))


def embed_text_sync(text: str, **kwargs) -> List[float]:
    """Synchronous wrapper for embed_text()."""
    return asyncio.run(embed_text(text, **kwargs))


def embed_texts_sync(texts: List[str], **kwargs) -> List[List[float]]:
    """Synchronous wrapper for embed_texts()."""
    return asyncio.run(embed_texts(texts, **kwargs))
