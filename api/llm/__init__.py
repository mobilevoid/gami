"""
GAMI LLM Module.

Provides unified interface for LLM and embedding operations across multiple providers.

Supported LLM Providers:
- vLLM (local GPU inference)
- Ollama (local CPU/GPU)
- OpenAI API
- Anthropic API (Claude)

Supported Embedding Providers:
- sentence-transformers (local CPU/GPU, recommended)
- Ollama
- OpenAI

Usage:
    from api.llm import complete, embed

    # LLM completion with default provider
    response = await complete("What is 2+2?")

    # Specify provider and model
    response = await complete(
        "Explain quantum physics",
        provider="anthropic",
        model="claude-sonnet-4-20250514",
    )

    # Embedding with default (sentence-transformers)
    vectors = await embed(["Hello", "World"])

    # Force CPU embedding
    vectors = await embed(["Hello"], device="cpu")

    # Use OpenAI embeddings
    vectors = await embed(["Hello"], provider="openai")
"""

# Core client functions (backwards compatible)
from .client import (
    call_vllm,
    call_ollama,
    call_vllm_sync,
    call_ollama_sync,
    parse_json_from_llm,
)

# Embedding functions (backwards compatible)
from .embeddings import (
    embed_text_sync,
    embed_texts_sync,
    embed_texts_batch,
)

# New unified provider system
from .providers import (
    # Enums
    LLMProvider,
    EmbeddingProvider,
    DeviceType,
    # Data classes
    LLMRequest,
    LLMResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ProviderConfig,
    # Provider classes
    VLLMProvider,
    OllamaLLMProvider,
    OllamaEmbeddingProvider,
    OpenAILLMProvider,
    OpenAIEmbeddingProvider,
    AnthropicProvider,
    SentenceTransformersProvider,
    # Registry
    ProviderRegistry,
    # Convenience functions
    complete,
    embed,
    embed_text,
    embed_texts,
    complete_sync,
    embed_sync,
    embed_text_sync as embed_text_sync_new,
    embed_texts_sync as embed_texts_sync_new,
)

# MCP Integration
from .mcp_integration import (
    MCPTool,
    MCPToolResult,
    MCPClient,
    MCPEnabledLLM,
    DatabaseMCPTools,
    complete_with_mcp,
    get_gami_mcp_tools,
)

__all__ = [
    # Legacy client
    "call_vllm",
    "call_ollama",
    "call_vllm_sync",
    "call_ollama_sync",
    "parse_json_from_llm",
    # Legacy embeddings
    "embed_text_sync",
    "embed_texts_sync",
    "embed_texts_batch",
    # Enums
    "LLMProvider",
    "EmbeddingProvider",
    "DeviceType",
    # Data classes
    "LLMRequest",
    "LLMResponse",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "ProviderConfig",
    # Providers
    "VLLMProvider",
    "OllamaLLMProvider",
    "OllamaEmbeddingProvider",
    "OpenAILLMProvider",
    "OpenAIEmbeddingProvider",
    "AnthropicProvider",
    "SentenceTransformersProvider",
    "ProviderRegistry",
    # Convenience
    "complete",
    "embed",
    "embed_text",
    "embed_texts",
    "complete_sync",
    "embed_sync",
    # MCP
    "MCPTool",
    "MCPToolResult",
    "MCPClient",
    "MCPEnabledLLM",
    "DatabaseMCPTools",
    "complete_with_mcp",
    "get_gami_mcp_tools",
]
