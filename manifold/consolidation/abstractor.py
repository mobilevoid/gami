"""Abstraction generation for memory clusters.

Uses LLM to create concise abstractions from groups of related memories,
capturing the essential information while reducing redundancy.
"""

import logging
from typing import List, Optional

logger = logging.getLogger("gami.consolidation.abstractor")

# Default prompt for abstraction generation
ABSTRACTION_SYSTEM_PROMPT = """You summarize groups of related memories into a single coherent abstraction.
Your abstraction should:
1. Capture the essential information from all memories
2. Be more concise than the combined inputs
3. Preserve important facts, names, numbers, and technical details
4. Remove redundancy while maintaining completeness
5. Be written as a factual statement, not a meta-description

Return ONLY the abstraction text, no explanations or formatting."""

ABSTRACTION_USER_TEMPLATE = """Create a brief abstraction from these {count} related memories:

---
{texts}
---

Write a single concise paragraph that captures the essential information from all of these memories."""


async def generate_abstraction(
    texts: List[str],
    model: str = None,
    temperature: float = 0.3,
    max_tokens: int = 500,
) -> Optional[str]:
    """Generate abstraction from a cluster of texts using LLM.

    Args:
        texts: List of text strings to abstract
        model: Model to use (None = default)
        temperature: LLM temperature (lower = more focused)
        max_tokens: Maximum tokens in response

    Returns:
        Abstraction text or None if generation fails
    """
    if not texts:
        return None

    if len(texts) == 1:
        # Single text doesn't need abstraction
        return texts[0]

    # Truncate texts to avoid context overflow
    truncated_texts = []
    total_chars = 0
    max_chars = 8000  # ~2000 tokens for input

    for text in texts:
        if total_chars + len(text) > max_chars:
            remaining = max_chars - total_chars
            if remaining > 100:
                truncated_texts.append(text[:remaining] + "...")
            break
        truncated_texts.append(text)
        total_chars += len(text)

    combined = "\n---\n".join(truncated_texts)
    user_prompt = ABSTRACTION_USER_TEMPLATE.format(
        count=len(truncated_texts),
        texts=combined,
    )

    try:
        from api.llm import complete

        response = await complete(
            prompt=user_prompt,
            system_prompt=ABSTRACTION_SYSTEM_PROMPT,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
        )

        if response and response.strip():
            abstraction = response.strip()
            logger.debug(
                "Generated abstraction from %d texts (%d chars -> %d chars)",
                len(texts), total_chars, len(abstraction)
            )
            return abstraction

    except Exception as e:
        logger.error("Abstraction generation failed: %s", e)

    return None


def generate_abstraction_sync(
    texts: List[str],
    model: str = None,
    temperature: float = 0.3,
    max_tokens: int = 500,
) -> Optional[str]:
    """Synchronous wrapper for generate_abstraction."""
    import asyncio

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(
        generate_abstraction(texts, model, temperature, max_tokens)
    )


async def generate_abstractions_batch(
    clusters: List[List[str]],
    model: str = None,
    temperature: float = 0.3,
    max_tokens: int = 500,
    concurrency: int = 3,
) -> List[Optional[str]]:
    """Generate abstractions for multiple clusters with controlled concurrency.

    Args:
        clusters: List of text lists, one per cluster
        model: Model to use
        temperature: LLM temperature
        max_tokens: Max tokens per response
        concurrency: Max concurrent LLM calls

    Returns:
        List of abstractions (or None for failed clusters)
    """
    import asyncio

    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_generate(texts: List[str]) -> Optional[str]:
        async with semaphore:
            return await generate_abstraction(texts, model, temperature, max_tokens)

    tasks = [bounded_generate(texts) for texts in clusters]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Replace exceptions with None
    return [
        r if isinstance(r, str) or r is None else None
        for r in results
    ]
