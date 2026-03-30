"""LLM client for GAMI — async and sync wrappers for vLLM and Ollama."""
import json
import logging
import re
from typing import Optional

import httpx

from api.config import settings

logger = logging.getLogger("gami.llm.client")


# ---------------------------------------------------------------------------
# JSON parsing helper
# ---------------------------------------------------------------------------

def parse_json_from_llm(text: str) -> list | dict | None:
    """Extract JSON from an LLM response.

    Handles:
    - Bare JSON
    - ```json ... ``` fenced blocks
    - ``` ... ``` fenced blocks
    - Partial / truncated JSON arrays (tries to close them)
    """
    if not text or not text.strip():
        return None

    raw = text.strip()

    # Strip Qwen3 <think>...</think> blocks
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    # Also handle unclosed think blocks (thinking truncated)
    raw = re.sub(r"<think>.*", "", raw, flags=re.DOTALL).strip()

    # Strip markdown fences
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", raw, re.DOTALL)
    if fence_match:
        raw = fence_match.group(1).strip()

    # Try direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Find the first [ or { and try to parse from there
    for start_char, end_char in [("[", "]"), ("{", "}")]:
        idx = raw.find(start_char)
        if idx == -1:
            continue
        candidate = raw[idx:]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        # Try to fix truncated JSON by closing brackets
        depth = 0
        in_string = False
        escape_next = False
        for i, c in enumerate(candidate):
            if escape_next:
                escape_next = False
                continue
            if c == "\\":
                escape_next = True
                continue
            if c == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if c in ("[", "{"):
                depth += 1
            elif c in ("]", "}"):
                depth -= 1
        if depth > 0:
            # Close open brackets/braces
            fixed = candidate
            for _ in range(depth):
                # Trim trailing comma
                fixed = fixed.rstrip().rstrip(",")
                if start_char == "[":
                    fixed += "]"
                else:
                    fixed += "}"
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass

    logger.warning("Failed to parse JSON from LLM response: %.200s...", raw)
    return None


def _strip_thinking(text: str) -> str:
    """Remove Qwen3 <think>...</think> blocks from response."""
    if not text:
        return text
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()
    return text


# ---------------------------------------------------------------------------
# Async clients
# ---------------------------------------------------------------------------

async def call_vllm(
    prompt: str,
    system_prompt: Optional[str] = None,
    max_tokens: int = 4096,
    temperature: float = 0.1,
    model: Optional[str] = None,
) -> str:
    """Call vLLM (OpenAI-compatible) and return the assistant content."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model or settings.EXTRACTION_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        # Disable Qwen3 thinking mode for structured extraction
        "chat_template_kwargs": {"enable_thinking": False},
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{settings.VLLM_URL}/chat/completions",
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return _strip_thinking(content)


async def call_ollama(
    prompt: str,
    model: str = "qwen3:8b",
    max_tokens: int = 2048,
    system_prompt: Optional[str] = None,
) -> str:
    """Call Ollama generate endpoint and return the response text."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": 0.1,
        },
    }
    if system_prompt:
        payload["system"] = system_prompt

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{settings.OLLAMA_URL}/api/generate",
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()["response"]


# ---------------------------------------------------------------------------
# Sync clients (for Celery workers)
# ---------------------------------------------------------------------------

def call_vllm_sync(
    prompt: str,
    system_prompt: Optional[str] = None,
    max_tokens: int = 4096,
    temperature: float = 0.1,
    model: Optional[str] = None,
) -> str:
    """Sync version of call_vllm for Celery workers."""
    import requests

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model or settings.EXTRACTION_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        # Disable Qwen3 thinking mode for structured extraction
        "chat_template_kwargs": {"enable_thinking": False},
    }

    resp = requests.post(
        f"{settings.VLLM_URL}/chat/completions",
        json=payload,
        timeout=300,
    )
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    return _strip_thinking(content)


def call_ollama_sync(
    prompt: str,
    model: str = "qwen3:8b",
    max_tokens: int = 2048,
    system_prompt: Optional[str] = None,
) -> str:
    """Sync version of call_ollama for Celery workers."""
    import requests

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": 0.1,
        },
    }
    if system_prompt:
        payload["system"] = system_prompt

    resp = requests.post(
        f"{settings.OLLAMA_URL}/api/generate",
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"]
