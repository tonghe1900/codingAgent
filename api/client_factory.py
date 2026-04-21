"""
api/client_factory.py
=====================
Factory that constructs the right :class:`BaseClient` for a given provider.

Supported providers
-------------------
  ``anthropic``  — Default; uses the Anthropic Python SDK.
  ``deepseek``   — DeepSeek AI; OpenAI-compatible API at api.deepseek.com.
  ``openai``     — OpenAI; standard OpenAI chat-completions API.
  ``<custom>``   — Any OpenAI-compatible endpoint; pass ``base_url`` explicitly.

Configuration via Settings
--------------------------
    settings.provider  = "deepseek"
    settings.api_key   = os.getenv("DEEPSEEK_API_KEY")
    settings.base_url  = "https://api.deepseek.com"   # optional override

Environment variables
---------------------
  ANTHROPIC_API_KEY   — API key for Anthropic provider.
  DEEPSEEK_API_KEY    — API key for DeepSeek provider.
  OPENAI_API_KEY      — API key for OpenAI provider.
"""

from __future__ import annotations

import os
from typing import Optional

from claude_code.api.base_client import BaseClient
from claude_code.api.openai_compat_client import PROVIDER_BASE_URLS

# Maps provider names to their default environment variable names.
_API_KEY_ENV: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "deepseek":  "DEEPSEEK_API_KEY",
    "openai":    "OPENAI_API_KEY",
}


def create_client(
    provider: str = "anthropic",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    max_retries: int = 3,
) -> BaseClient:
    """
    Build and return the appropriate :class:`BaseClient` subclass.

    Parameters
    ----------
    provider:    ``"anthropic"``, ``"deepseek"``, ``"openai"``, or any string
                 when combined with a custom ``base_url``.
    api_key:     Explicit API key; falls back to the relevant env var.
    base_url:    Override the provider's default API endpoint.
    max_retries: Passed to the constructed client.

    Raises
    ------
    ValueError   If no API key is found for the requested provider.
    """
    provider = provider.lower()

    # Resolve API key.
    resolved_key = api_key or os.getenv(_API_KEY_ENV.get(provider, ""))
    if not resolved_key:
        env_var = _API_KEY_ENV.get(provider, f"{provider.upper()}_API_KEY")
        raise ValueError(
            f"No API key found for provider '{provider}'. "
            f"Set the {env_var} environment variable or pass api_key=."
        )

    if provider == "anthropic":
        from claude_code.api.claude_client import ClaudeClient  # avoid circular
        return ClaudeClient(api_key=resolved_key, max_retries=max_retries)

    # All other providers use the OpenAI-compatible client.
    from claude_code.api.openai_compat_client import OpenAICompatClient

    resolved_url = base_url or PROVIDER_BASE_URLS.get(provider, "https://api.openai.com/v1")
    return OpenAICompatClient(
        api_key=resolved_key,
        base_url=resolved_url,
        max_retries=max_retries,
    )


def provider_from_model(model: str) -> str:
    """
    Infer the likely provider from a model name string.

    Useful for auto-routing when the user specifies a model but not a provider:
      ``"claude-sonnet-4-6"``  → ``"anthropic"``
      ``"deepseek-chat"``      → ``"deepseek"``
      ``"gpt-4o"``             → ``"openai"``
    """
    model_lower = model.lower()
    if model_lower.startswith("claude"):
        return "anthropic"
    if model_lower.startswith("deepseek"):
        return "deepseek"
    if model_lower.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai"
    return "anthropic"   # safe fallback
