"""claude_code.api — Claude API client, OpenAI-compatible client, and error types."""
from claude_code.api.base_client import BaseClient
from claude_code.api.claude_client import ClaudeClient, TurnResult, ToolUseBlock
from claude_code.api.openai_compat_client import OpenAICompatClient, PROVIDER_BASE_URLS
from claude_code.api.client_factory import create_client, provider_from_model
from claude_code.api.errors import (
    ClaudeAPIError,
    RetryableError,
    RateLimitError,
    ServiceUnavailableError,
    NetworkError,
    AuthenticationError,
    PermissionDeniedError,
    ValidationError,
    QuotaExceededError,
    ContextWindowExceededError,
    UnknownAPIError,
    classify_error,
)

__all__ = [
    "BaseClient",
    "ClaudeClient",
    "OpenAICompatClient",
    "PROVIDER_BASE_URLS",
    "create_client",
    "provider_from_model",
    "TurnResult",
    "ToolUseBlock",
    "ClaudeAPIError",
    "RetryableError",
    "RateLimitError",
    "ServiceUnavailableError",
    "NetworkError",
    "AuthenticationError",
    "PermissionDeniedError",
    "ValidationError",
    "QuotaExceededError",
    "ContextWindowExceededError",
    "UnknownAPIError",
    "classify_error",
]
