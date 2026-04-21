"""
api/errors.py
=============
Error hierarchy for the Claude API client.

Maps HTTP/SDK error codes to typed Python exceptions so the rest of the
codebase can react to specific failure modes (e.g. retry on 429, bail out
on 401) without inspecting raw HTTP status codes everywhere.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class ClaudeAPIError(Exception):
    """Root exception for all Claude-API-related errors."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


# ---------------------------------------------------------------------------
# Retryable errors — caller may back off and try again
# ---------------------------------------------------------------------------

class RetryableError(ClaudeAPIError):
    """Transient failure; safe to retry with exponential back-off."""


class RateLimitError(RetryableError):
    """HTTP 429 — too many requests."""


class ServiceUnavailableError(RetryableError):
    """HTTP 503 / 529 — overloaded or temporarily down."""


class NetworkError(RetryableError):
    """Connection dropped, DNS failure, timeout, etc."""


# ---------------------------------------------------------------------------
# Fatal / non-retryable errors
# ---------------------------------------------------------------------------

class AuthenticationError(ClaudeAPIError):
    """HTTP 401 — missing or invalid API key."""


class PermissionDeniedError(ClaudeAPIError):
    """HTTP 403 — key exists but lacks access to this resource."""


class NotFoundError(ClaudeAPIError):
    """HTTP 404 — model or resource does not exist."""


class ValidationError(ClaudeAPIError):
    """HTTP 422 — malformed request payload."""


class QuotaExceededError(ClaudeAPIError):
    """Monthly / daily credit quota exhausted (distinct from per-minute rate limit)."""


class InsufficientBalanceError(ClaudeAPIError):
    """
    HTTP 402 — account has no credits remaining.

    DeepSeek and some other providers return 402 when the account balance
    is zero or the prepaid quota has been used up.  Distinct from
    QuotaExceededError (which is a rate/usage quota) so the UI can show
    a billing-specific message with a link to the top-up page.
    """


class ContextWindowExceededError(ClaudeAPIError):
    """The request exceeds the model's context-window limit."""


class UnknownAPIError(ClaudeAPIError):
    """Any other non-2xx response that doesn't map to a known category."""


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def classify_error(exc: Exception) -> ClaudeAPIError:
    """
    Convert an Anthropic SDK exception (or generic exception) into one of
    our typed errors.

    The Anthropic SDK raises ``anthropic.APIStatusError`` sub-classes; we
    inspect their ``status_code`` to pick the right bucket.
    """
    # Import here to avoid making anthropic a hard dep for unit tests that
    # mock this function.
    try:
        import anthropic as _anthropic  # type: ignore[import]
    except ImportError:
        return NetworkError(str(exc))

    if isinstance(exc, _anthropic.AuthenticationError):
        return AuthenticationError(str(exc), status_code=401)
    if isinstance(exc, _anthropic.PermissionDeniedError):
        return PermissionDeniedError(str(exc), status_code=403)
    if isinstance(exc, _anthropic.NotFoundError):
        return NotFoundError(str(exc), status_code=404)
    if isinstance(exc, _anthropic.UnprocessableEntityError):
        return ValidationError(str(exc), status_code=422)
    if isinstance(exc, _anthropic.RateLimitError):
        return RateLimitError(str(exc), status_code=429)
    if isinstance(exc, _anthropic.InternalServerError):
        return ServiceUnavailableError(str(exc), status_code=500)
    if isinstance(exc, _anthropic.APIConnectionError):
        return NetworkError(str(exc))
    if isinstance(exc, _anthropic.APIStatusError):
        return UnknownAPIError(str(exc), status_code=exc.status_code)

    # Fallback for plain network / timeout exceptions.
    return NetworkError(str(exc))
