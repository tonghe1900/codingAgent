"""
api/openai_compat_client.py
============================
OpenAI-compatible LLM client.

Supports any provider that implements the OpenAI Chat Completions API,
including:
  * DeepSeek  (base_url = "https://api.deepseek.com")
  * OpenAI    (base_url = "https://api.openai.com/v1")
  * Azure OpenAI
  * Local models via ollama / llama.cpp

Uses the ``openai`` Python SDK (which DeepSeek officially recommends)
so we get streaming, retry, and timeout handling for free.

History is stored in Anthropic format by :class:`HistoryManager`;
this client translates to/from OpenAI format transparently.

Usage
-----
    client = OpenAICompatClient(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
    )
    result = client.stream_turn(
        model="deepseek-chat",
        system="You are helpful.",
        messages=[{"role": "user", "content": "Hello"}],
        tools=[],
        on_text=print,
    )
"""

from __future__ import annotations

import json
import logging
import random
import time
from typing import Callable, Dict, List, Optional

from claude_code.api.base_client import BaseClient
from claude_code.api.claude_client import TurnResult, ToolUseBlock
from claude_code.api.errors import (
    RetryableError, RateLimitError, ServiceUnavailableError,
    NetworkError, AuthenticationError, ValidationError,
    PermissionDeniedError, UnknownAPIError, InsufficientBalanceError,
)
from claude_code.api.message_converter import (
    messages_to_openai,
    tools_to_openai,
    openai_choice_to_content_blocks,
)

logger = logging.getLogger(__name__)

# Default base URLs per provider name.
PROVIDER_BASE_URLS: Dict[str, str] = {
    "deepseek": "https://api.deepseek.com",
    "openai":   "https://api.openai.com/v1",
}


class OpenAICompatClient(BaseClient):
    """
    Provider client for OpenAI-compatible APIs (DeepSeek, OpenAI, …).

    Parameters
    ----------
    api_key:
        API key for the provider.
    base_url:
        Base URL for the API.  Defaults to OpenAI's endpoint.
    max_retries:
        Retry attempts for transient errors.
    retry_base_delay:
        Base back-off delay (seconds) for the first retry.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
    ) -> None:
        try:
            from openai import OpenAI  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for OpenAI-compatible providers. "
                "Run: pip install openai"
            ) from exc

        self._openai_client = OpenAI(api_key=api_key, base_url=base_url)
        self._base_url = base_url
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay

    # ------------------------------------------------------------------
    # BaseClient interface
    # ------------------------------------------------------------------

    def stream_turn(
        self,
        *,
        model: str,
        system: str,
        messages: List[Dict],
        tools: List[Dict],
        max_tokens: int = 8192,
        on_text: Optional[Callable[[str], None]] = None,
        thinking: bool = False,
        thinking_budget: int = 5000,
    ) -> TurnResult:
        """Stream one assistant turn and return a TurnResult."""
        attempt = 0
        while True:
            try:
                return self._do_stream(
                    model=model,
                    system=system,
                    messages=messages,
                    tools=tools,
                    max_tokens=max_tokens,
                    on_text=on_text,
                )
            except Exception as raw_exc:
                typed = self._classify(raw_exc)
                if not isinstance(typed, RetryableError) or attempt >= self._max_retries:
                    raise typed from raw_exc
                delay = self._backoff(attempt)
                logger.warning(
                    "Retryable error (%s). Retrying in %.1fs (attempt %d/%d).",
                    type(typed).__name__, delay, attempt + 1, self._max_retries,
                )
                time.sleep(delay)
                attempt += 1

    # ------------------------------------------------------------------
    # Internal streaming
    # ------------------------------------------------------------------

    def _do_stream(
        self,
        *,
        model: str,
        system: str,
        messages: List[Dict],
        tools: List[Dict],
        max_tokens: int,
        on_text: Optional[Callable[[str], None]],
    ) -> TurnResult:
        """
        Convert messages to OpenAI format, open a streaming request, collect
        the response, and return a provider-agnostic TurnResult.
        """
        oai_messages = messages_to_openai(messages, system_prompt=system)
        oai_tools = tools_to_openai(tools) if tools else []

        # Accumulate streamed content.
        text_buffer: str = ""
        # tool_calls accumulate: id → {id, name, arguments_so_far}
        tool_call_buffer: Dict[int, Dict] = {}

        finish_reason: str = "stop"
        input_tokens = 0
        output_tokens = 0

        create_kwargs: Dict = dict(
            model=model,
            messages=oai_messages,
            max_tokens=max_tokens,
            stream=True,
        )
        if oai_tools:
            create_kwargs["tools"] = oai_tools
            create_kwargs["tool_choice"] = "auto"

        with self._openai_client.chat.completions.create(**create_kwargs) as stream:
            for chunk in stream:
                # Usage (some providers send this on the last chunk).
                if hasattr(chunk, "usage") and chunk.usage:
                    input_tokens  = getattr(chunk.usage, "prompt_tokens",     0)
                    output_tokens = getattr(chunk.usage, "completion_tokens", 0)

                choices = getattr(chunk, "choices", [])
                if not choices:
                    continue

                choice = choices[0]
                delta  = getattr(choice, "delta", None)

                if getattr(choice, "finish_reason", None):
                    finish_reason = choice.finish_reason or "stop"

                if delta is None:
                    continue

                # Text delta.
                text_chunk = getattr(delta, "content", None)
                if text_chunk:
                    text_buffer += text_chunk
                    if on_text:
                        on_text(text_chunk)

                # Tool-call deltas.
                for tc_delta in getattr(delta, "tool_calls", None) or []:
                    idx = tc_delta.index
                    if idx not in tool_call_buffer:
                        tool_call_buffer[idx] = {
                            "id":   getattr(tc_delta, "id", "") or "",
                            "name": "",
                            "args": "",
                        }
                    buf = tool_call_buffer[idx]

                    fn = getattr(tc_delta, "function", None)
                    if fn:
                        if getattr(fn, "name", None):
                            buf["name"] += fn.name
                        if getattr(fn, "arguments", None):
                            buf["args"] += fn.arguments
                    # Some providers send id on subsequent deltas.
                    if getattr(tc_delta, "id", None):
                        buf["id"] = tc_delta.id

        # Build Anthropic-format content blocks from the assembled response.
        content_blocks: List[Dict] = []
        if text_buffer:
            content_blocks.append({"type": "text", "text": text_buffer})

        tool_uses: List[ToolUseBlock] = []
        for idx in sorted(tool_call_buffer):
            buf = tool_call_buffer[idx]
            try:
                input_data = json.loads(buf["args"]) if buf["args"] else {}
            except json.JSONDecodeError:
                input_data = {}

            block: Dict = {
                "type": "tool_use",
                "id":    buf["id"],
                "name":  buf["name"],
                "input": input_data,
            }
            content_blocks.append(block)
            tool_uses.append(ToolUseBlock(id=buf["id"], name=buf["name"], input=input_data))

        # Map finish_reason to Anthropic stop_reason vocabulary.
        stop_reason = {
            "stop":         "end_turn",
            "tool_calls":   "tool_use",
            "length":       "max_tokens",
            "content_filter": "end_turn",
        }.get(finish_reason, "end_turn")

        return TurnResult(
            message={"role": "assistant", "content": content_blocks},
            tool_uses=tool_uses,
            stop_reason=stop_reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    # ------------------------------------------------------------------
    # Error classification
    # ------------------------------------------------------------------

    @staticmethod
    def _classify(exc: Exception) -> Exception:
        """Map openai SDK exceptions to our typed error hierarchy."""
        try:
            import openai as _openai  # type: ignore[import]
        except ImportError:
            return NetworkError(str(exc))

        if isinstance(exc, _openai.AuthenticationError):
            return AuthenticationError(str(exc), status_code=401)
        if isinstance(exc, _openai.PermissionDeniedError):
            return PermissionDeniedError(str(exc), status_code=403)
        if isinstance(exc, _openai.RateLimitError):
            return RateLimitError(str(exc), status_code=429)
        if isinstance(exc, _openai.BadRequestError):
            return ValidationError(str(exc), status_code=400)
        if isinstance(exc, _openai.InternalServerError):
            return ServiceUnavailableError(str(exc), status_code=500)
        if isinstance(exc, _openai.APIConnectionError):
            return NetworkError(str(exc))
        if isinstance(exc, _openai.APIStatusError):
            # HTTP 402 = payment required / insufficient balance (DeepSeek, etc.)
            if exc.status_code == 402:
                return InsufficientBalanceError(str(exc), status_code=402)
            return UnknownAPIError(str(exc), status_code=exc.status_code)

        return NetworkError(str(exc))

    # ------------------------------------------------------------------
    # Backoff
    # ------------------------------------------------------------------

    def _backoff(self, attempt: int) -> float:
        base = self._retry_base_delay * (2 ** attempt)
        return base + random.uniform(0, base * 0.25)
