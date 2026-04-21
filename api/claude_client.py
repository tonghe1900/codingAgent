"""
api/claude_client.py
====================
Thin wrapper around the Anthropic Python SDK.

Responsibilities
----------------
* Build the full ``messages.create`` payload (model, tools, system, messages).
* Stream responses token-by-token, yielding text chunks to the caller.
* Collect tool_use blocks from the streamed response.
* Implement exponential back-off retry for transient failures.
* Return a :class:`TurnResult` that bundles the assistant message, any
  tool-use blocks requested, and token-usage figures.

Retry strategy
--------------
Retryable errors (rate-limit, 503, network) are retried up to ``max_retries``
times with exponential back-off + jitter:

    delay = base * (2 ** attempt) + random(0, jitter_cap)

Non-retryable errors (auth, validation, 404) propagate immediately.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, Iterator, List, Optional

from claude_code.api.base_client import BaseClient
from claude_code.api.errors import RetryableError, classify_error

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------

@dataclass
class ToolUseBlock:
    """A single tool_use block from an assistant turn."""
    id: str
    name: str
    input: Dict[str, Any]


@dataclass
class TurnResult:
    """Everything returned from one assistant turn."""

    # The full assistant message dict (ready to append to history).
    message: Dict[str, Any]

    # Extracted tool-use requests (empty list when Claude stops without tools).
    tool_uses: List[ToolUseBlock] = field(default_factory=list)

    # Stop reason reported by the API.
    stop_reason: str = "end_turn"

    # Token counts from the Usage object.
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class ClaudeClient(BaseClient):
    """
    Wraps the Anthropic SDK for use by the query engine.

    Parameters
    ----------
    api_key:
        Overrides the ``ANTHROPIC_API_KEY`` environment variable.
    max_retries:
        Number of retry attempts for retryable errors (default 3).
    retry_base_delay:
        Base delay in seconds for the first retry (default 1.0).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
    ) -> None:
        import anthropic  # type: ignore[import]

        kwargs: Dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key

        self._client = anthropic.Anthropic(**kwargs)
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def supports_thinking(self) -> bool:
        return True

    def stream_turn(
        self,
        *,
        model: str,
        system: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        max_tokens: int = 8192,
        on_text: Optional[Callable[[str], None]] = None,
        thinking: bool = False,
        thinking_budget: int = 5000,
    ) -> TurnResult:
        """
        Send one assistant turn and return a :class:`TurnResult`.

        Streams text tokens through *on_text* callback as they arrive so
        the UI can print them incrementally.

        Parameters
        ----------
        model:       Claude model ID.
        system:      System prompt string.
        messages:    Conversation history (Anthropic SDK format).
        tools:       Tool definitions list.
        max_tokens:  Maximum output tokens.
        on_text:     Called with each streamed text chunk.

        Returns
        -------
        TurnResult with the complete assistant message + tool requests.
        """
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
                    thinking=thinking,
                    thinking_budget=thinking_budget,
                )
            except Exception as raw_exc:
                typed = classify_error(raw_exc)
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
    # Internal streaming implementation
    # ------------------------------------------------------------------

    def _do_stream(
        self,
        *,
        model: str,
        system: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        max_tokens: int,
        on_text: Optional[Callable[[str], None]],
        thinking: bool = False,
        thinking_budget: int = 5000,
    ) -> TurnResult:
        """
        Open a streaming request and collect the response into a TurnResult.

        The Anthropic streaming API emits events:
          - ``content_block_start``  / ``content_block_delta`` / ``content_block_stop``
          - ``message_start``  / ``message_delta`` / ``message_stop``

        We accumulate text blocks and tool_use blocks as they arrive.
        """
        content_blocks: List[Dict[str, Any]] = []
        current_block: Optional[Dict[str, Any]] = None

        stop_reason   = "end_turn"
        input_tokens  = 0
        output_tokens = 0
        cache_read    = 0
        cache_write   = 0

        # Build keyword arguments; add thinking config when requested.
        stream_kwargs: dict = dict(
            model=model,
            system=system,
            messages=messages,      # type: ignore[arg-type]
            tools=tools,            # type: ignore[arg-type]
            max_tokens=max_tokens,
        )
        if thinking:
            # Extended thinking requires the "interleaved-thinking-2025-05-14" beta.
            stream_kwargs["betas"] = ["interleaved-thinking-2025-05-14"]
            stream_kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget,
            }

        # Use the streaming context manager provided by the SDK.
        with self._client.messages.stream(**stream_kwargs) as stream:
            for event in stream:
                event_type = getattr(event, "type", None)

                # ----- message_start — grab initial usage figures -----
                if event_type == "message_start":
                    msg = getattr(event, "message", None)
                    if msg:
                        usage = getattr(msg, "usage", None)
                        if usage:
                            input_tokens  = getattr(usage, "input_tokens", 0)
                            cache_read    = getattr(usage, "cache_read_input_tokens", 0)
                            cache_write   = getattr(usage, "cache_creation_input_tokens", 0)

                # ----- content_block_start — open a new block -----
                elif event_type == "content_block_start":
                    cb = getattr(event, "content_block", None)
                    if cb:
                        btype = getattr(cb, "type", "")
                        if btype == "text":
                            current_block = {"type": "text", "text": ""}
                        elif btype == "tool_use":
                            current_block = {
                                "type": "tool_use",
                                "id":    getattr(cb, "id", ""),
                                "name":  getattr(cb, "name", ""),
                                "input": {},
                                "_raw_input": "",   # accumulate JSON string
                            }

                # ----- content_block_delta — append to current block -----
                elif event_type == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    if delta and current_block is not None:
                        dtype = getattr(delta, "type", "")
                        if dtype == "text_delta":
                            chunk = getattr(delta, "text", "")
                            current_block["text"] += chunk
                            if on_text and chunk:
                                on_text(chunk)
                        elif dtype == "input_json_delta":
                            # JSON arrives as partial strings; accumulate.
                            current_block["_raw_input"] += getattr(delta, "partial_json", "")

                # ----- content_block_stop — finalise the block -----
                elif event_type == "content_block_stop":
                    if current_block is not None:
                        if current_block["type"] == "tool_use":
                            import json
                            raw = current_block.pop("_raw_input", "{}")
                            try:
                                current_block["input"] = json.loads(raw) if raw else {}
                            except Exception:
                                current_block["input"] = {}
                        content_blocks.append(current_block)
                        current_block = None

                # ----- message_delta — stop_reason + final usage -----
                elif event_type == "message_delta":
                    delta = getattr(event, "delta", None)
                    usage = getattr(event, "usage", None)
                    if delta:
                        stop_reason = getattr(delta, "stop_reason", stop_reason) or stop_reason
                    if usage:
                        output_tokens = getattr(usage, "output_tokens", 0)

        # Build tool_uses list.
        tool_uses = [
            ToolUseBlock(id=b["id"], name=b["name"], input=b["input"])
            for b in content_blocks
            if b["type"] == "tool_use"
        ]

        # Build the assistant message dict (history format).
        assistant_message = {"role": "assistant", "content": content_blocks}

        return TurnResult(
            message=assistant_message,
            tool_uses=tool_uses,
            stop_reason=stop_reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
        )

    # ------------------------------------------------------------------
    # Back-off helper
    # ------------------------------------------------------------------

    def _backoff(self, attempt: int) -> float:
        """Return exponential back-off delay with ±25 % jitter."""
        base = self._retry_base_delay * (2 ** attempt)
        jitter = base * 0.25
        return base + random.uniform(-jitter, jitter)
