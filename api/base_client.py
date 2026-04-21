"""
api/base_client.py
==================
Abstract base class that every LLM provider client must implement.

The query engine only depends on this interface, so swapping providers
(Anthropic → DeepSeek → OpenAI) requires no changes outside of this
package.

All clients must return the same :class:`~claude_code.api.claude_client.TurnResult`
shape so the rest of the codebase is provider-agnostic.
"""

from __future__ import annotations

import abc
from typing import Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from claude_code.api.claude_client import TurnResult


class BaseClient(abc.ABC):
    """
    Provider-agnostic LLM client interface.

    Subclasses handle the wire protocol (Anthropic streaming API, OpenAI
    chat-completions API, etc.) and return a uniform :class:`TurnResult`.
    """

    @abc.abstractmethod
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
    ) -> "TurnResult":
        """
        Send one assistant turn, stream the response, and return a TurnResult.

        Parameters
        ----------
        model:            Provider-specific model ID.
        system:           System prompt string.
        messages:         Conversation history in **Anthropic format**.
                          Clients are responsible for converting to their
                          native wire format before sending.
        tools:            Tool definitions in **Anthropic format**.
        max_tokens:       Maximum output tokens.
        on_text:          Called with each streamed text chunk.
        thinking:         Enable extended thinking / chain-of-thought.
        thinking_budget:  Token budget for thinking (Anthropic only).
        """

    # ------------------------------------------------------------------
    # Optional overrides
    # ------------------------------------------------------------------

    @property
    def supports_thinking(self) -> bool:
        """Return True if this provider supports extended thinking mode."""
        return False

    @property
    def supports_tool_use(self) -> bool:
        """Return True if this provider supports structured tool calls."""
        return True
