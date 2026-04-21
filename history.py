"""
history.py
==========
Manages the in-memory message history for one conversation.

Responsibilities
----------------
* Append user, assistant, and tool-result messages in the correct format
  expected by the Anthropic SDK.
* Detect when the history is approaching the context-window limit and
  compact (summarise) older messages to free space.
* Provide the current history slice to the query engine before each turn.

Message format
--------------
We keep messages in the Anthropic SDK dict format so they can be passed
directly to ``client.messages.create(messages=...)``:

    [
        {"role": "user",      "content": "Hello"},
        {"role": "assistant", "content": [{"type": "text", "text": "Hi!"}]},
        ...
    ]
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# A single message dict as accepted by the Anthropic SDK.
Message = Dict[str, Any]


class HistoryManager:
    """
    In-memory conversation history with optional compaction.

    Parameters
    ----------
    max_tokens:
        The model's context-window size.  Used only as a reference for the
        compaction heuristic — we rely on approximate character counts rather
        than tokenising (a full tokeniser is not worth the dependency here).
    compact_threshold:
        When estimated token usage exceeds ``max_tokens * compact_threshold``
        the manager will compact the oldest 50 % of non-system messages.
    """

    # Rough characters-per-token estimate (conservative).
    _CHARS_PER_TOKEN = 3.5

    def __init__(
        self,
        max_tokens: int = 8192,
        compact_threshold: float = 0.85,
    ) -> None:
        self._messages: List[Message] = []
        self._max_tokens = max_tokens
        self._compact_threshold = compact_threshold

    # ------------------------------------------------------------------
    # Append helpers
    # ------------------------------------------------------------------

    def add_user_message(self, content: str | List[Any]) -> None:
        """Append a user turn.  *content* may be a plain string or a list of blocks."""
        self._messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str | List[Any]) -> None:
        """Append an assistant turn."""
        self._messages.append({"role": "assistant", "content": content})

    def add_tool_result(
        self,
        tool_use_id: str,
        result_content: str,
        is_error: bool = False,
    ) -> None:
        """
        Append a tool-result user turn.

        The Anthropic API requires tool results to be sent as a user message
        containing a ``tool_result`` content block.
        """
        block: Dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": result_content,
        }
        if is_error:
            block["is_error"] = True

        # Merge into the last user message if it already contains tool_results,
        # otherwise open a new user turn.
        if (
            self._messages
            and self._messages[-1]["role"] == "user"
            and isinstance(self._messages[-1]["content"], list)
            and any(
                b.get("type") == "tool_result"
                for b in self._messages[-1]["content"]
            )
        ):
            self._messages[-1]["content"].append(block)
        else:
            self._messages.append({"role": "user", "content": [block]})

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def get_messages(self) -> List[Message]:
        """Return the current (possibly compacted) message list."""
        self._maybe_compact()
        return list(self._messages)

    def __len__(self) -> int:
        return len(self._messages)

    def clear(self) -> None:
        """Wipe all messages (start a fresh conversation)."""
        self._messages.clear()

    def remove_last(self) -> Optional[Message]:
        """
        Remove and return the most-recent message (undo last append).

        Returns ``None`` when the history is already empty.  Useful for
        retry logic where the last user or assistant turn should be replayed.
        """
        if not self._messages:
            return None
        return self._messages.pop()

    # ------------------------------------------------------------------
    # Compaction
    # ------------------------------------------------------------------

    def _estimated_tokens(self) -> int:
        """
        Rough token count for the current history.

        We serialise each message to a string and divide by the
        ``_CHARS_PER_TOKEN`` constant.  This is cheap and ~accurate enough
        for triggering compaction decisions.
        """
        total_chars = sum(
            len(str(m.get("content", "")))
            for m in self._messages
        )
        return int(total_chars / self._CHARS_PER_TOKEN)

    def _maybe_compact(self) -> None:
        """Compact the oldest half of messages if nearing the token budget."""
        if self._estimated_tokens() < self._max_tokens * self._compact_threshold:
            return

        logger.info(
            "History approaching context limit (~%d tokens). Compacting.",
            self._estimated_tokens(),
        )
        self._compact()

    def _compact(self) -> None:
        """
        Remove the oldest ~50 % of messages, replacing them with a synthetic
        summary placeholder so Claude understands the context gap.

        Safe-cut rule
        -------------
        The cut must land on an *assistant* message so that the injected
        summary_block (a plain-text user message) immediately precedes it.
        This guarantees:

        * user → assistant alternation remains valid for the Anthropic API.
        * No orphan ``tool``-role messages appear at the start of the
          compacted slice in the OpenAI wire format: every ``tool`` result
          message is still preceded by the assistant message whose
          ``tool_calls`` it answers (DeepSeek / OpenAI return HTTP 400 if
          that pairing is broken).

        The search goes *forward* from the midpoint first, then *backward*,
        so long single-task agentic loops (where assistant messages appear
        at every other index starting from 1) always find a cut point.
        """
        n = len(self._messages)
        mid = max(1, n // 2)

        # Search forward from mid, then backward — stop at the first
        # assistant message found.
        cutoff: Optional[int] = None
        for i in range(mid, n):
            if self._messages[i]["role"] == "assistant":
                cutoff = i
                break
        if cutoff is None:
            for i in range(mid - 1, 0, -1):
                if self._messages[i]["role"] == "assistant":
                    cutoff = i
                    break

        if cutoff is None:
            # Degenerate history with no assistant messages — nothing to do.
            logger.debug(
                "Compaction skipped: no assistant message found in "
                "%d-message history.", n
            )
            return

        summary_block: Message = {
            "role": "user",
            "content": (
                "[System: Earlier conversation history was compacted to stay "
                "within the context window. The conversation continues below.]"
            ),
        }

        self._messages = [summary_block] + self._messages[cutoff:]
        logger.info(
            "Compacted history from %d to %d messages (cut at index %d).",
            n, len(self._messages), cutoff,
        )
