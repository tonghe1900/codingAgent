"""
query_engine.py
===============
The core conversation loop — the heart of Claude Code.

Agentic loop
------------
1. Build the full API request (system prompt, messages, tools, thinking).
2. Stream the assistant's response.
3. If tools were requested, execute them under the permission manager.
4. Feed results back and repeat until ``stop_reason == "end_turn"``.

Improvements over the initial version
--------------------------------------
* **Budget enforcement**: the loop aborts and informs Claude when the
  session has exceeded ``settings.max_budget_usd``.
* **Fallback model**: on context-window-exceeded or quota errors the engine
  switches to ``settings.fallback_model`` (if configured) and retries once.
* **Context window handling**: :class:`ContextWindowExceededError` triggers
  an immediate history compaction before retrying.
* **Thinking mode**: passes ``thinking=True`` and ``thinking_budget`` when
  ``settings.thinking_enabled`` is set (Anthropic only).
* **Provider-agnostic**: accepts any :class:`BaseClient` subclass.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from claude_code.api.base_client import BaseClient
from claude_code.api.claude_client import TurnResult, ToolUseBlock
from claude_code.api.errors import (
    ContextWindowExceededError, QuotaExceededError, ClaudeAPIError
)
from claude_code.config.settings import Settings
from claude_code.context import build_system_prompt
from claude_code.cost_tracker import CostTracker
from claude_code.history import HistoryManager
from claude_code.permissions.permission_manager import PermissionManager
from claude_code.tool import ToolRegistry, ToolResult

logger = logging.getLogger(__name__)

# Safety cap: maximum agentic tool-use iterations per user query.
MAX_TURNS = 150


class QueryEngine:
    """
    Orchestrates multi-turn conversations with Claude (or any provider),
    including tool execution, budget enforcement, and fallback handling.

    Parameters
    ----------
    settings:       Active session settings.
    client:         Primary LLM client (any BaseClient subclass).
    registry:       Tool registry containing all available tools.
    permission_manager: Permission evaluator.
    history:        Conversation history (shared across queries).
    cost_tracker:   Accumulates token/cost data.
    fallback_client:
        Optional secondary client used when the primary model fails due to
        context-window or quota errors.
    """

    def __init__(
        self,
        settings: Settings,
        client: BaseClient,
        registry: ToolRegistry,
        permission_manager: PermissionManager,
        history: HistoryManager,
        cost_tracker: Optional[CostTracker] = None,
        fallback_client: Optional[BaseClient] = None,
    ) -> None:
        self._settings         = settings
        self._client           = client
        self._fallback_client  = fallback_client
        self._registry         = registry
        self._permission_mgr   = permission_manager
        self._history          = history
        self._cost_tracker     = cost_tracker or CostTracker()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        user_message: str,
        on_text: Optional[Callable[[str], None]] = None,
        on_tool_start: Optional[Callable[[str, dict], None]] = None,
        on_tool_end: Optional[Callable[[str, ToolResult], None]] = None,
        on_turn_complete: Optional[Callable[[TurnResult], None]] = None,
    ) -> str:
        """
        Process one user message through the full agentic loop.

        Returns the final assistant text (concatenated across all turns).
        Raises :class:`ClaudeAPIError` on unrecoverable API failures.
        """
        # --- Budget pre-check ---
        if self._over_budget():
            warning = (
                f"[Budget limit reached: ${self._settings.max_budget_usd:.2f}. "
                "Please start a new session.]"
            )
            if on_text:
                on_text(warning)
            return warning

        self._history.add_user_message(user_message)

        system_prompt = self._build_system_prompt()
        tools_payload = self._registry.to_api_list()

        final_text_parts: list[str] = []
        turns_taken = 0
        active_client = self._client
        active_model  = self._settings.model

        # ----------------------------------------------------------------
        # Agentic loop
        # ----------------------------------------------------------------
        while turns_taken < MAX_TURNS:
            turns_taken += 1
            logger.debug("Agentic turn %d/%d (model=%s)", turns_taken, MAX_TURNS, active_model)

            # --- Call provider ---
            try:
                turn_result = active_client.stream_turn(
                    model=active_model,
                    system=system_prompt,
                    messages=self._history.get_messages(),
                    tools=tools_payload,
                    max_tokens=self._settings.max_tokens,
                    on_text=on_text,
                    thinking=self._settings.thinking_enabled,
                    thinking_budget=self._settings.thinking_budget,
                )
            except ContextWindowExceededError:
                logger.warning("Context window exceeded. Compacting history and retrying.")
                self._history._compact()   # force immediate compact
                # Retry this turn with compacted history.
                try:
                    turn_result = active_client.stream_turn(
                        model=active_model,
                        system=system_prompt,
                        messages=self._history.get_messages(),
                        tools=tools_payload,
                        max_tokens=self._settings.max_tokens,
                        on_text=on_text,
                    )
                except ClaudeAPIError as exc:
                    raise
            except (QuotaExceededError, ClaudeAPIError) as exc:
                # Try fallback model/client.
                fb_result = self._try_fallback(
                    exc=exc,
                    model=active_model,
                    system_prompt=system_prompt,
                    tools_payload=tools_payload,
                    on_text=on_text,
                )
                if fb_result is None:
                    raise
                turn_result = fb_result
                active_client = self._fallback_client or active_client
                active_model  = self._settings.fallback_model or active_model

            # --- Record usage and check post-turn budget ---
            turn_usage = self._cost_tracker.record(
                model=active_model,
                input_tokens=turn_result.input_tokens,
                output_tokens=turn_result.output_tokens,
                cache_read_tokens=turn_result.cache_read_tokens,
                cache_write_tokens=turn_result.cache_write_tokens,
            )
            if on_turn_complete:
                on_turn_complete(turn_result)

            # --- Append assistant message ---
            self._history.add_assistant_message(turn_result.message["content"])

            # Collect text.
            for block in turn_result.message.get("content", []):
                if isinstance(block, dict) and block.get("type") == "text":
                    final_text_parts.append(block.get("text", ""))

            # --- Budget post-check ---
            if self._over_budget():
                warn = (
                    f"\n\n[Session budget ${self._settings.max_budget_usd:.2f} "
                    "reached. Stopping.]"
                )
                if on_text:
                    on_text(warn)
                final_text_parts.append(warn)
                break

            # --- End turn ---
            if turn_result.stop_reason == "end_turn" or not turn_result.tool_uses:
                break

            # --- Execute tools ---
            self._execute_tool_uses(
                turn_result.tool_uses,
                on_tool_start=on_tool_start,
                on_tool_end=on_tool_end,
            )

        else:
            warn = f"\n\n[Warning: reached the {MAX_TURNS}-turn safety limit.]"
            if on_text:
                on_text(warn)
            final_text_parts.append(warn)

        return "".join(final_text_parts)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        return build_system_prompt(
            cwd=self._settings.cwd,
            extra=self._settings.system_prompt_extra,
            append=self._settings.append_system_prompt,
            custom=self._settings.custom_system_prompt,
        )

    def _over_budget(self) -> bool:
        """Return True when the session has exceeded the configured cost cap."""
        cap = self._settings.max_budget_usd
        if cap is None:
            return False
        return self._cost_tracker.totals.total_cost_usd >= cap

    def _try_fallback(
        self,
        exc: ClaudeAPIError,
        model: str,
        system_prompt: str,
        tools_payload: list,
        on_text: Optional[Callable[[str], None]],
    ) -> Optional[TurnResult]:
        """
        Attempt one retry using the fallback model/client.

        Returns the TurnResult on success, or None if no fallback is configured.
        """
        if not self._settings.fallback_model:
            return None
        if self._fallback_client is None:
            return None

        logger.warning(
            "Primary model '%s' failed (%s). Retrying with fallback '%s'.",
            model, type(exc).__name__, self._settings.fallback_model,
        )
        try:
            return self._fallback_client.stream_turn(
                model=self._settings.fallback_model,
                system=system_prompt,
                messages=self._history.get_messages(),
                tools=tools_payload,
                max_tokens=self._settings.max_tokens,
                on_text=on_text,
            )
        except ClaudeAPIError:
            return None

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    def _execute_tool_uses(
        self,
        tool_uses: list[ToolUseBlock],
        on_tool_start: Optional[Callable[[str, dict], None]],
        on_tool_end: Optional[Callable[[str, ToolResult], None]],
    ) -> None:
        for tool_use in tool_uses:
            tool = self._registry.find(tool_use.name)

            if tool is None:
                result = ToolResult.error(
                    f"Tool '{tool_use.name}' is not registered."
                )
            else:
                if on_tool_start:
                    on_tool_start(tool_use.name, tool_use.input)
                logger.debug("Executing '%s'", tool_use.name)
                try:
                    result = tool.execute(
                        input_data=tool_use.input,
                        permission_manager=self._permission_mgr,
                    )
                except Exception as exc:
                    logger.exception("Unexpected error in tool '%s'", tool_use.name)
                    result = ToolResult.error(
                        f"Unexpected error in {tool_use.name}: {exc}"
                    )

            if on_tool_end:
                on_tool_end(tool_use.name, result)

            self._history.add_tool_result(
                tool_use_id=tool_use.id,
                result_content=result.content,
                is_error=result.is_error,
            )
