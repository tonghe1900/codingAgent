"""
api/message_converter.py
========================
Bidirectional conversion between Anthropic and OpenAI message formats.

This module is used by :class:`~claude_code.api.openai_compat_client.OpenAICompatClient`
to translate the Anthropic-format history stored in :class:`HistoryManager`
into the OpenAI chat-completions wire format, and to translate OpenAI
responses back into the :class:`TurnResult` the query engine expects.

Anthropic format
----------------
System prompt is passed separately (not in the messages list).

    User turn:
        {"role": "user", "content": "plain text"}
        {"role": "user", "content": [{"type": "text", "text": "..."}]}
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "id",
                                      "content": "...", "is_error": False}]}

    Assistant turn:
        {"role": "assistant", "content": [{"type": "text", "text": "..."}]}
        {"role": "assistant", "content": [{"type": "tool_use",  "id": "...",
                                           "name": "...", "input": {...}}]}

OpenAI format
-------------
System prompt is the first message with role "system".

    User turn:       {"role": "user",      "content": "text"}
    Assistant turn:  {"role": "assistant", "content": "text"}
                     {"role": "assistant", "content": null,
                      "tool_calls": [{"id": "...", "type": "function",
                                      "function": {"name": "...",
                                                   "arguments": "{...}"}}]}
    Tool result:     {"role": "tool", "tool_call_id": "...", "content": "..."}

Tool definition format
----------------------
Anthropic:  {"name": "...", "description": "...", "input_schema": {...}}
OpenAI:     {"type": "function", "function": {"name": "...",
              "description": "...", "parameters": {...}}}
"""

from __future__ import annotations

import json
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def tools_to_openai(anthropic_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert Anthropic tool definitions to OpenAI function-call format."""
    result = []
    for t in anthropic_tools:
        result.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
            },
        })
    return result


# ---------------------------------------------------------------------------
# Messages: Anthropic → OpenAI
# ---------------------------------------------------------------------------

def messages_to_openai(
    anthropic_messages: List[Dict[str, Any]],
    system_prompt: str = "",
) -> List[Dict[str, Any]]:
    """
    Convert a list of Anthropic-format messages (plus a system prompt) to
    the OpenAI chat-completions message list.

    The system prompt is prepended as a ``{"role": "system"}`` message.
    """
    result: List[Dict[str, Any]] = []

    if system_prompt:
        result.append({"role": "system", "content": system_prompt})

    for msg in anthropic_messages:
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            result.extend(_user_msg_to_openai(content))
        elif role == "assistant":
            result.extend(_assistant_msg_to_openai(content))
        # Other roles are skipped (should not appear in well-formed history).

    return result


def _user_msg_to_openai(content: Any) -> List[Dict[str, Any]]:
    """Convert an Anthropic user-message content to one or more OpenAI messages."""
    # Plain string content.
    if isinstance(content, str):
        return [{"role": "user", "content": content}]

    if not isinstance(content, list):
        return [{"role": "user", "content": str(content)}]

    # List of content blocks.  Separate text blocks from tool_result blocks.
    text_parts: List[str] = []
    tool_results: List[Dict[str, Any]] = []

    for block in content:
        btype = block.get("type", "")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "tool_result":
            tool_results.append({
                "role": "tool",
                "tool_call_id": block.get("tool_use_id", ""),
                "content": block.get("content", ""),
            })
        # image_url and other exotic types are skipped.

    result: List[Dict[str, Any]] = []
    if text_parts:
        result.append({"role": "user", "content": " ".join(text_parts)})
    result.extend(tool_results)
    return result


def _assistant_msg_to_openai(content: Any) -> List[Dict[str, Any]]:
    """Convert an Anthropic assistant-message content to an OpenAI assistant message."""
    if isinstance(content, str):
        return [{"role": "assistant", "content": content}]

    if not isinstance(content, list):
        return [{"role": "assistant", "content": str(content)}]

    text_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []

    for block in content:
        btype = block.get("type", "")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "tool_use":
            tool_calls.append({
                "id": block.get("id", ""),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input", {})),
                },
            })
        # thinking blocks are stripped (not sent back).

    msg: Dict[str, Any] = {"role": "assistant"}
    msg["content"] = " ".join(text_parts) if text_parts else None

    if tool_calls:
        msg["tool_calls"] = tool_calls

    return [msg]


# ---------------------------------------------------------------------------
# OpenAI response → Anthropic TurnResult building blocks
# ---------------------------------------------------------------------------

def openai_choice_to_content_blocks(choice: Any) -> List[Dict[str, Any]]:
    """
    Extract content blocks (Anthropic format) from an OpenAI ChatCompletion choice.

    Called after the stream is fully assembled.
    """
    import json

    blocks: List[Dict[str, Any]] = []

    # Text content.
    text = getattr(choice.message, "content", None) or ""
    if text:
        blocks.append({"type": "text", "text": text})

    # Tool calls.
    for tc in getattr(choice.message, "tool_calls", None) or []:
        fn = tc.function
        try:
            input_data = json.loads(fn.arguments or "{}")
        except (json.JSONDecodeError, AttributeError):
            input_data = {}
        blocks.append({
            "type": "tool_use",
            "id": tc.id,
            "name": fn.name,
            "input": input_data,
        })

    return blocks
