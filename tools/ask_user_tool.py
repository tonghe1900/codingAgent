"""
tools/ask_user_tool.py
======================
AskUserQuestionTool — lets Claude ask the user a clarifying question
mid-task and receive a typed response.

In interactive mode the tool prints the question and reads a line from
stdin.  In non-interactive mode (``interactive=False``) it returns an
empty string immediately, which Claude interprets as "no answer provided".

This mirrors the TypeScript ``AskUserQuestionTool`` used in the agentic
loop when Claude is uncertain about intent.
"""

from __future__ import annotations

from typing import Any, Dict

from claude_code.tool import Tool, ToolResult


class AskUserQuestionTool(Tool):
    """Ask the user a clarifying question and return their answer."""

    name = "AskUser"
    description = (
        "Ask the user a clarifying question when intent is ambiguous. "
        "Returns the user's typed response. "
        "Use sparingly — prefer reasonable assumptions over interruption."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question to ask the user.",
            },
            "options": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of suggested answers to display.",
            },
        },
        "required": ["question"],
    }
    dangerous = False

    def __init__(self, interactive: bool = True) -> None:
        self._interactive = interactive

    def execute(
        self,
        input_data: Dict[str, Any],
        permission_manager=None,
    ) -> ToolResult:
        question = input_data.get("question", "").strip()
        if not question:
            return ToolResult.error("No question provided.")

        options: list = input_data.get("options") or []

        if not self._interactive:
            return ToolResult.ok(
                "(Non-interactive mode: no user response available.)",
                answered=False,
            )

        print(f"\n[Claude asks]: {question}")
        if options:
            for i, opt in enumerate(options, start=1):
                print(f"  {i}. {opt}")

        try:
            answer = input("Your answer: ").strip()
        except (EOFError, KeyboardInterrupt):
            return ToolResult.ok(
                "(User did not respond.)", answered=False
            )

        return ToolResult.ok(answer, answered=bool(answer))
