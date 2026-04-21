"""
tools/sleep_tool.py
===================
SleepTool — pauses execution for a specified number of seconds.

Used by agentic loops when Claude needs to wait for an external process,
a deployment to complete, a service to restart, etc.

Safety limits
-------------
* Minimum sleep: 0.1 s
* Maximum sleep: 60 s (to avoid indefinitely hanging the REPL)
"""

from __future__ import annotations

import time
from typing import Any, Dict

from claude_code.tool import Tool, ToolResult


class SleepTool(Tool):
    """Pause execution for N seconds."""

    name = "Sleep"
    description = (
        "Pause execution for a specified number of seconds (max 60). "
        "Useful when waiting for an external process, deployment, or service restart."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "seconds": {
                "type": "number",
                "description": "Duration to sleep in seconds (0.1 – 60).",
            },
        },
        "required": ["seconds"],
    }
    dangerous = False

    def execute(
        self,
        input_data: Dict[str, Any],
        permission_manager=None,
    ) -> ToolResult:
        try:
            raw = float(input_data.get("seconds", 1))
        except (TypeError, ValueError):
            return ToolResult.error("'seconds' must be a number.")

        # Clamp to [0.1, 60].
        duration = max(0.1, min(raw, 60.0))
        if raw > 60:
            note = f"(clamped from {raw}s to 60s max)"
        elif raw < 0.1:
            note = f"(clamped from {raw}s to 0.1s min)"
        else:
            note = ""

        time.sleep(duration)
        msg = f"Slept for {duration}s. {note}".strip()
        return ToolResult.ok(msg, seconds=duration)
