"""
tools/file_write_tool.py
========================
Creates or overwrites files on the filesystem.

Design
------
* Marked ``dangerous=True`` because it overwrites files without undo.
* Creates parent directories automatically so Claude can scaffold projects.
* Refuses to write to ``/etc``, ``/sys``, ``/proc`` (and similar) as a
  basic guard against accidents outside the project tree.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from claude_code.tool import Tool, ToolResult, PermissionDecision

# Directories that are always off-limits (quick safety heuristic).
_BLOCKED_PREFIXES = ("/etc/", "/sys/", "/proc/", "/dev/", "/boot/")


class FileWriteTool(Tool):
    """Create a new file or completely overwrite an existing one."""

    name = "Write"
    description = (
        "Write content to a file, creating it (and any parent directories) "
        "if it does not exist, or overwriting it if it does. "
        "Always read a file before overwriting it to avoid losing content."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute path of the file to write.",
            },
            "content": {
                "type": "string",
                "description": "Text content to write to the file.",
            },
        },
        "required": ["file_path", "content"],
    }
    dangerous = True

    def execute(
        self,
        input_data: Dict[str, Any],
        permission_manager=None,
    ) -> ToolResult:
        """Write *content* to *file_path*, creating parent dirs as needed."""

        # Permission check.
        decision = self.check_permission(input_data, permission_manager)
        if decision == PermissionDecision.DENY:
            return ToolResult.error("Permission denied: file write was blocked.")

        raw_path = input_data.get("file_path", "")
        if not raw_path:
            return ToolResult.error("No file_path provided.")

        content = input_data.get("content", "")
        if content is None:
            return ToolResult.error("No content provided.")

        path = Path(raw_path).expanduser().resolve()

        # Safety guard.
        for prefix in _BLOCKED_PREFIXES:
            if str(path).startswith(prefix):
                return ToolResult.error(
                    f"Writing to {prefix} is not allowed for safety reasons."
                )

        # Track whether this is a new file or an update to an existing one
        # (mirrors TypeScript FileWriteTool output type: "create" | "update").
        is_new_file = not path.exists()

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        except OSError as exc:
            return ToolResult.error(f"Could not write file: {exc}")

        lines = content.count("\n") + 1
        op = "Created" if is_new_file else "Updated"
        return ToolResult.ok(
            f"{op} {path} ({lines} lines, {len(content)} bytes)",
            operation="create" if is_new_file else "update",
            file_path=str(path),
        )
