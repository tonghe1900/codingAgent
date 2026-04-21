"""
tools/glob_tool.py
==================
Find files matching a glob pattern, sorted by modification time.

This mirrors the TypeScript ``GlobTool`` which wraps fast-glob.  We use
Python's ``pathlib.Path.rglob`` and ``glob`` for standard patterns.

Typical use cases
-----------------
* Find all TypeScript files: ``**/*.ts``
* Find a config file:       ``**/tsconfig.json``
* Find test files:          ``tests/**/*.py``
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

from claude_code.tool import Tool, ToolResult

MAX_RESULTS = 500   # cap to avoid overwhelming the context window


class GlobTool(Tool):
    """Find files matching a glob pattern in a directory tree."""

    name = "Glob"
    description = (
        "Find files that match a glob pattern. "
        "Returns matching paths sorted by modification time (newest first). "
        "Supports patterns like '**/*.py', 'src/**/*.ts', etc."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern to match (e.g. '**/*.py').",
            },
            "path": {
                "type": "string",
                "description": "Root directory to search in (default: current directory).",
            },
            "follow_symlinks": {
                "type": "boolean",
                "description": (
                    "Whether to follow symbolic links when traversing directories. "
                    "Default false (symlinks are listed but not descended into)."
                ),
                "default": False,
            },
        },
        "required": ["pattern"],
    }
    dangerous = False

    def execute(
        self,
        input_data: Dict[str, Any],
        permission_manager=None,
    ) -> ToolResult:
        """Run the glob and return matched paths sorted by mtime."""
        pattern = input_data.get("pattern", "").strip()
        if not pattern:
            return ToolResult.error("No pattern provided.")

        root_str = input_data.get("path") or os.getcwd()
        root = Path(root_str).expanduser().resolve()
        follow_symlinks = bool(input_data.get("follow_symlinks", False))

        if not root.exists():
            return ToolResult.error(f"Directory not found: {root}")
        if not root.is_dir():
            return ToolResult.error(f"Not a directory: {root}")

        try:
            matched: List[Path] = list(root.glob(pattern))
        except ValueError as exc:
            return ToolResult.error(f"Invalid glob pattern: {exc}")

        # Only files (not broken paths).  Symlink handling:
        # - When follow_symlinks=True  → include symlinks that point to files.
        # - When follow_symlinks=False → include only real files; skip dangling
        #   or directory symlinks (prevents accidental infinite-loop traversal).
        files: List[Path] = []
        for p in matched:
            try:
                if follow_symlinks:
                    # resolve() follows links; is_file() on the resolved path.
                    if p.resolve().is_file():
                        files.append(p)
                else:
                    # is_file() already returns False for directories even via symlinks.
                    if p.is_file() and not p.is_symlink():
                        files.append(p)
                    elif p.is_file():
                        # It's a symlink to a regular file — include it but don't follow.
                        files.append(p)
            except OSError:
                # Permission error or broken symlink — skip silently.
                continue

        # Sort by modification time, newest first.
        files.sort(key=lambda p: p.stat(follow_symlinks=follow_symlinks).st_mtime, reverse=True)

        total = len(files)
        files = files[:MAX_RESULTS]

        if not files:
            return ToolResult.ok(f"No files matched '{pattern}' in {root}")

        lines = [str(p) for p in files]
        output = "\n".join(lines)

        if total > MAX_RESULTS:
            output += f"\n\n[Showing first {MAX_RESULTS} of {total} matches.]"

        return ToolResult.ok(output, total_matches=total)
