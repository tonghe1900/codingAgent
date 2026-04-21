"""
tools/file_read_tool.py
=======================
Reads files from the filesystem and returns their content as text.

Improvements vs. the original version
--------------------------------------
* Device path blocking  — ``/dev/zero``, ``/dev/tty``, etc. are blocked to
  prevent hangs.
* Jupyter notebook support — ``.ipynb`` files are rendered as readable
  cell-by-cell text rather than raw JSON.
* Similar-file suggestions — when a file is not found, the tool scans the
  CWD for files with similar names and suggests the closest match.
* Line-number format — ``cat -n`` style output (``1\tline``).
* Mtime-based in-process caching to avoid re-reading unchanged files.
"""

from __future__ import annotations

import json
import os
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from claude_code.tool import Tool, ToolResult

DEFAULT_LINE_LIMIT = 2000

# Refuse to read files larger than this (mirrors TypeScript's maxSizeBytes=256KB
# default in tools/FileReadTool/limits.ts).  Very large files would consume the
# entire context window and are almost always binary or generated files that
# Claude cannot usefully reason about.
MAX_READ_FILE_SIZE = 256 * 1024   # 256 KB

# Paths that should never be read (they hang or are meaningless as text).
BLOCKED_DEVICE_PATHS: set = {
    "/dev/zero", "/dev/random", "/dev/urandom", "/dev/null",
    "/dev/stdin", "/dev/stdout", "/dev/stderr", "/dev/tty",
    "/dev/console", "/dev/full",
}

# Simple (path, mtime) → content cache.
_cache: Dict[Tuple[str, float], str] = {}


class FileReadTool(Tool):
    """Read a file and return its content with line numbers."""

    name = "Read"
    description = (
        "Read a file from the filesystem. "
        "Returns content with line numbers (cat -n format). "
        "Supports .ipynb notebooks (rendered as cells). "
        "Use offset and limit to read a specific range of lines."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute path to the file to read.",
            },
            "offset": {
                "type": "integer",
                "description": "1-indexed line number to start reading from (default 1).",
            },
            "limit": {
                "type": "integer",
                "description": f"Max lines to return (default {DEFAULT_LINE_LIMIT}).",
            },
        },
        "required": ["file_path"],
    }
    dangerous = False

    def execute(
        self,
        input_data: Dict[str, Any],
        permission_manager=None,
    ) -> ToolResult:
        raw_path = input_data.get("file_path", "")
        if not raw_path:
            return ToolResult.error("No file_path provided.")

        path = Path(raw_path).expanduser()

        # Device-path guard.
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if str(resolved) in BLOCKED_DEVICE_PATHS or str(path) in BLOCKED_DEVICE_PATHS:
            return ToolResult.error(
                f"Reading device path '{path}' is not allowed (would hang or produce noise)."
            )

        if not path.exists():
            suggestion = _find_similar_file(path)
            msg = f"File not found: {path}"
            if suggestion:
                msg += f"\nDid you mean: {suggestion}?"
            return ToolResult.error(msg)

        if path.is_dir():
            return ToolResult.error(f"Path is a directory, not a file: {path}")

        # File size guard — refuse files that are too large to be useful as
        # text context (mirrors TypeScript maxSizeBytes=256 KB default).
        try:
            file_size = path.stat().st_size
        except OSError as exc:
            return ToolResult.error(f"Could not stat file: {exc}")

        if file_size > MAX_READ_FILE_SIZE:
            return ToolResult.error(
                f"File is too large to read ({file_size:,} bytes > "
                f"{MAX_READ_FILE_SIZE:,} byte limit): {path}\n"
                "Use the Bash tool with head/tail to inspect specific sections, "
                "or increase the limit via settings."
            )

        # Jupyter notebook: render as cell text.
        if path.suffix.lower() == ".ipynb":
            return self._read_notebook(path)

        # Plain text file.
        offset = max(1, int(input_data.get("offset", 1) or 1))
        limit  = max(1, int(input_data.get("limit",  DEFAULT_LINE_LIMIT) or DEFAULT_LINE_LIMIT))

        try:
            content = _read_cached(path)
        except (OSError, UnicodeDecodeError) as exc:
            return ToolResult.error(f"Could not read file: {exc}")

        lines = content.splitlines(keepends=True)
        total = len(lines)

        start = offset - 1
        end   = min(start + limit, total)
        chunk = lines[start:end]

        numbered = "".join(f"{start + i + 1}\t{line}" for i, line in enumerate(chunk))

        if end < total:
            numbered += (
                f"\n[Showing lines {offset}–{end} of {total}. "
                f"Use offset={end + 1} to read more.]"
            )

        return ToolResult.ok(numbered, total_lines=total)

    # ------------------------------------------------------------------
    # Jupyter notebook renderer
    # ------------------------------------------------------------------

    @staticmethod
    def _read_notebook(path: Path) -> ToolResult:
        """Parse an .ipynb file and return a readable cell-by-cell summary."""
        try:
            nb = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            return ToolResult.error(f"Could not parse notebook: {exc}")

        cells = nb.get("cells", [])
        parts: List[str] = [f"Notebook: {path}  ({len(cells)} cells)\n"]

        for i, cell in enumerate(cells):
            ctype  = cell.get("cell_type", "code")
            source = "".join(cell.get("source", []))
            # Line-number the source.
            src_lines = source.splitlines()
            numbered  = "\n".join(f"{j + 1}\t{l}" for j, l in enumerate(src_lines))

            header = f"[{i}] {ctype.upper()}"
            parts.append(f"{header}\n{numbered}")

            # Include text outputs (skip images / binary data).
            for out in cell.get("outputs", []):
                otype = out.get("output_type", "")
                if otype in ("stream", "execute_result", "display_data"):
                    text = out.get("text") or out.get("data", {}).get("text/plain", [])
                    if isinstance(text, list):
                        text = "".join(text)
                    if text:
                        parts.append(f"  → output:\n    {text.strip()}")

        return ToolResult.ok("\n\n".join(parts), cell_count=len(cells))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_cached(path: Path) -> str:
    """Return file content using an mtime-keyed cache."""
    mtime = path.stat().st_mtime
    key   = (str(path), mtime)
    if key not in _cache:
        _cache[key] = path.read_text(encoding="utf-8", errors="replace")
    return _cache[key]


def _find_similar_file(path: Path) -> Optional[str]:
    """
    Return the path of a file in the same directory (or CWD) whose name is
    close to *path.name*.  Returns None when nothing useful is found.
    """
    search_dir = path.parent if path.parent.exists() else Path(os.getcwd())
    try:
        candidates = [p.name for p in search_dir.iterdir() if p.is_file()]
    except OSError:
        return None

    matches = get_close_matches(path.name, candidates, n=1, cutoff=0.6)
    if matches:
        return str(search_dir / matches[0])
    return None
