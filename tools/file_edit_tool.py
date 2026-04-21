"""
tools/file_edit_tool.py
=======================
Performs exact string-replacement edits on existing files.

This mirrors the TypeScript ``FileEditTool`` / ``Edit`` tool:
* Takes ``old_string`` and ``new_string``.
* Replaces the first occurrence of ``old_string`` in the file with
  ``new_string`` (or all occurrences when ``replace_all=True``).
* Refuses to operate on files that do not yet exist — use FileWriteTool
  for new files.
* Returns a small context diff so Claude can verify the change.

Improvements vs. earlier version
---------------------------------
* File size limit   — refuses to edit files larger than MAX_EDIT_FILE_SIZE
  (10 MB) to prevent OOM on binary/generated files.
* Line-ending preservation — detects CRLF vs LF and writes back in the
  same style so edits never silently normalise line endings.
* JSON/settings validation — when editing a ``.json`` file, validates that
  the result is well-formed JSON before writing.

Design rationale
----------------
Sending only a diff (rather than the full file) keeps token usage low when
editing large files.  Claude Code's TypeScript source does the same.
"""

from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import Any, Dict, Optional

from claude_code.tool import Tool, ToolResult, PermissionDecision

# Refuse edits on files larger than this to prevent OOM.
MAX_EDIT_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

# ---------------------------------------------------------------------------
# Curly-quote normalisation
# (mirrors TypeScript tools/FileEditTool/utils.ts  normalizeQuotes /
#  findActualString / preserveQuoteStyle)
# ---------------------------------------------------------------------------

# Mapping of curly-quote characters → their ASCII equivalents.
_CURLY_TO_STRAIGHT: dict[str, str] = {
    "\u2018": "'",   # LEFT  SINGLE QUOTATION MARK  '
    "\u2019": "'",   # RIGHT SINGLE QUOTATION MARK  '
    "\u201c": '"',   # LEFT  DOUBLE QUOTATION MARK  "
    "\u201d": '"',   # RIGHT DOUBLE QUOTATION MARK  "
}
_STRAIGHT_TO_LEFT: dict[str, str] = {
    "'": "\u2018",
    '"': "\u201c",
}
_STRAIGHT_TO_RIGHT: dict[str, str] = {
    "'": "\u2019",
    '"': "\u201d",
}


def _normalize_quotes(text: str) -> str:
    """Replace curly/typographic quotes with their ASCII equivalents."""
    for curly, straight in _CURLY_TO_STRAIGHT.items():
        text = text.replace(curly, straight)
    return text


def _find_with_quote_fallback(
    file_content: str, search_string: str
) -> Optional[str]:
    """
    Return the string in *file_content* that matches *search_string*,
    trying first an exact match, then a quote-normalised match.

    Returns ``None`` when no match is found.
    """
    # Exact match first.
    if search_string in file_content:
        return search_string

    # Try with quotes normalised on both sides.
    norm_search = _normalize_quotes(search_string)
    norm_file   = _normalize_quotes(file_content)
    idx = norm_file.find(norm_search)
    if idx != -1:
        # Return the original slice from the file so callers know the actual
        # characters they are replacing (important for style preservation).
        return file_content[idx: idx + len(search_string)]

    return None


class FileEditTool(Tool):
    """Perform an exact string replacement inside an existing file."""

    name = "Edit"
    description = (
        "Replace a specific string inside an existing file. "
        "The old_string must match exactly (including whitespace). "
        "Use replace_all=true to replace every occurrence. "
        "Line endings are preserved (CRLF or LF). "
        ".json files are validated after editing."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute path to the file to edit.",
            },
            "old_string": {
                "type": "string",
                "description": "The exact text to search for.",
            },
            "new_string": {
                "type": "string",
                "description": "The text to replace old_string with.",
            },
            "replace_all": {
                "type": "boolean",
                "description": "Replace all occurrences (default false).",
                "default": False,
            },
        },
        "required": ["file_path", "old_string", "new_string"],
    }
    dangerous = True

    def execute(
        self,
        input_data: Dict[str, Any],
        permission_manager=None,
    ) -> ToolResult:
        """Apply the replacement and return a unified diff of the change."""

        decision = self.check_permission(input_data, permission_manager)
        if decision == PermissionDecision.DENY:
            return ToolResult.error("Permission denied: file edit was blocked.")

        raw_path = input_data.get("file_path", "")
        old_string = input_data.get("old_string", "")
        new_string = input_data.get("new_string", "")
        replace_all = bool(input_data.get("replace_all", False))

        if not raw_path:
            return ToolResult.error("No file_path provided.")

        path = Path(raw_path).expanduser()

        if not path.exists():
            return ToolResult.error(
                f"File not found: {path}. Use the Write tool to create new files."
            )

        # File size guard — prevents OOM on huge binary/generated files.
        try:
            file_size = path.stat().st_size
        except OSError as exc:
            return ToolResult.error(f"Could not stat file: {exc}")

        if file_size > MAX_EDIT_FILE_SIZE:
            return ToolResult.error(
                f"File is too large to edit ({file_size:,} bytes > "
                f"{MAX_EDIT_FILE_SIZE:,} byte limit): {path}"
            )

        # Read raw bytes so we can detect and preserve the line-ending style.
        try:
            raw_bytes = path.read_bytes()
        except OSError as exc:
            return ToolResult.error(f"Could not read file: {exc}")

        # Detect line-ending style: CRLF if any \r\n present, else LF.
        uses_crlf = b"\r\n" in raw_bytes

        try:
            original = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            return ToolResult.error(f"File is not valid UTF-8: {exc}")

        # Normalise to LF internally for reliable matching.
        original_lf = original.replace("\r\n", "\n")

        # Normalise old_string / new_string to LF.
        old_lf = old_string.replace("\r\n", "\n")
        new_lf = new_string.replace("\r\n", "\n")

        # Try exact match first; fall back to curly-quote normalised match.
        # This handles the common case where the model outputs straight quotes
        # but the file contains curly/typographic quotes (or vice-versa).
        actual_old = _find_with_quote_fallback(original_lf, old_lf)
        if actual_old is None:
            return ToolResult.error(
                f"old_string not found in {path}. "
                "Make sure it matches the file content exactly (including whitespace)."
            )

        # Count occurrences of the *actual* (possibly quote-normalised) string.
        count = original_lf.count(actual_old)
        if count > 1 and not replace_all:
            return ToolResult.error(
                f"old_string appears {count} times in {path}. "
                "Provide a more specific string or set replace_all=true."
            )

        # Perform replacement on the LF-normalised content.
        if replace_all:
            modified_lf = original_lf.replace(actual_old, new_lf)
        else:
            modified_lf = original_lf.replace(actual_old, new_lf, 1)

        # JSON validation for .json files (settings, keybindings, tsconfig, etc.)
        if path.suffix.lower() == ".json":
            try:
                json.loads(modified_lf)
            except json.JSONDecodeError as exc:
                return ToolResult.error(
                    f"Edit would produce invalid JSON in {path}: {exc}\n"
                    "The file was NOT modified."
                )

        # Restore original line-ending style.
        if uses_crlf:
            modified_out = modified_lf.replace("\n", "\r\n")
        else:
            modified_out = modified_lf

        try:
            path.write_bytes(modified_out.encode("utf-8"))
        except OSError as exc:
            return ToolResult.error(f"Could not write file: {exc}")

        # Build a unified diff for Claude to review (always shown in LF form).
        diff_lines = list(
            difflib.unified_diff(
                original_lf.splitlines(keepends=True),
                modified_lf.splitlines(keepends=True),
                fromfile=f"a/{path.name}",
                tofile=f"b/{path.name}",
                n=3,   # 3 lines of context
            )
        )
        diff_text = "".join(diff_lines) if diff_lines else "(no diff — strings are identical)"

        replacements = count if replace_all else 1
        line_ending_note = " (CRLF preserved)" if uses_crlf else ""
        return ToolResult.ok(
            f"Replaced {replacements} occurrence(s) in {path}{line_ending_note}.\n\n{diff_text}"
        )
