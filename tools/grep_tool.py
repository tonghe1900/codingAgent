"""
tools/grep_tool.py
==================
Search file contents for a regex pattern.

Improvements vs. the original version
--------------------------------------
* Context lines  — ``-A``/``-B``/``-C`` (after / before / around each match).
* Pagination     — ``head_limit`` (max results) + ``offset`` (skip first N).
* Multiline mode — ``multiline=True`` lets ``.`` match newlines.
* File-type filter — ``type`` maps to ``rg --type`` (e.g. ``"py"``, ``"ts"``).
* VCS exclusion  — ``.git``, ``.svn``, ``.hg``, ``node_modules`` are always
  excluded from directory walks so they never pollute results.

Tries ripgrep (``rg``) when available; falls back to pure Python.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from claude_code.tool import Tool, ToolResult

# Default cap — keeps context window usage sane.
DEFAULT_HEAD_LIMIT = 250

# Directories that are always excluded from recursive searches.
VCS_DIRECTORIES = {".git", ".svn", ".hg", ".bzr", ".darcs", "node_modules",
                   "__pycache__", ".mypy_cache", ".pytest_cache", ".tox",
                   "dist", "build", ".eggs"}


class GrepTool(Tool):
    """Search file contents using a regular expression."""

    name = "Grep"
    description = (
        "Search file contents for a regular expression. "
        "Returns matching lines with file path and line number. "
        "Supports context lines (-A/-B/-C), pagination (head_limit/offset), "
        "multiline mode, and file-type filtering."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Regular expression to search for.",
            },
            "path": {
                "type": "string",
                "description": "Directory or file to search in (default: current directory).",
            },
            "glob": {
                "type": "string",
                "description": "File glob filter, e.g. '*.py' or '**/*.ts'.",
            },
            "type": {
                "type": "string",
                "description": "File type shorthand for rg --type (e.g. 'py', 'ts', 'js', 'rust').",
            },
            "-i": {
                "type": "boolean",
                "description": "Case-insensitive search (default false).",
                "default": False,
            },
            "-A": {
                "type": "integer",
                "description": "Lines of trailing context after each match.",
            },
            "-B": {
                "type": "integer",
                "description": "Lines of leading context before each match.",
            },
            "-C": {
                "type": "integer",
                "description": "Lines of context both before and after each match.",
            },
            "multiline": {
                "type": "boolean",
                "description": "Enable multiline mode (dot matches newlines). Default false.",
                "default": False,
            },
            "output_mode": {
                "type": "string",
                "enum": ["content", "files_with_matches", "count"],
                "description": "How to report results (default 'content').",
                "default": "content",
            },
            "head_limit": {
                "type": "integer",
                "description": f"Max output lines to return (default {DEFAULT_HEAD_LIMIT}, 0 = unlimited).",
                "default": DEFAULT_HEAD_LIMIT,
            },
            "offset": {
                "type": "integer",
                "description": "Skip the first N lines/matches before applying head_limit.",
                "default": 0,
            },
            "-n": {
                "type": "boolean",
                "description": "Show line numbers (default true).",
                "default": True,
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
        pattern = input_data.get("pattern", "").strip()
        if not pattern:
            return ToolResult.error("No pattern provided.")

        root_str = input_data.get("path") or os.getcwd()
        root = Path(root_str).expanduser().resolve()

        glob_filter: Optional[str] = input_data.get("glob")
        file_type:   Optional[str] = input_data.get("type")
        case_insens: bool = bool(input_data.get("-i", False))
        multiline:   bool = bool(input_data.get("multiline", False))
        output_mode: str  = input_data.get("output_mode", "content")

        # Context lines
        after_ctx:  Optional[int] = input_data.get("-A")
        before_ctx: Optional[int] = input_data.get("-B")
        around_ctx: Optional[int] = input_data.get("-C")

        # Pagination
        raw_limit  = input_data.get("head_limit", DEFAULT_HEAD_LIMIT)
        head_limit = int(raw_limit) if raw_limit else DEFAULT_HEAD_LIMIT
        offset     = max(0, int(input_data.get("offset", 0) or 0))

        # Try ripgrep first.
        rg = self._try_ripgrep(
            pattern, root, glob_filter, file_type, case_insens, multiline,
            output_mode, after_ctx, before_ctx, around_ctx, head_limit, offset,
        )
        if rg is not None:
            return rg

        # Python fallback.
        return self._python_grep(
            pattern, root, glob_filter, case_insens, multiline,
            output_mode, after_ctx, before_ctx, around_ctx, head_limit, offset,
        )

    # ------------------------------------------------------------------
    # ripgrep implementation
    # ------------------------------------------------------------------

    @staticmethod
    def _try_ripgrep(
        pattern: str, root: Path,
        glob_filter: Optional[str], file_type: Optional[str],
        case_insens: bool, multiline: bool,
        output_mode: str,
        after_ctx: Optional[int], before_ctx: Optional[int], around_ctx: Optional[int],
        head_limit: int, offset: int,
    ) -> Optional[ToolResult]:
        """Run rg and return ToolResult, or None if rg is unavailable."""
        try:
            cmd = ["rg", "--no-heading", "-n", pattern, str(root)]

            # Flags
            if case_insens:
                cmd.append("-i")
            if multiline:
                cmd += ["-U", "--multiline-dotall"]

            # File filters
            if glob_filter:
                cmd += ["--glob", glob_filter]
            if file_type:
                cmd += ["--type", file_type]

            # Always exclude VCS/build directories
            for excl in VCS_DIRECTORIES:
                cmd += ["--glob", f"!{excl}"]

            # Context
            if around_ctx is not None:
                cmd += ["-C", str(around_ctx)]
            elif after_ctx is not None or before_ctx is not None:
                if after_ctx is not None:
                    cmd += ["-A", str(after_ctx)]
                if before_ctx is not None:
                    cmd += ["-B", str(before_ctx)]

            # Output mode
            if output_mode == "files_with_matches":
                cmd.append("-l")
            elif output_mode == "count":
                cmd.append("-c")

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            lines = result.stdout.splitlines()

            if not lines and result.returncode != 0:
                return ToolResult.ok("No matches found.")

            # Pagination
            total = len(lines)
            if offset:
                lines = lines[offset:]
            if head_limit > 0:
                truncated = len(lines) > head_limit
                lines = lines[:head_limit]
            else:
                truncated = False

            output = "\n".join(lines)
            if truncated:
                output += (
                    f"\n\n[Showing results {offset + 1}–{offset + len(lines)} of {total}. "
                    f"Use offset={offset + head_limit} to see more.]"
                )
            return ToolResult.ok(output or "No matches found.", total_matches=total)

        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None  # rg not available

    # ------------------------------------------------------------------
    # Pure-Python fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _python_grep(
        pattern: str, root: Path,
        glob_filter: Optional[str], case_insens: bool, multiline: bool,
        output_mode: str,
        after_ctx: Optional[int], before_ctx: Optional[int], around_ctx: Optional[int],
        head_limit: int, offset: int,
    ) -> ToolResult:
        flags  = 0
        if case_insens:
            flags |= re.IGNORECASE
        if multiline:
            flags |= re.MULTILINE | re.DOTALL

        try:
            compiled = re.compile(pattern, flags)
        except re.error as exc:
            return ToolResult.error(f"Invalid regex: {exc}")

        # Resolve context window size.
        ctx_before = (around_ctx or 0) + (before_ctx or 0)
        ctx_after  = (around_ctx or 0) + (after_ctx  or 0)

        # Collect files — skip VCS directories.
        if root.is_file():
            files: List[Path] = [root]
        else:
            glob_pat = glob_filter or "**/*"
            files = [
                p for p in root.glob(glob_pat)
                if p.is_file() and not any(part in VCS_DIRECTORIES for part in p.parts)
            ]

        all_lines:    List[str] = []
        files_matched: List[str] = []
        total_count   = 0

        for file_path in files:
            try:
                text = file_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            file_lines = text.splitlines()
            file_hit   = False
            hit_indices: List[int] = []

            for lineno, line in enumerate(file_lines):
                if compiled.search(line):
                    hit_indices.append(lineno)
                    total_count += 1
                    file_hit = True

            if file_hit:
                files_matched.append(str(file_path))

            if output_mode == "content":
                # Expand context around each hit.
                shown: set = set()
                for idx in hit_indices:
                    lo = max(0, idx - ctx_before)
                    hi = min(len(file_lines) - 1, idx + ctx_after)
                    for i in range(lo, hi + 1):
                        if i not in shown:
                            shown.add(i)
                            prefix = ">" if i == idx else " "
                            all_lines.append(
                                f"{file_path}:{i + 1}:{prefix}{file_lines[i]}"
                            )

        # Build output from mode.
        if output_mode == "files_with_matches":
            raw_output = files_matched
        elif output_mode == "count":
            raw_output = [f"Total matches: {total_count}"]
        else:
            raw_output = all_lines

        # Pagination
        total_lines = len(raw_output)
        if offset:
            raw_output = raw_output[offset:]
        truncated = head_limit > 0 and len(raw_output) > head_limit
        if head_limit > 0:
            raw_output = raw_output[:head_limit]

        output = "\n".join(raw_output)
        if truncated:
            output += (
                f"\n\n[Showing {offset + 1}–{offset + len(raw_output)} of {total_lines}. "
                f"Use offset={offset + head_limit} to see more.]"
            )
        if not output:
            output = "No matches found."

        return ToolResult.ok(output, total_matches=total_count)
