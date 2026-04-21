"""
tools/bash_tool.py
==================
Executes arbitrary shell commands in the user's working directory.

This is the most powerful (and most dangerous) tool in the toolbox.
It is marked ``dangerous=True`` so the permission manager will prompt
before execution in ``default`` mode.

Security notes
--------------
* Commands run in a subprocess with the user's own privileges.
* There is an optional ``timeout`` parameter to avoid runaway processes.
* The permission manager should always be consulted before execution.

Improvements vs. earlier version
---------------------------------
* Tail-aware truncation — when output exceeds MAX_OUTPUT_CHARS the *last*
  N characters are kept (not the first), because the tail of command output
  (error messages, final test summary) is usually more useful than the head.
* Read-only classifier — ``is_readonly_command()`` identifies commands like
  ``git log``, ``cat``, ``ls``, ``echo`` that cannot mutate state, allowing
  callers and permission managers to skip the dangerous-flag prompt.
* Command-specific exit-code semantics — mirrors TypeScript's
  ``commandSemantics.ts``.  Commands like ``grep`` use exit code 1 to mean
  "no matches" (not an error); ``diff`` uses 1 to mean "files differ".
  This prevents Claude treating those as failures.
"""

from __future__ import annotations

import re
import subprocess
from typing import Any, Dict

from claude_code.tool import Tool, ToolResult, PermissionDecision

# Maximum output to return to Claude (prevents flooding the context window).
MAX_OUTPUT_CHARS = 50_000


# ---------------------------------------------------------------------------
# Command-specific exit-code semantics
# (mirrors TypeScript tools/BashTool/commandSemantics.ts)
# ---------------------------------------------------------------------------

def _heuristic_base_command(command: str) -> str:
    """
    Extract the *last* command in a pipeline (its exit code determines the
    pipeline exit status when pipefail is off, which is the default).

    Very approximate — only used for semantic classification, never security.
    """
    # Split on pipes, semicolons, &&, ||
    segments = re.split(r"[|;&]", command)
    last = segments[-1].strip() if segments else command.strip()
    # Strip leading env-var assignments (VAR=val cmd ...)
    tokens = last.split()
    for tok in tokens:
        if "=" not in tok or tok.startswith("-"):
            return tok.lower()
    return tokens[0].lower() if tokens else ""


def interpret_exit_code(
    command: str,
    exit_code: int,
    stdout: str,
    stderr: str,
) -> tuple[bool, str | None]:
    """
    Return ``(is_error, message)`` for *exit_code* given the *command*.

    Most commands: non-zero → error.
    Special cases:
    * ``grep``/``rg``: 1 = no matches (not an error); 2+ = error.
    * ``diff``:         1 = files differ (not an error); 2+ = error.
    * ``find``:         1 = partial (some dirs inaccessible); 2+ = error.
    * ``test``/``[``:  1 = condition false (not an error); 2+ = error.
    """
    base = _heuristic_base_command(command)

    if base in ("grep", "rg", "ag", "ack"):
        if exit_code == 0:
            return False, None
        if exit_code == 1:
            return False, "No matches found."   # informational, not an error
        return True, f"Command failed with exit code {exit_code}"

    if base == "diff":
        if exit_code == 0:
            return False, None
        if exit_code == 1:
            return False, "Files differ."        # informational
        return True, f"Command failed with exit code {exit_code}"

    if base == "find":
        if exit_code == 0:
            return False, None
        if exit_code == 1:
            return False, "Some directories were inaccessible."
        return True, f"Command failed with exit code {exit_code}"

    if base in ("test", "["):
        if exit_code <= 1:
            return False, None
        return True, f"Command failed with exit code {exit_code}"

    # Default: any non-zero is an error.
    if exit_code != 0:
        return True, f"Exit code: {exit_code}"
    return False, None

# Commands whose leading token is always read-only (cannot mutate state).
_READONLY_COMMANDS: frozenset = frozenset({
    "cat", "head", "tail", "less", "more", "bat",
    "ls", "ll", "la", "dir",
    "echo", "printf", "print",
    "pwd", "which", "type", "command",
    "git log", "git show", "git diff", "git status", "git blame",
    "git branch", "git remote", "git stash list", "git tag",
    "grep", "rg", "ag", "awk", "sed -n",
    "find", "locate", "wc",
    "env", "export -p", "set",
    "ps", "top", "htop", "jobs",
    "df", "du", "free",
    "uname", "hostname", "whoami", "id", "date",
    "python", "python3", "node", "ruby",   # read-only only when -c / script is safe
})

# Patterns that indicate a write/mutation even if the leading token looks safe.
_MUTATING_PATTERNS: tuple = (
    r"\brm\b", r"\bmv\b", r"\bcp\b", r"\bmkdir\b", r"\brmdir\b",
    r"\bchmod\b", r"\bchown\b", r"\bln\b",
    r"\bkill\b", r"\bpkill\b", r"\bkillall\b",
    r">\s*\S",          # output redirection  (> file)
    r">>\s*\S",         # append redirection  (>> file)
    r"\bdd\b",
    r"\bmkfs\b",
    r"\bsudo\b",
    r"\bsu\b",
    r"\bcurl\b.*\|\s*(?:sh|bash|zsh)",   # curl | sh (remote exec)
    r"\bwget\b.*-O\s*-",                 # wget pipe
    r"\bpip\s+install\b",
    r"\bnpm\s+install\b",
    r"\bapt\b",
    r"\bbrew\s+install\b",
)
_MUTATING_RE = re.compile("|".join(_MUTATING_PATTERNS))


def is_readonly_command(command: str) -> bool:
    """
    Return ``True`` if *command* is unlikely to mutate any system state.

    This is a conservative heuristic — false negatives (calling something
    safe that isn't) are possible; callers should use it to optimise the
    *fast path*, not as a security gate.
    """
    cmd = command.strip()
    # Any mutating pattern voids read-only status immediately.
    if _MUTATING_RE.search(cmd):
        return False
    # Check if the leading token (ignoring env-var assignments) is in the
    # known read-only set.
    tokens = cmd.split()
    if not tokens:
        return False
    # Skip over "VAR=value" prefixes.
    for tok in tokens:
        if "=" in tok and not tok.startswith("-"):
            continue
        leading = tok.lower()
        if leading in _READONLY_COMMANDS:
            return True
        # Multi-token read-only commands like "git log".
        if len(tokens) > 1:
            two = f"{leading} {tokens[1].lower()}"
            if two in _READONLY_COMMANDS:
                return True
        return False
    return False


class BashTool(Tool):
    """Execute a shell command and return its combined stdout + stderr."""

    name = "Bash"
    description = (
        "Execute a shell command in the current working directory. "
        "Returns combined stdout and stderr (tail-truncated if large). "
        "Use this to run tests, build code, inspect git history, etc."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute.",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default 120, max 600).",
                "default": 120,
            },
            "cwd": {
                "type": "string",
                "description": "Working directory override (default: session CWD).",
            },
        },
        "required": ["command"],
    }
    dangerous = True   # Always asks in 'default' permission mode

    def execute(
        self,
        input_data: Dict[str, Any],
        permission_manager=None,
    ) -> ToolResult:
        """
        Run *input_data["command"]* in a subprocess.

        Steps
        -----
        1. Check permission — deny immediately if blocked.
        2. Clamp the timeout to [1, 600].
        3. Run via ``subprocess.run`` with ``shell=True``.
        4. Capture output; tail-truncate if over MAX_OUTPUT_CHARS.
        5. Return exit-code information in the result.
        """
        # Step 1 — permission check.
        decision = self.check_permission(input_data, permission_manager)
        if decision == PermissionDecision.DENY:
            return ToolResult.error(
                f"Permission denied: '{input_data.get('command', '')}' was blocked."
            )

        command = input_data.get("command", "").strip()
        if not command:
            return ToolResult.error("No command provided.")

        # Step 2 — clamp timeout.
        raw_timeout = input_data.get("timeout", 120)
        try:
            timeout = max(1, min(int(raw_timeout), 600))
        except (TypeError, ValueError):
            timeout = 120

        cwd = input_data.get("cwd") or None

        # Step 3 — run.
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
            )
        except subprocess.TimeoutExpired:
            return ToolResult.error(
                f"Command timed out after {timeout} seconds: {command}"
            )
        except OSError as exc:
            return ToolResult.error(f"OS error running command: {exc}")

        # Step 4 — combine and tail-truncate output.
        # Keeping the *tail* is more useful than the head: error messages and
        # test summaries appear at the end of most command output streams.
        output = result.stdout + result.stderr
        total_chars = len(output)
        if total_chars > MAX_OUTPUT_CHARS:
            output = (
                f"[Output truncated — showing last {MAX_OUTPUT_CHARS} of "
                f"{total_chars} chars]\n\n"
                + output[-MAX_OUTPUT_CHARS:]
            )

        # Step 5 — apply command-specific exit-code semantics.
        is_error, message = interpret_exit_code(
            command, result.returncode, result.stdout, result.stderr
        )
        if is_error:
            if message and message not in output:
                output = f"{message}\n{output}"
            return ToolResult.error(output, exit_code=result.returncode)

        # Informational message (e.g. "No matches found.") — prepend if helpful.
        if message and not output.strip():
            output = message

        return ToolResult.ok(output or "(no output)", exit_code=result.returncode)
