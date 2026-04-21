"""
context.py
==========
Builds the system prompt sent to Claude at the start of every request.

The system prompt is assembled from several parts:
1. Core identity / capability description (always present).
2. Working-directory information and current date/time.
3. Git repository status (branch, dirty files, recent commits) — injected
   when the CWD is inside a git repo.
4. CLAUDE.md content (if a file exists in the CWD or any parent directory).
5. Any extra instructions supplied via settings.

Keeping this logic here makes it easy to test and extend independently.
"""

from __future__ import annotations

import datetime
import logging
import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLAUDE.md loader
# ---------------------------------------------------------------------------

def _load_claude_md(cwd: Optional[Path] = None) -> str:
    """
    Return the content of the first CLAUDE.md (or claude.md) found by
    searching from *cwd* up to the filesystem root, then ``~/.claude/``.

    Returns an empty string when no file is found.
    """
    search_dirs = []

    start = Path(cwd or os.getcwd()).resolve()
    current = start
    while True:
        search_dirs.append(current)
        parent = current.parent
        if parent == current:
            break
        current = parent

    search_dirs.append(Path.home() / ".claude")

    for directory in search_dirs:
        for filename in ("CLAUDE.md", "claude.md"):
            candidate = directory / filename
            if candidate.is_file():
                try:
                    content = candidate.read_text(encoding="utf-8")
                    return f"\n\n---\nProject instructions from {candidate}:\n{content}"
                except OSError:
                    pass

    return ""


# ---------------------------------------------------------------------------
# Git status
# ---------------------------------------------------------------------------

def _run_git(args: list[str], cwd: Path) -> str:
    """Run a git command and return stdout, or '' on any error."""
    try:
        result = subprocess.run(
            ["git"] + args,
            capture_output=True, text=True, timeout=5, cwd=str(cwd)
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except (OSError, subprocess.TimeoutExpired):
        return ""


MAX_STATUS_CHARS = 2000  # Truncate git status at this length to keep prompt short.


def get_git_context(cwd: Path) -> str:
    """
    Return a short multi-line git status block to inject into the system
    prompt, or an empty string when the CWD is not inside a git repo.

    Includes:
    * Current branch name.
    * Main/default branch name.
    * Short status of changed files (``git status --short``), truncated at
      MAX_STATUS_CHARS so massive repos don't bloat the system prompt.
    * The 5 most-recent commit one-liners.
    """
    # Quick check: is this a git repo at all?
    root = _run_git(["rev-parse", "--show-toplevel"], cwd)
    if not root:
        return ""

    branch    = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd)
    status    = _run_git(["status", "--short"], cwd)
    log       = _run_git(["log", "--oneline", "-5"], cwd)
    git_user  = _run_git(["config", "user.name"], cwd)

    # Truncate status output so large repos don't flood the system prompt.
    if len(status) > MAX_STATUS_CHARS:
        status = (
            status[:MAX_STATUS_CHARS]
            + '\n... (truncated because it exceeds 2k characters. '
            'If you need more information, run "git status" using the Bash tool)'
        )

    # Try to detect the main/default branch.
    for candidate in ("main", "master", "develop"):
        if _run_git(["rev-parse", "--verify", candidate], cwd):
            main_branch = candidate
            break
    else:
        main_branch = "main"

    parts = [
        "\n---\nThis is the git status at the start of the conversation. "
        "Note that this status is a snapshot in time, and will not update during the conversation.",
    ]
    if branch:
        parts.append(f"Current branch: {branch}")
        parts.append(f"Main branch (you will usually use this for PRs): {main_branch}")
    if git_user:
        parts.append(f"Git user: {git_user}")
    if status:
        parts.append(f"Status:\n{status}")
    else:
        parts.append("Status:\n(clean)")
    if log:
        parts.append(f"Recent commits:\n{log}")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Runtime / interpreter detection
# ---------------------------------------------------------------------------

def _probe_binary(binary: str) -> Optional[str]:
    """
    Return the version string reported by *binary* (e.g. "Python 3.13.3"),
    or None if the binary is not found / errors out.
    """
    path = shutil.which(binary)
    if not path:
        return None
    try:
        result = subprocess.run(
            [binary, "--version"],
            capture_output=True, text=True, timeout=3,
        )
        # Some old binaries print to stderr; try stdout first.
        raw = (result.stdout.strip() or result.stderr.strip())
        return f"{path}  ({raw})" if raw else path
    except (OSError, subprocess.TimeoutExpired):
        return None


@lru_cache(maxsize=1)
def get_runtime_context() -> str:
    """
    Probe the common Python (and Node) interpreter names and return a block
    like:

        Python interpreters on this system:
          python3.13  /usr/local/bin/python3.13  (Python 3.13.3)
          python3     /usr/local/bin/python3     (Python 3.13.3)
          python      /usr/bin/python            (Python 2.7.18)  ← WARNING: Python 2

        WARNING: `python` resolves to Python 2.7.18 — NEVER use it for
        Python 3 projects.  Always run tests with `python3` or `python3.13`.

    Returns an empty string if nothing interesting is detected (e.g. only
    python3 is present, no python2 alias exists).
    """
    candidates = [
        "python3.14", "python3.13", "python3.12", "python3.11",
        "python3.10", "python3.9", "python3", "python2", "python",
    ]

    found: list[tuple[str, str]] = []  # (name, "path  (version)")
    for name in candidates:
        info = _probe_binary(name)
        if info:
            found.append((name, info))

    if not found:
        return ""

    lines = ["Python interpreters on this system:"]
    python2_alias: Optional[str] = None  # version string if `python` → Py2

    for name, info in found:
        suffix = ""
        # Detect `python` or `python2` resolving to Python 2.x
        if ("python2" in name or name == "python") and " 2." in info:
            suffix = "  ← WARNING: Python 2 (EOL)"
            if name == "python":
                # Extract short version for the warning message below
                try:
                    python2_alias = info.split("(")[1].rstrip(")")
                except IndexError:
                    python2_alias = "Python 2"
        lines.append(f"  {name:<14} {info}{suffix}")

    block = "\n".join(lines)

    if python2_alias:
        block += (
            f"\n\nWARNING: `python` resolves to {python2_alias} — "
            "NEVER use it for Python 3 projects.  "
            "Always invoke tests with `python3` or a versioned binary like `python3.13`."
        )

    # Check for Node.js too
    node_info = _probe_binary("node")
    if node_info:
        block += f"\n\nNode.js: node  {node_info}"

    return block


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------

_CORE_SYSTEM_PROMPT = """\
You are Claude Code, an AI-powered interactive development assistant running
in a terminal.  You help engineers read, write, edit, and reason about code.

Core capabilities
-----------------
* Execute shell commands (Bash) to build, test, and explore the system.
* Read, write, and edit files (Read, Write, Edit).
* Search the codebase (Glob, Grep) for files and patterns.
* Fetch web pages and search the web (WebFetch, WebSearch).
* Manage tasks (TodoWrite) and configuration (Config).

Working style
-------------
* Prefer small, focused changes over large rewrites.
* Always read a file before editing it.
* Explain what you are doing and why when it is not obvious.
* Ask for clarification rather than guessing when requirements are ambiguous.
* Never commit, push, or deploy without explicit user permission.

Runtime and environment rules  ← follow these before touching any code
-----------------------------------------------------------------------
1. VERIFY THE RUNTIME FIRST.
   Before diagnosing a syntax error or "module not found" error, confirm
   you are using the correct tool version.  A wrong binary is the most
   common cause of mysterious failures.

2. PYTHON VERSION.
   - `python` on macOS and many Linux systems resolves to Python 2 (EOL).
     ALWAYS use `python3` or an explicit version such as `python3.13`.
   - Check once per session: `python3 --version` (or `python3.13 --version`).
   - Never run Python 3 test suites with `python`; use `python3 -m pytest`.
   - Use the same binary for every command in a session:
       BAD:  python setup.py install && python3 -m pytest
       GOOD: python3 setup.py install && python3 -m pytest

3. DIAGNOSE ENVIRONMENT BEFORE REWRITING CODE.
   When a command fails unexpectedly, run a minimal diagnostic FIRST:
       # Wrong Python?
       python --version && python3 --version
       # Missing dependency?
       python3 -c "import requests"
       # Wrong directory?
       pwd && ls
   Only modify code AFTER confirming the environment is correct.

4. SHEBANG LINES.
   Always write `#!/usr/bin/env python3` in new Python scripts — never
   `#!/usr/bin/env python` or `#!/usr/bin/env python2`.

5. OTHER RUNTIMES (Node, Ruby, Java, …).
   Apply the same rule: check `node --version`, `ruby --version`, etc.
   before assuming the code is wrong when a test fails.
"""


def build_system_prompt(
    cwd: Optional[str | Path] = None,
    extra: str = "",
    append: str = "",
    custom: Optional[str] = None,
    include_git: bool = True,
    include_date: bool = True,
) -> str:
    """
    Assemble and return the full system prompt string.

    Parameters
    ----------
    cwd:
        Working directory to report and to search for CLAUDE.md / git repo.
    extra:
        Additional instructions from settings (``system_prompt_extra``),
        appended after the core prompt.
    append:
        Additional text appended last (``append_system_prompt``).
    custom:
        If provided, *replaces* the core prompt entirely.
    include_git:
        Whether to inject git status (default True).
    include_date:
        Whether to inject the current date/time (default True).
    """
    cwd_path = Path(cwd or os.getcwd()).resolve()

    if custom:
        # Custom prompt replaces core; still append git / CLAUDE.md / extra.
        parts = [custom.strip()]
    else:
        parts = [_CORE_SYSTEM_PROMPT]

    # Current date/time.
    if include_date:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        parts.append(f"\nCurrent date/time: {now}")

    # Working directory.
    parts.append(f"Current working directory: {cwd_path}")

    # Runtime context (detected Python/Node versions).
    runtime_ctx = get_runtime_context()
    if runtime_ctx:
        parts.append(f"\n---\nRuntime environment:\n{runtime_ctx}")

    # Git context.
    if include_git:
        git_ctx = get_git_context(cwd_path)
        if git_ctx:
            parts.append(git_ctx)

    # CLAUDE.md file content.
    claude_md = _load_claude_md(cwd_path)
    if claude_md:
        parts.append(claude_md)

    # Extra instructions from settings.
    if extra.strip():
        parts.append(f"\n\n---\nAdditional instructions:\n{extra.strip()}")

    # Append system prompt (last, highest specificity).
    if append.strip():
        parts.append(f"\n{append.strip()}")

    return "\n".join(parts)
