"""
permissions/permission_manager.py
==================================
Controls whether a tool call is allowed to proceed.

Three modes (matching the TypeScript source):

* ``default`` — dangerous tools prompt the user; safe tools auto-allow.
* ``bypass``  — every tool is allowed without prompting (CI/automation).
* ``auto``    — a simple heuristic classifier decides; falls back to prompt.

The manager also keeps a ``DenialLog`` so the query engine can surface a
summary of what was blocked during a session.

Improvements vs. earlier version
---------------------------------
* Dangerous-file / directory guard — paths to sensitive config files
  (.gitconfig, .bashrc, .env, .ssh/*, etc.) are blocked for Write/Edit
  operations regardless of mode.
* Path-based allow/deny rules — callers can register glob-style path
  patterns to unconditionally allow or block file operations.
"""

from __future__ import annotations

import fnmatch
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from claude_code.tool import PermissionDecision

if TYPE_CHECKING:
    from claude_code.tool import Tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dangerous files / directories
# ---------------------------------------------------------------------------

# Files that should never be written or edited by the AI without explicit
# user approval.  Matched case-insensitively against the *basename*.
DANGEROUS_FILES: frozenset = frozenset({
    ".gitconfig", ".gitignore_global",
    ".bashrc", ".bash_profile", ".bash_logout",
    ".zshrc", ".zprofile", ".zshenv",
    ".profile",
    ".env", ".env.local", ".env.production", ".env.staging",
    "id_rsa", "id_ed25519", "id_ecdsa", "id_dsa",
    "authorized_keys", "known_hosts",
    "passwd", "shadow", "sudoers",
    ".netrc", ".npmrc", ".pypirc",
    "credentials",           # AWS credentials file
    "config",                # can be ~/.ssh/config
})

# Directory names that are always blocked for recursive file-write operations.
DANGEROUS_DIRECTORIES: frozenset = frozenset({
    ".ssh",
    ".gnupg",
    ".aws",
    ".kube",
    ".docker",
})

# Tools that perform file writes (Write or Edit).
_FILE_WRITE_TOOLS: frozenset = frozenset({"Write", "Edit"})


# ---------------------------------------------------------------------------
# Denial log entry
# ---------------------------------------------------------------------------

@dataclass
class DenialEntry:
    """One record of a denied tool call."""
    tool_name: str
    input_summary: str   # Short human-readable description of what was blocked
    reason: str          # "denied_by_rule" | "user_declined" | "auto_blocked"


# ---------------------------------------------------------------------------
# Permission manager
# ---------------------------------------------------------------------------

class PermissionManager:
    """
    Stateful permission evaluator for one session.

    Parameters
    ----------
    mode:
        One of ``"default"``, ``"bypass"``, or ``"auto"``.
    allowed_tools:
        Tool names that are always permitted (explicit allow-list).
    denied_tools:
        Tool names that are always blocked (explicit deny-list).
    interactive:
        When ``True`` the manager may print a prompt and read stdin.
        Set to ``False`` in non-interactive / test contexts.
    allowed_paths:
        Glob patterns for file paths that are unconditionally allowed for
        Write/Edit operations (e.g. ``["src/**", "tests/**"]``).
    denied_paths:
        Glob patterns for file paths that are unconditionally denied for
        Write/Edit operations (e.g. ``["/etc/**", "~/.ssh/**"]``).
    """

    def __init__(
        self,
        mode: str = "default",
        allowed_tools: Optional[List[str]] = None,
        denied_tools: Optional[List[str]] = None,
        interactive: bool = True,
        allowed_paths: Optional[List[str]] = None,
        denied_paths: Optional[List[str]] = None,
        prompt_fn: Optional[Callable[[str, str, str], bool]] = None,
    ) -> None:
        """
        Parameters
        ----------
        prompt_fn:
            Optional callable used to ask the user for permission.
            Signature: ``(tool_name: str, input_summary: str, full_input: str) -> bool``
            Returns ``True`` to allow, ``False`` to deny.
            When omitted the manager falls back to plain ``print``/``input``.
            Inject a rich-based prompt from ``main.py`` for better UX.
        """
        self.mode = mode
        self._allowed: set[str] = set(allowed_tools or [])
        self._denied:  set[str] = set(denied_tools  or [])
        self._allowed_paths: List[str] = list(allowed_paths or [])
        self._denied_paths:  List[str] = list(denied_paths  or [])
        self.interactive = interactive
        self._prompt_fn = prompt_fn
        self.denial_log: List[DenialEntry] = []

    # ------------------------------------------------------------------
    # Core check
    # ------------------------------------------------------------------

    def check(
        self,
        tool: "Tool",
        input_data: Dict[str, Any],
    ) -> PermissionDecision:
        """
        Return the :class:`PermissionDecision` for executing *tool* with
        *input_data*.

        Side effects: appends to :attr:`denial_log` on DENY, and may print
        an interactive prompt on ASK.
        """
        name = tool.name

        # Explicit deny-list overrides everything.
        if name in self._denied:
            self._log_denial(name, input_data, "denied_by_rule")
            return PermissionDecision.DENY

        # Dangerous-file guard for write/edit tools (checked before allow-list
        # so it can never be accidentally bypassed by an allow-list entry).
        if name in _FILE_WRITE_TOOLS:
            path_check = self._check_path_safety(tool, input_data)
            if path_check == PermissionDecision.DENY:
                return PermissionDecision.DENY

        # Path-based deny patterns (user-configured).
        if name in _FILE_WRITE_TOOLS:
            path_str = str(input_data.get("file_path", ""))
            for pat in self._denied_paths:
                if fnmatch.fnmatch(path_str, pat):
                    self._log_denial(name, input_data, "denied_by_path_rule")
                    return PermissionDecision.DENY

        # Path-based allow patterns (user-configured) — short-circuit.
        if name in _FILE_WRITE_TOOLS and self._allowed_paths:
            path_str = str(input_data.get("file_path", ""))
            for pat in self._allowed_paths:
                if fnmatch.fnmatch(path_str, pat):
                    return PermissionDecision.ALLOW

        # Explicit allow-list also overrides mode logic.
        if name in self._allowed:
            return PermissionDecision.ALLOW

        # Bypass mode: allow everything not explicitly denied.
        if self.mode == "bypass":
            return PermissionDecision.ALLOW

        # Default / auto: delegate to mode-specific logic.
        if self.mode == "auto":
            return self._auto_check(tool, input_data)

        # Default mode: dangerous tools ask the user.
        return self._default_check(tool, input_data)

    # ------------------------------------------------------------------
    # Dangerous-path guard
    # ------------------------------------------------------------------

    def _check_path_safety(
        self,
        tool: "Tool",
        input_data: Dict[str, Any],
    ) -> PermissionDecision:
        """
        Return DENY when *input_data* targets a sensitive system file or
        directory, regardless of the current permission mode.

        This prevents Claude from accidentally overwriting ``~/.bashrc``,
        ``~/.ssh/id_rsa``, or similar critical configuration files.
        """
        raw_path = input_data.get("file_path", "")
        if not raw_path:
            return PermissionDecision.ALLOW

        try:
            path = Path(raw_path).expanduser().resolve()
        except (OSError, ValueError):
            return PermissionDecision.ALLOW

        # Check every component of the path for dangerous directory names.
        for part in path.parts:
            if part in DANGEROUS_DIRECTORIES:
                self._log_denial(tool.name, input_data, "dangerous_path")
                return PermissionDecision.DENY

        # Check the basename against the dangerous-file set.
        if path.name.lower() in DANGEROUS_FILES:
            self._log_denial(tool.name, input_data, "dangerous_path")
            return PermissionDecision.DENY

        return PermissionDecision.ALLOW

    # ------------------------------------------------------------------
    # Mode-specific logic
    # ------------------------------------------------------------------

    def _default_check(
        self,
        tool: "Tool",
        input_data: Dict[str, Any],
    ) -> PermissionDecision:
        """
        In ``default`` mode:
        * Safe tools (dangerous=False) are auto-allowed.
        * Dangerous tools prompt the user if interactive, else deny.
        """
        if not tool.dangerous:
            return PermissionDecision.ALLOW

        if not self.interactive:
            self._log_denial(tool.name, input_data, "auto_blocked")
            return PermissionDecision.DENY

        # Interactive prompt — use injected prompt_fn when available so the
        # caller (e.g. main.py) can render a rich-formatted prompt.
        #
        # prompt_fn returns one of three strings:
        #   "yes"  — allow this call only
        #   "no"   — deny this call
        #   "all"  — allow this call AND all subsequent dangerous calls
        #            (switches this session to bypass mode)
        summary    = self._summarise_input(input_data)
        full_input = str(input_data)

        if self._prompt_fn is not None:
            decision = self._prompt_fn(tool.name, summary, full_input)
        else:
            print(f"\n[Permission] Tool '{tool.name}' wants to run: {summary}")
            while True:
                raw = input("Allow? [y/n/a] (a=allow all future) ").strip().lower()
                if raw in ("y", "yes"):
                    decision = "yes"
                    break
                elif raw in ("n", "no", ""):
                    decision = "no"
                    break
                elif raw == "a":
                    decision = "all"
                    break
                # Any other input: re-prompt (infinite wait for valid choice)

        if decision == "all":
            # Switch to bypass for the rest of the session.
            self.mode = "bypass"
            logger.info("Permission mode switched to bypass — all future tool calls allowed.")
            return PermissionDecision.ALLOW

        if decision == "yes":
            return PermissionDecision.ALLOW

        self._log_denial(tool.name, input_data, "user_declined")
        return PermissionDecision.DENY

    def _auto_check(
        self,
        tool: "Tool",
        input_data: Dict[str, Any],
    ) -> PermissionDecision:
        """
        Simple heuristic classifier for ``auto`` mode.

        Blocks patterns that are commonly destructive; allows everything
        else.  This is intentionally conservative — a production classifier
        would use an ML model.
        """
        # Always allow read-only tools.
        READ_ONLY = {"Read", "Glob", "Grep", "WebFetch", "WebSearch"}
        if tool.name in READ_ONLY:
            return PermissionDecision.ALLOW

        # Block clearly destructive bash patterns.
        if tool.name == "Bash":
            cmd = str(input_data.get("command", ""))
            dangerous_patterns = ["rm -rf", "mkfs", "dd if=", "> /dev/", ":(){ :|:& };:"]
            for pat in dangerous_patterns:
                if pat in cmd:
                    self._log_denial(tool.name, input_data, "auto_blocked")
                    return PermissionDecision.DENY

        # Default to allow for everything else.
        return PermissionDecision.ALLOW

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log_denial(
        self,
        tool_name: str,
        input_data: Dict[str, Any],
        reason: str,
    ) -> None:
        entry = DenialEntry(
            tool_name=tool_name,
            input_summary=self._summarise_input(input_data),
            reason=reason,
        )
        self.denial_log.append(entry)
        logger.debug("Tool '%s' denied (%s): %s", tool_name, reason, entry.input_summary)

    @staticmethod
    def _summarise_input(input_data: Dict[str, Any]) -> str:
        """Return a short (≤120-char) human-readable summary of *input_data*."""
        text = str(input_data)
        return text[:120] + ("…" if len(text) > 120 else "")

    def denial_summary(self) -> str:
        """Human-readable summary of all denials this session."""
        if not self.denial_log:
            return "No tool calls were denied this session."
        lines = ["Denied tool calls:"]
        for entry in self.denial_log:
            lines.append(f"  [{entry.reason}] {entry.tool_name}: {entry.input_summary}")
        return "\n".join(lines)
