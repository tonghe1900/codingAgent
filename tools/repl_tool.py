"""
tools/repl_tool.py
==================
A persistent interactive shell session tool.

Unlike ``BashTool`` which spawns a fresh process for every command, ``REPLTool``
keeps a long-lived ``bash`` process alive between calls so that:

* Environment variables set in one call are visible in the next.
* ``cd`` (directory changes) persist across calls.
* Shell functions / aliases defined once stay defined.
* Heavy initialisation (e.g. ``nvm use``, ``conda activate``) runs once.

Design
------
A persistent ``bash`` subprocess is kept in a per-session registry keyed by
``session_id``.  A background reader thread drains stdout into a queue.  A
unique sentinel string is echoed after each command so the caller knows
exactly when output ends.  Stderr is merged into stdout via ``2>&1``.

Limitations
-----------
* Interactive programs that read from stdin (``vim``, ``python`` REPL, etc.)
  will hang — use ``BashTool`` for those.
* The shell process is killed when the Python process exits or when the
  caller sends ``"__reset__"`` as the command.
* Maximum output per call is MAX_OUTPUT_CHARS (50 000 chars), tail-truncated.
"""

from __future__ import annotations

import atexit
import os
import queue
import subprocess
import threading
import uuid
from typing import Any, Dict, Optional

from claude_code.tool import Tool, ToolResult, PermissionDecision

MAX_OUTPUT_CHARS = 50_000
_DEFAULT_TIMEOUT = 30.0   # seconds
_MAX_TIMEOUT     = 300.0  # 5 minutes


# ---------------------------------------------------------------------------
# Per-session shell
# ---------------------------------------------------------------------------

class _PersistentShell:
    """
    Wraps a long-lived bash subprocess.

    A background daemon thread reads every line from stdout and pushes it
    into a ``queue.Queue``.  ``run()`` sends a command, then drains the queue
    until it sees the sentinel line.
    """

    _SENTINEL_BASE = "__REPL_END__"

    def __init__(self, cwd: Optional[str] = None) -> None:
        env = os.environ.copy()
        self._process = subprocess.Popen(
            ["bash", "--norc", "--noprofile"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,   # merge stderr into stdout
            env=env,
            cwd=cwd,
        )
        self._queue: queue.Queue = queue.Queue()
        self._sentinel = f"{self._SENTINEL_BASE}_{uuid.uuid4().hex}"
        self._lock = threading.Lock()

        # Background reader daemon — drains stdout into the queue.
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read_loop(self) -> None:
        """Background thread: read lines from the shell's stdout."""
        try:
            for raw_line in self._process.stdout:
                line = raw_line.decode("utf-8", errors="replace").rstrip("\n")
                self._queue.put(line)
        except (OSError, ValueError):
            pass
        finally:
            self._queue.put(None)  # EOF sentinel

    def run(self, command: str, timeout: float = _DEFAULT_TIMEOUT) -> str:
        """
        Execute *command* in the persistent shell and return its output.

        Raises ``TimeoutError`` if the command does not finish within *timeout*
        seconds.  Raises ``OSError`` if the shell has died.
        """
        if self._process.poll() is not None:
            raise OSError("Shell process has already exited.")

        with self._lock:
            # Write command + sentinel echo to stdin.
            full_cmd = (
                f"{command} 2>&1\n"
                f"echo '{self._sentinel}'\n"
            )
            try:
                self._process.stdin.write(full_cmd.encode("utf-8"))
                self._process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                raise OSError(f"Could not write to shell stdin: {exc}") from exc

            # Collect lines until we see the sentinel.
            output_lines: list[str] = []
            while True:
                try:
                    line = self._queue.get(timeout=timeout)
                except queue.Empty:
                    raise TimeoutError(
                        f"Command timed out after {timeout}s."
                    )
                if line is None:
                    raise OSError("Shell process closed stdout unexpectedly.")
                if line == self._sentinel:
                    break
                output_lines.append(line)

        return "\n".join(output_lines)

    def terminate(self) -> None:
        """Kill the underlying shell process."""
        try:
            self._process.stdin.close()
        except Exception:
            pass
        try:
            self._process.terminate()
            self._process.wait(timeout=5)
        except Exception:
            try:
                self._process.kill()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Session registry
# ---------------------------------------------------------------------------

_shells: Dict[str, _PersistentShell] = {}
_shells_lock = threading.Lock()


def _get_or_create_shell(session_id: str, cwd: Optional[str]) -> _PersistentShell:
    with _shells_lock:
        shell = _shells.get(session_id)
        if shell is None or shell._process.poll() is not None:
            _shells[session_id] = _PersistentShell(cwd=cwd)
    return _shells[session_id]


def _reset_shell(session_id: str, cwd: Optional[str]) -> _PersistentShell:
    with _shells_lock:
        old = _shells.pop(session_id, None)
        if old:
            old.terminate()
        _shells[session_id] = _PersistentShell(cwd=cwd)
    return _shells[session_id]


def _cleanup_all_shells() -> None:
    """Kill all open shells at process exit."""
    with _shells_lock:
        for shell in list(_shells.values()):
            shell.terminate()
        _shells.clear()


atexit.register(_cleanup_all_shells)


# ---------------------------------------------------------------------------
# REPLTool
# ---------------------------------------------------------------------------

class REPLTool(Tool):
    """
    Execute commands in a persistent bash shell session.

    Environment variables, working-directory changes, and shell functions
    set in one call persist to subsequent calls with the same session_id.
    Send ``command="__reset__"`` to kill and restart the shell.
    """

    name = "REPL"
    description = (
        "Execute a shell command in a *persistent* bash session. "
        "Unlike Bash, environment variables, cd, and shell functions persist "
        "between calls sharing the same session_id. "
        "Send command='__reset__' to restart the session."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command to run, or '__reset__' to restart the session.",
            },
            "session_id": {
                "type": "string",
                "description": "Identifier for the persistent session (default 'default').",
                "default": "default",
            },
            "timeout": {
                "type": "number",
                "description": (
                    f"Per-command timeout in seconds "
                    f"(default {_DEFAULT_TIMEOUT}, max {_MAX_TIMEOUT})."
                ),
                "default": _DEFAULT_TIMEOUT,
            },
            "cwd": {
                "type": "string",
                "description": "Initial working directory for a NEW session (ignored if session already exists).",
            },
        },
        "required": ["command"],
    }
    dangerous = True

    def execute(
        self,
        input_data: Dict[str, Any],
        permission_manager=None,
    ) -> ToolResult:
        decision = self.check_permission(input_data, permission_manager)
        if decision == PermissionDecision.DENY:
            return ToolResult.error("Permission denied: REPL command was blocked.")

        command = input_data.get("command", "").strip()
        if not command:
            return ToolResult.error("No command provided.")

        session_id = str(input_data.get("session_id") or "default")
        cwd        = input_data.get("cwd") or None

        raw_timeout = input_data.get("timeout", _DEFAULT_TIMEOUT)
        try:
            timeout = float(max(1, min(float(raw_timeout), _MAX_TIMEOUT)))
        except (TypeError, ValueError):
            timeout = _DEFAULT_TIMEOUT

        # Special command: reset the session.
        if command == "__reset__":
            _reset_shell(session_id, cwd)
            return ToolResult.ok(
                f"Session '{session_id}' has been reset.",
                session_id=session_id,
                reset=True,
            )

        shell = _get_or_create_shell(session_id, cwd)

        try:
            output = shell.run(command, timeout=timeout)
        except TimeoutError:
            _reset_shell(session_id, cwd)
            return ToolResult.error(
                f"Command timed out after {timeout}s: {command}\n"
                "The session has been reset.",
                session_id=session_id,
            )
        except OSError as exc:
            _reset_shell(session_id, cwd)
            return ToolResult.error(
                f"Shell error: {exc}\nThe session has been reset.",
                session_id=session_id,
            )

        # Tail-truncate large output (most useful info is at the end).
        total = len(output)
        if total > MAX_OUTPUT_CHARS:
            output = (
                f"[Output truncated — showing last {MAX_OUTPUT_CHARS} of {total} chars]\n\n"
                + output[-MAX_OUTPUT_CHARS:]
            )

        return ToolResult.ok(
            output or "(no output)",
            session_id=session_id,
        )
