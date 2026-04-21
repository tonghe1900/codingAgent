"""
tools/config_tool.py
====================
ConfigTool — lets Claude read (and optionally update) the current session
settings.

This mirrors the TypeScript ``ConfigTool`` which exposes the settings object
to Claude so it can report the active model, permission mode, and other
parameters without the user having to type ``/config``.

Operations
----------
* ``get``  — return all settings (or one key when ``key`` is specified).
* ``set``  — update a key in ``~/.claude/settings.json``.
           Only a safe subset of keys can be set at runtime.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any, Dict, List, Optional

from claude_code.tool import Tool, ToolResult

# Keys that Claude is allowed to set at runtime (safety subset).
_SETTABLE_KEYS: set[str] = {
    "model",
    "max_tokens",
    "verbose",
    "stream",
    "thinking_enabled",
    "thinking_budget",
    "append_system_prompt",
    "system_prompt_extra",
    "permission_mode",
}


class ConfigTool(Tool):
    """Read or update the active session configuration."""

    name = "Config"
    description = (
        "Read or update Claude Code session settings. "
        "Use 'get' to read all settings or a specific key. "
        "Use 'set' to update a setting (limited to safe runtime-settable keys)."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["get", "set"],
                "description": "Whether to read or write a setting.",
            },
            "key": {
                "type": "string",
                "description": "Setting key (optional for 'get', required for 'set').",
            },
            "value": {
                "description": "New value for the key (required for 'set').",
            },
        },
        "required": ["operation"],
    }
    dangerous = False

    # The current settings object is injected at construction time so Claude
    # can read live values without needing access to the filesystem.
    def __init__(self, settings=None) -> None:  # type: ignore[override]
        self._settings = settings

    def execute(
        self,
        input_data: Dict[str, Any],
        permission_manager=None,
    ) -> ToolResult:
        op = input_data.get("operation", "get")

        if op == "get":
            return self._get(input_data)
        if op == "set":
            return self._set(input_data)

        return ToolResult.error(f"Unknown operation '{op}'. Must be 'get' or 'set'.")

    # ------------------------------------------------------------------
    # get
    # ------------------------------------------------------------------

    def _get(self, data: Dict[str, Any]) -> ToolResult:
        if self._settings is None:
            return ToolResult.error("No settings object is available.")

        key = data.get("key")
        settings_dict = dataclasses.asdict(self._settings)

        # Mask the API key.
        if "api_key" in settings_dict and settings_dict["api_key"]:
            settings_dict["api_key"] = settings_dict["api_key"][:8] + "…"

        if key:
            if key not in settings_dict:
                return ToolResult.error(
                    f"Unknown setting key '{key}'. "
                    f"Available keys: {sorted(settings_dict)}"
                )
            value = settings_dict[key]
            return ToolResult.ok(f"{key} = {json.dumps(value)}")

        # Return all settings as JSON.
        return ToolResult.ok(json.dumps(settings_dict, indent=2))

    # ------------------------------------------------------------------
    # set
    # ------------------------------------------------------------------

    def _set(self, data: Dict[str, Any]) -> ToolResult:
        key   = data.get("key", "").strip()
        value = data.get("value")

        if not key:
            return ToolResult.error("'key' is required for the set operation.")
        if value is None:
            return ToolResult.error("'value' is required for the set operation.")
        if key not in _SETTABLE_KEYS:
            return ToolResult.error(
                f"Key '{key}' cannot be changed at runtime. "
                f"Settable keys: {sorted(_SETTABLE_KEYS)}"
            )

        # Update in-memory settings.
        if self._settings is not None and hasattr(self._settings, key):
            try:
                setattr(self._settings, key, value)
            except (TypeError, ValueError) as exc:
                return ToolResult.error(f"Invalid value for '{key}': {exc}")

        # Persist to user settings file.
        try:
            from claude_code.config.settings import SettingsLoader
            SettingsLoader.save_user_settings({key: value})
        except Exception as exc:
            return ToolResult.error(f"Could not persist setting: {exc}")

        return ToolResult.ok(f"Set {key} = {json.dumps(value)}")
