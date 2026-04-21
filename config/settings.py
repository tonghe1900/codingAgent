"""
config/settings.py
==================
Multi-source settings system.

Precedence order (highest → lowest):
  1. CLI flags (injected via ``override`` at construction time)
  2. Project config  (.claude.json in the current working directory)
  3. User settings   (~/.claude/settings.json)
  4. User config     (~/.claude/config.json)
  5. Environment variables  (ANTHROPIC_API_KEY, CLAUDE_MODEL, …)
  6. Built-in defaults

The :class:`Settings` dataclass is intentionally flat — deep nesting makes
validation and hot-reload harder than it is worth.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL      = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 8192
DEFAULT_PERMISSION_MODE = "default"   # "default" | "bypass" | "auto"
CLAUDE_DIR = Path.home() / ".claude"


# ---------------------------------------------------------------------------
# Settings dataclass
# ---------------------------------------------------------------------------

@dataclass
class Settings:
    """
    All runtime settings for one Claude Code session.

    Attributes are intentionally plain Python types so they can round-trip
    through JSON without custom serialisers.
    """

    # --- Provider & Model --------------------------------------------------
    provider: str = "anthropic"            # "anthropic" | "deepseek" | "openai" | custom
    model: str = DEFAULT_MODEL
    fallback_model: Optional[str] = None   # Used when primary model fails
    base_url: Optional[str] = None         # Override API endpoint (OpenAI-compat providers)
    max_tokens: int = DEFAULT_MAX_TOKENS
    api_key: Optional[str] = None          # Overrides ANTHROPIC_API_KEY / DEEPSEEK_API_KEY etc.

    # --- Permissions -------------------------------------------------------
    permission_mode: str = DEFAULT_PERMISSION_MODE
    # Tools whose execution is always allowed without prompting.
    allowed_tools: List[str] = field(default_factory=list)
    # Tools that are always denied.
    denied_tools: List[str] = field(default_factory=list)

    # --- System prompt customisation ---------------------------------------
    system_prompt_extra: str = ""           # Appended to the built-in system prompt
    append_system_prompt: str = ""          # Additional text appended after CLAUDE.md
    custom_system_prompt: Optional[str] = None  # Fully replaces the built-in prompt

    # --- Session behaviour -------------------------------------------------
    cwd: str = field(default_factory=os.getcwd)
    verbose: bool = False
    stream: bool = True                     # Use streaming API (recommended)

    # --- Cost / budget limits ----------------------------------------------
    max_budget_usd: Optional[float] = None  # Hard cap on total session cost (USD)
    task_budget_usd: Optional[float] = None # Soft cap per task/query

    # --- Thinking mode (Anthropic extended thinking) -----------------------
    thinking_enabled: bool = False
    thinking_budget: int = 5000             # Token budget for thinking blocks

    # --- Session persistence -----------------------------------------------
    sessions_dir: str = field(
        default_factory=lambda: str(CLAUDE_DIR / "sessions")
    )

    # --- Context window management -----------------------------------------
    # When the conversation history exceeds this fraction of max_tokens,
    # the history manager will compact older messages.
    compact_threshold: float = 0.85


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class SettingsLoader:
    """
    Builds a :class:`Settings` instance by merging all configuration sources.

    Typical usage::

        loader = SettingsLoader(cwd=Path.cwd())
        settings = loader.load(overrides={"model": "claude-opus-4-6"})
    """

    def __init__(self, cwd: Optional[Path] = None) -> None:
        self._cwd = cwd or Path.cwd()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, overrides: Optional[Dict[str, Any]] = None) -> Settings:
        """
        Merge all sources and return a validated :class:`Settings`.

        ``overrides`` (from CLI flags) take the highest precedence.
        """
        merged: Dict[str, Any] = {}

        # Start with built-in defaults.
        defaults = asdict(Settings())
        merged.update(defaults)

        # Layer 1 — environment variables.
        merged.update(self._from_env())

        # Layer 2 — user config (~/.claude/config.json).
        merged.update(self._read_json(CLAUDE_DIR / "config.json"))

        # Layer 3 — user settings (~/.claude/settings.json).
        merged.update(self._read_json(CLAUDE_DIR / "settings.json"))

        # Layer 4 — project config (.claude.json in CWD).
        merged.update(self._read_json(self._cwd / ".claude.json"))

        # Layer 5 — explicit CLI overrides.
        if overrides:
            # Filter out None values so they don't shadow lower layers.
            merged.update({k: v for k, v in overrides.items() if v is not None})

        return self._validate(merged)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _from_env() -> Dict[str, Any]:
        """Extract settings from well-known environment variables."""
        result: Dict[str, Any] = {}

        # Provider detection: check which API key env var is set.
        if key := os.getenv("ANTHROPIC_API_KEY"):
            result["api_key"] = key
            result.setdefault("provider", "anthropic")
        elif key := os.getenv("DEEPSEEK_API_KEY"):
            result["api_key"] = key
            result["provider"] = "deepseek"
        elif key := os.getenv("OPENAI_API_KEY"):
            result["api_key"] = key
            result["provider"] = "openai"

        if model := os.getenv("CLAUDE_MODEL"):
            result["model"] = model
        if provider := os.getenv("CLAUDE_PROVIDER"):
            result["provider"] = provider
        if base_url := os.getenv("CLAUDE_BASE_URL"):
            result["base_url"] = base_url
        if mode := os.getenv("CLAUDE_PERMISSION_MODE"):
            result["permission_mode"] = mode
        if verbose := os.getenv("CLAUDE_VERBOSE"):
            result["verbose"] = verbose.lower() in ("1", "true", "yes")
        if thinking := os.getenv("CLAUDE_THINKING"):
            result["thinking_enabled"] = thinking.lower() in ("1", "true", "yes")

        return result

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        """
        Read *path* as JSON and return its top-level dict.

        Returns ``{}`` silently if the file does not exist or is malformed;
        this keeps startup robust against partial configs.
        """
        if not path.exists():
            return {}
        try:
            with path.open() as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                logger.warning("Settings file %s is not a JSON object — ignored.", path)
                return {}
            return data
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read %s: %s", path, exc)
            return {}

    @staticmethod
    def _validate(data: Dict[str, Any]) -> Settings:
        """
        Construct and validate a :class:`Settings` from the merged dict.

        Unknown keys are silently ignored (forward-compatible).
        Known keys are type-coerced where cheap; raises ``ValueError`` on
        obviously wrong values.
        """
        # Retain only keys that Settings knows about.
        valid_keys = {f.name for f in Settings.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in valid_keys}

        settings = Settings(**filtered)  # type: ignore[arg-type]

        # Validate permission_mode.
        valid_modes = {"default", "bypass", "auto"}
        if settings.permission_mode not in valid_modes:
            raise ValueError(
                f"Invalid permission_mode '{settings.permission_mode}'. "
                f"Must be one of {valid_modes}."
            )

        # Validate max_tokens range.
        if not (1 <= settings.max_tokens <= 100_000):
            raise ValueError(
                f"max_tokens must be between 1 and 100,000, got {settings.max_tokens}."
            )

        # Validate max_budget_usd.
        if settings.max_budget_usd is not None and settings.max_budget_usd <= 0:
            raise ValueError("max_budget_usd must be a positive number.")

        # Auto-infer provider from model name when not explicitly set.
        if settings.provider == "anthropic" and settings.model.startswith("deepseek"):
            settings.provider = "deepseek"
        elif settings.provider == "anthropic" and settings.model.startswith(("gpt-", "o1", "o3")):
            settings.provider = "openai"

        return settings

    # ------------------------------------------------------------------
    # Convenience: write back user settings
    # ------------------------------------------------------------------

    @staticmethod
    def save_user_settings(updates: Dict[str, Any]) -> None:
        """
        Merge *updates* into ``~/.claude/settings.json``.

        Creates the file and parent directory if they do not yet exist.
        """
        CLAUDE_DIR.mkdir(parents=True, exist_ok=True)
        path = CLAUDE_DIR / "settings.json"

        existing: Dict[str, Any] = {}
        if path.exists():
            try:
                with path.open() as fh:
                    existing = json.load(fh)
            except (json.JSONDecodeError, OSError):
                pass  # Overwrite corrupt file.

        existing.update(updates)
        with path.open("w") as fh:
            json.dump(existing, fh, indent=2)
