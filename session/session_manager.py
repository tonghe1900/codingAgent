"""
session/session_manager.py
==========================
Persists and loads conversation sessions to/from disk.

Each session is stored as a JSON file in ``~/.claude/sessions/``:

    ~/.claude/sessions/<session_id>.json

File schema
-----------
{
    "id":         "abc123",
    "created_at": 1700000000.0,
    "updated_at": 1700001234.5,
    "cwd":        "/home/user/project",
    "model":      "claude-sonnet-4-6",
    "messages":   [ ... Anthropic SDK message dicts ... ],
    "total_cost": 0.0042
}
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SESSIONS_DIR = Path.home() / ".claude" / "sessions"


# ---------------------------------------------------------------------------
# Session dataclass
# ---------------------------------------------------------------------------

@dataclass
class Session:
    """Serialisable record for one conversation session."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    cwd: str = field(default_factory=os.getcwd)
    model: str = "claude-sonnet-4-6"
    messages: List[Dict[str, Any]] = field(default_factory=list)
    total_cost: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict (JSON-safe)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        """Deserialise from a plain dict."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class SessionManager:
    """
    Saves and loads :class:`Session` objects.

    Parameters
    ----------
    sessions_dir:
        Directory to store session files.  Defaults to ``~/.claude/sessions``.
    """

    def __init__(self, sessions_dir: Optional[Path] = None) -> None:
        self._dir = Path(sessions_dir or SESSIONS_DIR)

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, session: Session) -> Path:
        """
        Write *session* to disk (creates the directory if needed).

        Returns the path of the written file.
        """
        self._dir.mkdir(parents=True, exist_ok=True)
        session.updated_at = time.time()

        path = self._dir / f"{session.id}.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(session.to_dict(), fh, indent=2)

        logger.debug("Session %s saved to %s", session.id, path)
        return path

    def load(self, session_id: str) -> Optional[Session]:
        """
        Load and return the session with *session_id*, or ``None`` if not found.
        """
        path = self._dir / f"{session_id}.json"
        if not path.exists():
            return None

        try:
            with path.open(encoding="utf-8") as fh:
                data = json.load(fh)
            return Session.from_dict(data)
        except (json.JSONDecodeError, OSError, TypeError) as exc:
            logger.warning("Could not load session %s: %s", session_id, exc)
            return None

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def list_sessions(self, limit: int = 20) -> List[Session]:
        """
        Return the *limit* most-recently-updated sessions.

        Missing or corrupt files are silently skipped.
        """
        if not self._dir.exists():
            return []

        files = sorted(
            self._dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        sessions: List[Session] = []
        for path in files[:limit]:
            try:
                with path.open(encoding="utf-8") as fh:
                    data = json.load(fh)
                sessions.append(Session.from_dict(data))
            except Exception:
                continue

        return sessions

    def delete(self, session_id: str) -> bool:
        """Delete the session file for *session_id*.  Returns True on success."""
        path = self._dir / f"{session_id}.json"
        if path.exists():
            path.unlink()
            return True
        return False

    def append_message(self, session: Session, message: Dict[str, Any]) -> None:
        """
        Append *message* to *session* and immediately persist the update.

        This is the incremental save path — called after every turn so
        the session is recoverable even if the process is killed mid-session.
        """
        session.messages.append(message)
        self.save(session)
