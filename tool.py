"""
tool.py
=======
Base abstractions for the tool system.

Every executable capability (BashTool, FileReadTool, …) inherits from
:class:`Tool` and is registered in the global :data:`TOOL_REGISTRY`.  The
query engine uses this registry to build the ``tools`` array sent to the
Claude API and to dispatch ``tool_use`` blocks back to the right handler.

Design notes
------------
* Tools are *stateless* — all context comes in through ``input_data``.
* A :class:`ToolResult` can carry either text content or an error flag;
  the query engine wraps the appropriate Anthropic SDK block type.
* The :class:`PermissionDecision` enum mirrors the TypeScript source's
  allow / deny / ask semantics.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from claude_code.permissions.permission_manager import PermissionManager


# ---------------------------------------------------------------------------
# Permission decision
# ---------------------------------------------------------------------------

class PermissionDecision(Enum):
    """Result of a permission check before tool execution."""
    ALLOW = "allow"      # Proceed without prompting
    DENY  = "deny"       # Block execution
    ASK   = "ask"        # Prompt the user interactively


# ---------------------------------------------------------------------------
# Tool result
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    """
    The value a tool returns after execution.

    ``content`` is always a string (the tool's output text).
    ``is_error`` signals that something went wrong; the query engine
    surfaces this as an error tool-result block to Claude.
    """
    content: str
    is_error: bool = False
    # Optional metadata for UI display (not sent to the API)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(cls, content: str, **metadata: Any) -> "ToolResult":
        """Convenience constructor for a successful result."""
        return cls(content=content, is_error=False, metadata=metadata)

    @classmethod
    def error(cls, message: str, **metadata: Any) -> "ToolResult":
        """Convenience constructor for an error result."""
        return cls(content=message, is_error=True, metadata=metadata)


# ---------------------------------------------------------------------------
# Tool base class
# ---------------------------------------------------------------------------

class Tool(abc.ABC):
    """
    Abstract base for all Claude Code tools.

    Sub-classes must set the class-level attributes and implement
    :meth:`execute`.  They *may* override :meth:`check_permission` if they
    need custom logic beyond the default ``permission_manager`` lookup.
    """

    # --- Class-level metadata (required by every sub-class) ---------------

    name: str = ""
    """Unique snake_case identifier sent to the Claude API."""

    description: str = ""
    """Plain-English description of what the tool does."""

    input_schema: Dict[str, Any] = {}
    """
    JSON Schema describing the tool's parameters.
    Must follow the subset supported by the Anthropic ``tools`` array.
    """

    # Whether the tool is considered "dangerous" by default.
    # Used by the permission manager when no explicit rule is configured.
    dangerous: bool = False

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def validate_input(self, input_data: Dict[str, Any]) -> Optional[str]:
        """
        Validate *input_data* against the tool's ``input_schema``.

        Returns an error message string if validation fails, or ``None`` when
        the input is valid.  The default implementation checks that all fields
        listed in the schema's ``required`` array are present and non-empty.
        Sub-classes may override for richer validation.
        """
        required_fields: List[str] = self.input_schema.get("required", [])
        missing = [f for f in required_fields if f not in input_data]
        if missing:
            return f"Missing required field(s): {', '.join(missing)}"
        return None

    @abc.abstractmethod
    def execute(
        self,
        input_data: Dict[str, Any],
        permission_manager: Optional["PermissionManager"] = None,
    ) -> ToolResult:
        """
        Run the tool with the given *input_data* dict.

        Parameters
        ----------
        input_data:
            The parsed JSON object from the ``tool_use`` block.
        permission_manager:
            Optional permission manager; tools should call
            :meth:`check_permission` rather than using it directly.

        Returns
        -------
        ToolResult
        """

    # ------------------------------------------------------------------
    # Permission helpers
    # ------------------------------------------------------------------

    def check_permission(
        self,
        input_data: Dict[str, Any],
        permission_manager: Optional["PermissionManager"],
    ) -> PermissionDecision:
        """
        Ask the permission manager whether this tool call is allowed.

        Falls back to ALLOW when no permission manager is provided (useful
        in tests).
        """
        if permission_manager is None:
            return PermissionDecision.ALLOW
        return permission_manager.check(tool=self, input_data=input_data)

    # ------------------------------------------------------------------
    # SDK serialisation
    # ------------------------------------------------------------------

    def to_api_dict(self) -> Dict[str, Any]:
        """
        Serialise into the dict shape the Anthropic API expects inside the
        ``tools`` array.
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def __repr__(self) -> str:
        return f"<Tool name={self.name!r}>"


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

class ToolRegistry:
    """
    A dict-like container that maps tool names to :class:`Tool` instances.

    The query engine calls :meth:`get_all` to build the API payload and
    :meth:`find` to dispatch ``tool_use`` results.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Add *tool* to the registry (raises if name already taken)."""
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered.")
        self._tools[tool.name] = tool

    def find(self, name: str) -> Optional[Tool]:
        """Return the tool with the given *name*, or ``None``."""
        return self._tools.get(name)

    def get_all(self) -> List[Tool]:
        """Return all registered tools in insertion order."""
        return list(self._tools.values())

    def to_api_list(self) -> List[Dict[str, Any]]:
        """Return the list of API dicts to pass as ``tools=`` to the SDK."""
        return [t.to_api_dict() for t in self.get_all()]

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
