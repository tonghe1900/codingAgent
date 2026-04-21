"""
tools/todo_tool.py
==================
TodoWriteTool — manages a persistent task list stored as JSON.

The task list lives at ``~/.claude/todos.json`` (project-agnostic) and is
shared across sessions.  Claude uses this tool to track multi-step plans,
mark items done, and communicate progress to the user.

Task schema (per item):
    {
        "id":      "t1",                  # unique string
        "content": "Refactor auth module", # human-readable description
        "status":  "pending",              # pending | in_progress | completed
        "priority": "high"                 # high | medium | low
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from claude_code.tool import Tool, ToolResult

TODOS_FILE = Path.home() / ".claude" / "todos.json"

# Valid status / priority values.
VALID_STATUSES   = {"pending", "in_progress", "completed"}
VALID_PRIORITIES = {"high", "medium", "low"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_todos() -> List[Dict[str, Any]]:
    """Load the todo list from disk, returning an empty list on any failure."""
    if not TODOS_FILE.exists():
        return []
    try:
        data = json.loads(TODOS_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _save_todos(todos: List[Dict[str, Any]]) -> None:
    """Write the todo list to disk (creates the directory if needed)."""
    TODOS_FILE.parent.mkdir(parents=True, exist_ok=True)
    TODOS_FILE.write_text(json.dumps(todos, indent=2), encoding="utf-8")


def _format_todos(todos: List[Dict[str, Any]]) -> str:
    """Return a human-readable representation of the todo list."""
    if not todos:
        return "Todo list is empty."
    lines = []
    for t in todos:
        status_icon = {"pending": "○", "in_progress": "◑", "completed": "●"}.get(
            t.get("status", "pending"), "○"
        )
        priority    = t.get("priority", "medium")
        lines.append(
            f"[{status_icon}] ({priority}) {t.get('id', '?')}: {t.get('content', '')}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

class TodoWriteTool(Tool):
    """
    Read or update the persistent task / todo list.

    Supports four operations via the ``operation`` parameter:
    * ``list``   — return the current todo list.
    * ``create`` — add a new item.
    * ``update`` — change the status or content of an item.
    * ``delete`` — remove an item by ID.
    """

    name = "TodoWrite"
    description = (
        "Manage a persistent task list to track multi-step plans. "
        "Use 'list' to view tasks, 'create' to add, 'update' to change status, "
        "'delete' to remove."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["list", "create", "update", "delete"],
                "description": "Action to perform on the todo list.",
            },
            "id": {
                "type": "string",
                "description": "Task ID (required for update and delete).",
            },
            "content": {
                "type": "string",
                "description": "Task description (required for create; optional for update).",
            },
            "status": {
                "type": "string",
                "enum": ["pending", "in_progress", "completed"],
                "description": "Task status (used in create and update).",
            },
            "priority": {
                "type": "string",
                "enum": ["high", "medium", "low"],
                "description": "Task priority (used in create; default 'medium').",
            },
        },
        "required": ["operation"],
    }
    dangerous = False

    def execute(
        self,
        input_data: Dict[str, Any],
        permission_manager=None,
    ) -> ToolResult:
        op = input_data.get("operation", "")
        dispatch = {
            "list":   self._list,
            "create": self._create,
            "update": self._update,
            "delete": self._delete,
        }
        handler = dispatch.get(op)
        if handler is None:
            return ToolResult.error(
                f"Unknown operation '{op}'. Must be one of: {list(dispatch)}."
            )
        return handler(input_data)

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def _list(self, _: Dict[str, Any]) -> ToolResult:
        todos = _load_todos()
        return ToolResult.ok(_format_todos(todos), count=len(todos))

    def _create(self, data: Dict[str, Any]) -> ToolResult:
        content = data.get("content", "").strip()
        if not content:
            return ToolResult.error("'content' is required for the create operation.")

        todos = _load_todos()
        # Auto-generate a unique ID.
        existing_ids = {t.get("id", "") for t in todos}
        task_id = data.get("id") or self._gen_id(existing_ids)

        if task_id in existing_ids:
            return ToolResult.error(f"A task with id '{task_id}' already exists.")

        status   = data.get("status",   "pending")
        priority = data.get("priority", "medium")
        if status not in VALID_STATUSES:
            return ToolResult.error(f"Invalid status '{status}'.")
        if priority not in VALID_PRIORITIES:
            return ToolResult.error(f"Invalid priority '{priority}'.")

        task: Dict[str, Any] = {
            "id": task_id, "content": content,
            "status": status, "priority": priority,
        }
        todos.append(task)
        _save_todos(todos)
        return ToolResult.ok(f"Created task [{task_id}]: {content}", task=task)

    def _update(self, data: Dict[str, Any]) -> ToolResult:
        task_id = data.get("id", "").strip()
        if not task_id:
            return ToolResult.error("'id' is required for the update operation.")

        todos = _load_todos()
        for task in todos:
            if task.get("id") == task_id:
                if "content"  in data: task["content"]  = data["content"]
                if "status"   in data:
                    if data["status"] not in VALID_STATUSES:
                        return ToolResult.error(f"Invalid status '{data['status']}'.")
                    task["status"]   = data["status"]
                if "priority" in data:
                    if data["priority"] not in VALID_PRIORITIES:
                        return ToolResult.error(f"Invalid priority '{data['priority']}'.")
                    task["priority"] = data["priority"]
                _save_todos(todos)
                return ToolResult.ok(f"Updated task [{task_id}].", task=task)

        return ToolResult.error(f"Task '{task_id}' not found.")

    def _delete(self, data: Dict[str, Any]) -> ToolResult:
        task_id = data.get("id", "").strip()
        if not task_id:
            return ToolResult.error("'id' is required for the delete operation.")

        todos = _load_todos()
        new_todos = [t for t in todos if t.get("id") != task_id]
        if len(new_todos) == len(todos):
            return ToolResult.error(f"Task '{task_id}' not found.")

        _save_todos(new_todos)
        return ToolResult.ok(f"Deleted task [{task_id}].")

    # ------------------------------------------------------------------
    # ID generation
    # ------------------------------------------------------------------

    @staticmethod
    def _gen_id(existing: set) -> str:
        """Return the next available ``t<n>`` ID."""
        n = 1
        while f"t{n}" in existing:
            n += 1
        return f"t{n}"
