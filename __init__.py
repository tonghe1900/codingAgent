"""
claude_code
===========
Python implementation of Claude Code — an AI-powered interactive
development assistant for the terminal.

Top-level package exports the most commonly used public classes so
callers can write::

    from claude_code import QueryEngine, Settings, ToolRegistry
"""

from claude_code.query_engine import QueryEngine
from claude_code.config.settings import Settings, SettingsLoader
from claude_code.tool import Tool, ToolResult, ToolRegistry, PermissionDecision
from claude_code.history import HistoryManager
from claude_code.cost_tracker import CostTracker
from claude_code.context import build_system_prompt
from claude_code.permissions.permission_manager import PermissionManager
from claude_code.session.session_manager import SessionManager, Session
from claude_code.api.claude_client import ClaudeClient
from claude_code.tools import build_default_registry

__all__ = [
    "QueryEngine",
    "Settings",
    "SettingsLoader",
    "Tool",
    "ToolResult",
    "ToolRegistry",
    "PermissionDecision",
    "HistoryManager",
    "CostTracker",
    "build_system_prompt",
    "PermissionManager",
    "SessionManager",
    "Session",
    "ClaudeClient",
    "build_default_registry",
]
