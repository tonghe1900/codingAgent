"""
tools/__init__.py
=================
Assembles the default :class:`~claude_code.tool.ToolRegistry` with every
built-in tool pre-registered.

    from claude_code.tools import build_default_registry
    registry = build_default_registry()
"""

from claude_code.tool import ToolRegistry
from claude_code.tools.bash_tool import BashTool
from claude_code.tools.file_read_tool import FileReadTool
from claude_code.tools.file_write_tool import FileWriteTool
from claude_code.tools.file_edit_tool import FileEditTool
from claude_code.tools.glob_tool import GlobTool
from claude_code.tools.grep_tool import GrepTool
from claude_code.tools.web_fetch_tool import WebFetchTool
from claude_code.tools.web_search_tool import WebSearchTool
from claude_code.tools.todo_tool import TodoWriteTool
from claude_code.tools.config_tool import ConfigTool
from claude_code.tools.notebook_edit_tool import NotebookEditTool
from claude_code.tools.sleep_tool import SleepTool
from claude_code.tools.ask_user_tool import AskUserQuestionTool
from claude_code.tools.repl_tool import REPLTool


def build_default_registry(settings=None) -> ToolRegistry:
    """
    Return a :class:`ToolRegistry` containing all built-in tools.

    Parameters
    ----------
    settings:
        Optional :class:`~claude_code.config.settings.Settings` instance.
        Passed to tools that need live configuration access (e.g. ConfigTool).
    """
    registry = ToolRegistry()

    # --- Filesystem (read-only first, mutating last) ---
    registry.register(FileReadTool())
    registry.register(FileWriteTool())
    registry.register(FileEditTool())
    registry.register(GlobTool())
    registry.register(GrepTool())

    # --- Execution ---
    registry.register(BashTool())
    registry.register(REPLTool())
    registry.register(SleepTool())

    # --- Web ---
    registry.register(WebFetchTool())
    registry.register(WebSearchTool())

    # --- Task management ---
    registry.register(TodoWriteTool())

    # --- Configuration ---
    registry.register(ConfigTool(settings=settings))

    # --- Notebook ---
    registry.register(NotebookEditTool())

    # --- User interaction ---
    registry.register(AskUserQuestionTool())

    return registry


__all__ = [
    "build_default_registry",
    "BashTool",
    "FileReadTool",
    "FileWriteTool",
    "FileEditTool",
    "GlobTool",
    "GrepTool",
    "WebFetchTool",
    "WebSearchTool",
    "TodoWriteTool",
    "ConfigTool",
    "NotebookEditTool",
    "SleepTool",
    "AskUserQuestionTool",
    "REPLTool",
]
