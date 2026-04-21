"""
tools/web_search_tool.py
========================
Performs a web search and returns result snippets.

Production integrations
-----------------------
The real Claude Code routes through Anthropic's search service.  This
implementation supports two configurable backends:

1. ``brave`` — Brave Search API (recommended; requires BRAVE_API_KEY env var).
2. ``duck``  — DuckDuckGo Instant Answer API (no key needed; limited results).

Set the backend via the ``CLAUDE_SEARCH_BACKEND`` environment variable or
via settings.  Falls back to duck when no Brave key is present.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

from claude_code.tool import Tool, ToolResult

MAX_RESULTS = 10


class WebSearchTool(Tool):
    """Search the web and return result titles, URLs, and snippets."""

    name = "WebSearch"
    description = (
        "Search the web for information. "
        "Returns titles, URLs, and short snippets for the top results. "
        "Use WebFetch to retrieve the full content of a specific page."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query.",
            },
            "num_results": {
                "type": "integer",
                "description": f"Number of results to return (max {MAX_RESULTS}, default 5).",
                "default": 5,
            },
        },
        "required": ["query"],
    }
    dangerous = False

    def execute(
        self,
        input_data: Dict[str, Any],
        permission_manager=None,
    ) -> ToolResult:
        query = input_data.get("query", "").strip()
        if not query:
            return ToolResult.error("No query provided.")

        num = min(int(input_data.get("num_results", 5) or 5), MAX_RESULTS)

        # Pick backend.
        brave_key = os.getenv("BRAVE_API_KEY")
        backend = os.getenv("CLAUDE_SEARCH_BACKEND", "brave" if brave_key else "duck")

        if backend == "brave" and brave_key:
            return self._brave_search(query, num, brave_key)
        return self._duck_search(query, num)

    # ------------------------------------------------------------------
    # Brave Search API
    # ------------------------------------------------------------------

    @staticmethod
    def _brave_search(query: str, num: int, api_key: str) -> ToolResult:
        """Use the Brave Search API (https://api.search.brave.com)."""
        try:
            import httpx  # type: ignore[import]
        except ImportError:
            return ToolResult.error("httpx is required for web search. Run: pip install httpx")

        try:
            with httpx.Client(timeout=15) as client:
                resp = client.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    headers={"Accept": "application/json", "X-Subscription-Token": api_key},
                    params={"q": query, "count": num},
                )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            return ToolResult.error(f"Brave search error: {exc}")

        results: List[str] = []
        for item in data.get("web", {}).get("results", [])[:num]:
            title   = item.get("title", "")
            url     = item.get("url", "")
            snippet = item.get("description", "")
            results.append(f"**{title}**\n{url}\n{snippet}")

        if not results:
            return ToolResult.ok("No results found.")
        return ToolResult.ok("\n\n".join(results), query=query, num_results=len(results))

    # ------------------------------------------------------------------
    # DuckDuckGo Instant Answer fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _duck_search(query: str, num: int) -> ToolResult:
        """Use the DuckDuckGo Instant Answer API (zero-click JSON endpoint)."""
        try:
            import httpx  # type: ignore[import]
        except ImportError:
            return ToolResult.error("httpx is required. Run: pip install httpx")

        try:
            with httpx.Client(timeout=15, follow_redirects=True) as client:
                resp = client.get(
                    "https://api.duckduckgo.com/",
                    params={"q": query, "format": "json", "no_redirect": "1"},
                    headers={"User-Agent": "ClaudeCode/1.0"},
                )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            return ToolResult.error(f"DuckDuckGo search error: {exc}")

        results: List[str] = []

        # Instant answer.
        if data.get("AbstractText"):
            results.append(
                f"**{data.get('Heading', query)}**\n"
                f"{data.get('AbstractURL', '')}\n"
                f"{data['AbstractText']}"
            )

        # Related topics.
        for topic in data.get("RelatedTopics", [])[:num - len(results)]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append(
                    f"{topic.get('FirstURL', '')}\n{topic['Text']}"
                )

        if not results:
            return ToolResult.ok(
                f"No instant results found for '{query}'. "
                "Consider using WebFetch on a search URL for richer results."
            )

        return ToolResult.ok("\n\n".join(results), query=query, num_results=len(results))
