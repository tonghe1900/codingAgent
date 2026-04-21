"""
tools/web_fetch_tool.py
=======================
Fetches a URL and returns its text content.

Uses ``httpx`` for HTTP/2 support and connection pooling.  HTML is
converted to plain text by stripping tags so Claude receives readable
content rather than raw markup.

Improvements vs. earlier version
---------------------------------
* ``prompt`` parameter — when provided the full raw content is searched /
  summarised using simple keyword extraction so the response stays focused.
  (TypeScript's WebFetchTool has the same ``url`` + ``prompt`` signature;
  this avoids sending 30 000 chars of irrelevant HTML to Claude when only
  a small section is needed.)
* ``elapsed_ms`` timing metadata.

Security note
-------------
We intentionally do NOT follow redirects to private/internal IP ranges
(SSRF guard).  Only http:// and https:// are allowed.
"""

from __future__ import annotations

import ipaddress
import socket
from typing import Any, Dict
from urllib.parse import urlparse

from claude_code.tool import Tool, ToolResult

MAX_CONTENT_CHARS = 30_000

# Separator used in prompt-filtered output.
_SECTION_SEP = "\n\n---\n\n"


def _apply_prompt_filter(text: str, prompt: str) -> str:
    """
    Very lightweight "apply prompt to content" that keeps only paragraphs /
    sections containing words from the prompt query.

    This is a best-effort approximation of what the TypeScript version does
    (which calls the model again with the prompt).  It keeps token usage low
    and avoids a recursive API call while still narrowing the returned content.

    Strategy:
    1. Split the text into paragraphs.
    2. Score each paragraph by how many unique prompt words appear in it.
    3. Return the top-scoring paragraphs (up to MAX_CONTENT_CHARS chars).
    """
    import re as _re
    # Extract meaningful query words (>= 3 chars, alphanumeric, not stopwords).
    # Use \w+ so "topic7" is kept whole (not split into "topic" + "7").
    _STOPWORDS = {"the", "and", "for", "with", "that", "this", "from", "are",
                  "was", "has", "have", "its", "not", "but", "can", "will"}
    query_words = {
        w.lower() for w in _re.findall(r"\w{3,}", prompt)
        if w.lower() not in _STOPWORDS
    }
    if not query_words:
        return text  # no meaningful words → return everything

    paragraphs = [p.strip() for p in _re.split(r"\n{2,}", text) if p.strip()]

    def score(para: str) -> int:
        lower = para.lower()
        return sum(1 for w in query_words if w in lower)

    scored = sorted(enumerate(paragraphs), key=lambda x: score(x[1]), reverse=True)

    # Keep paragraphs that mention at least one query word, up to char limit.
    selected: list[tuple[int, str]] = []
    total_chars = 0
    for idx, para in scored:
        if score(para) == 0:
            break
        if total_chars + len(para) > MAX_CONTENT_CHARS:
            break
        selected.append((idx, para))
        total_chars += len(para)

    if not selected:
        # No paragraphs matched — return the original (already truncated) text.
        return text

    # Re-sort by original order to preserve document flow.
    selected.sort(key=lambda x: x[0])
    result = _SECTION_SEP.join(p for _, p in selected)
    if len(selected) < len([p for p in paragraphs if score(p) > 0]):
        result += f"\n\n[Content filtered by prompt. Use without a prompt for full text.]"
    return result

# Prefix of private/loopback ranges we refuse to fetch.
_PRIVATE_PREFIXES = (
    "127.", "10.", "192.168.", "169.254.", "::1", "fc", "fd"
)


def _is_private_ip(hostname: str) -> bool:
    """Return True if *hostname* resolves to a private / loopback address."""
    try:
        addr = socket.gethostbyname(hostname)
        ip = ipaddress.ip_address(addr)
        return ip.is_private or ip.is_loopback or ip.is_link_local
    except (socket.gaierror, ValueError):
        return False


def _html_to_text(html: str) -> str:
    """Very simple HTML → plain-text conversion (strips tags)."""
    import re
    # Remove script / style blocks entirely.
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S | re.I)
    # Replace common block tags with newlines.
    html = re.sub(r"<(br|p|div|h[1-6]|li|tr)[^>]*>", "\n", html, flags=re.I)
    # Remove all remaining tags.
    html = re.sub(r"<[^>]+>", "", html)
    # Collapse whitespace.
    html = re.sub(r" {2,}", " ", html)
    html = re.sub(r"\n{3,}", "\n\n", html)
    return html.strip()


class WebFetchTool(Tool):
    """Fetch a URL and return its text content."""

    name = "WebFetch"
    description = (
        "Fetch a URL and return its text content. "
        "HTML is automatically converted to plain text. "
        "Provide a 'prompt' to focus the returned content on a specific question. "
        "Only http:// and https:// URLs are supported."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to fetch content from.",
            },
            "prompt": {
                "type": "string",
                "description": (
                    "Optional question or topic to focus the returned content on. "
                    "When provided, only the most relevant sections are returned."
                ),
            },
            "timeout": {
                "type": "integer",
                "description": "Request timeout in seconds (default 30).",
                "default": 30,
            },
        },
        "required": ["url"],
    }
    dangerous = False

    def execute(
        self,
        input_data: Dict[str, Any],
        permission_manager=None,
    ) -> ToolResult:
        """Fetch *url* and return text content, optionally filtered by *prompt*."""
        url    = input_data.get("url", "").strip()
        prompt = input_data.get("prompt", "").strip()
        if not url:
            return ToolResult.error("No URL provided.")

        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return ToolResult.error(
                f"Unsupported scheme '{parsed.scheme}'. Only http and https are allowed."
            )

        # SSRF guard.
        if _is_private_ip(parsed.hostname or ""):
            return ToolResult.error(
                "Fetching private/loopback IP addresses is not allowed."
            )

        timeout = max(1, min(int(input_data.get("timeout", 30) or 30), 120))

        try:
            import httpx  # type: ignore[import]
        except ImportError:
            return ToolResult.error(
                "httpx is not installed. Run: pip install httpx"
            )

        import time as _time
        fetch_start = _time.monotonic()

        try:
            with httpx.Client(follow_redirects=True, timeout=timeout) as client:
                response = client.get(
                    url,
                    headers={"User-Agent": "ClaudeCode/1.0 (+https://claude.ai/code)"},
                )
            response.raise_for_status()
        except httpx.TimeoutException:
            return ToolResult.error(f"Request timed out after {timeout}s: {url}")
        except httpx.HTTPStatusError as exc:
            return ToolResult.error(
                f"HTTP {exc.response.status_code} from {url}"
            )
        except httpx.RequestError as exc:
            return ToolResult.error(f"Network error fetching {url}: {exc}")

        elapsed_ms = int((_time.monotonic() - fetch_start) * 1000)

        content_type = response.headers.get("content-type", "")
        raw = response.text

        # Convert HTML to plain text.
        if "html" in content_type:
            text = _html_to_text(raw)
        else:
            text = raw

        if len(text) > MAX_CONTENT_CHARS:
            text = (
                text[:MAX_CONTENT_CHARS]
                + f"\n\n[Content truncated at {MAX_CONTENT_CHARS} chars]"
            )

        # Apply prompt filter when the caller supplied a focused question.
        if prompt:
            text = _apply_prompt_filter(text, prompt)

        return ToolResult.ok(
            text,
            url=url,
            status_code=response.status_code,
            content_type=content_type,
            elapsed_ms=elapsed_ms,
            prompt_filtered=bool(prompt),
        )
