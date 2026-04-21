# Claude Code — Architecture & Implementation Reference

> A Python reimplementation of Claude Code: an AI-powered interactive development
> assistant for the terminal.  This document explains the **high-level design**,
> every **component**, and every **source file** — how each piece works and how
> it fits into the larger system.

---

## Table of Contents

1. [High-Level Architecture](#1-high-level-architecture)
2. [Component Map](#2-component-map)
3. [Component Deep-Dives](#3-component-deep-dives)
   - [3.1 Core Layer](#31-core-layer)
   - [3.2 API Layer (`api/`)](#32-api-layer-api)
   - [3.3 Config Layer (`config/`)](#33-config-layer-config)
   - [3.4 Permissions Layer (`permissions/`)](#34-permissions-layer-permissions)
   - [3.5 Session Layer (`session/`)](#35-session-layer-session)
   - [3.6 Tools Layer (`tools/`)](#36-tools-layer-tools)
4. [Data-Flow Walkthrough](#4-data-flow-walkthrough)
5. [File-by-File Reference](#5-file-by-file-reference)

---

## 1. High-Level Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                         Caller / REPL / CLI                         │
└──────────────────────────────┬─────────────────────────────────────┘
                               │ user_message (str)
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                         QueryEngine                                   │
│  • Owns the agentic loop (up to 150 turns)                           │
│  • Budget enforcement   • Fallback model   • Tool dispatch           │
└────┬──────────────┬────────────────┬────────────────┬───────────────┘
     │              │                │                │
     ▼              ▼                ▼                ▼
 Settings      HistoryManager    BaseClient      PermissionManager
 (config/)     (history.py)      (api/)           (permissions/)
     │              │                │                │
     │              │         ┌──────┴──────┐         │
     │              │    ClaudeClient  OpenAICompatClient
     │              │    (Anthropic)   (DeepSeek/OpenAI)
     │              │                │
     ▼              │                ▼
 CostTracker        │           TurnResult
 (cost_tracker.py)  │    (message + tool_uses + usage)
                    │
               ┌────┴─────────────────────────┐
               │        ToolRegistry           │
               │  (tool.py + tools/__init__.py) │
               └──────────────────────────────┘
                    │ 14 concrete tools
                    ▼
        Bash / Read / Write / Edit / Glob / Grep /
        WebFetch / WebSearch / TodoWrite / Config /
        NotebookEdit / Sleep / AskUser / REPL
```

### Design Philosophy

| Principle | How it is applied |
|-----------|-------------------|
| **Provider-agnostic** | `QueryEngine` only depends on `BaseClient`; swapping Anthropic for DeepSeek/OpenAI requires no core changes |
| **History in Anthropic format** | All providers store history in Anthropic's dict format; `OpenAICompatClient` translates on-the-fly |
| **Stateless tools** | Every `Tool.execute()` is a pure function — all context comes in through `input_data` |
| **Layered settings** | 6 sources merged lowest-to-highest: defaults → env vars → user config → user settings → project config → CLI flags |
| **Defence in depth** | `PermissionManager` has 4 independent guards: deny-list, dangerous-path check, path-pattern rules, mode-based logic |

---

## 2. Component Map

The codebase has **7 components** organised into packages:

| # | Component | Package/File | Responsibility |
|---|-----------|--------------|----------------|
| 1 | **Core layer** | Root-level files | `QueryEngine`, `Tool` base, `HistoryManager`, `CostTracker`, `context` builder |
| 2 | **API layer** | `api/` | All LLM provider communication |
| 3 | **Config layer** | `config/` | Settings loading and validation |
| 4 | **Permissions layer** | `permissions/` | Access control before every tool call |
| 5 | **Session layer** | `session/` | Disk-based conversation persistence |
| 6 | **Tools layer** | `tools/` | 14 executable capabilities exposed to Claude |
| 7 | **Package init** | `__init__.py` | Public surface re-exported for callers |

---

## 3. Component Deep-Dives

### 3.1 Core Layer

The core layer contains the most important files — the agentic loop, message history,
cost tracking, system-prompt construction, and the abstract tool contract.

#### `query_engine.py` — The Heart

`QueryEngine` owns the **agentic loop**: the cycle of calling the LLM, running
any requested tools, feeding results back, and repeating until `stop_reason == "end_turn"`
or a safety cap is hit.

```
run(user_message)
  ├── budget pre-check
  ├── history.add_user_message
  ├── build_system_prompt()
  ├── loop (max 150 turns):
  │     ├── active_client.stream_turn(...)   ← call LLM
  │     ├── cost_tracker.record(...)
  │     ├── history.add_assistant_message
  │     ├── budget post-check
  │     ├── if stop_reason == end_turn → break
  │     └── _execute_tool_uses(...)          ← run each tool
  │           └── history.add_tool_result(...)
  └── return concatenated text
```

Error resilience:
- `ContextWindowExceededError` → force compact history, retry same turn
- `QuotaExceededError / ClaudeAPIError` → `_try_fallback()` with secondary client/model
- Unexpected tool exceptions are caught and returned as error `ToolResult`s (loop continues)

Key design: the loop never re-raises most errors to the caller; it degrades gracefully
so the user gets a useful message rather than a crash.

#### `tool.py` — Tool Contract

Defines three things used everywhere:

1. **`PermissionDecision`** enum — `ALLOW`, `DENY`, `ASK`
2. **`ToolResult`** dataclass — `content: str`, `is_error: bool`, `metadata: dict`
   - Factory methods: `ToolResult.ok(...)`, `ToolResult.error(...)`
3. **`Tool`** abstract base class
   - Class attributes: `name`, `description`, `input_schema`, `dangerous`
   - Abstract method: `execute(input_data, permission_manager) → ToolResult`
   - Helper: `check_permission(...)` delegates to `PermissionManager`
   - Serialisation: `to_api_dict()` → the JSON dict the Anthropic API expects
4. **`ToolRegistry`** — ordered dict of `Tool` instances
   - `register(tool)` → adds tool (raises on duplicate name)
   - `find(name)` → lookup by name
   - `to_api_list()` → list of API dicts passed as `tools=` to the SDK

#### `history.py` — Conversation Memory

Stores the message list in **Anthropic SDK dict format** so it can be passed
directly to `client.messages.create(messages=...)`.

Key operations:
- `add_user_message(content)` — plain string or content-block list
- `add_assistant_message(content)` — content-block list
- `add_tool_result(tool_use_id, result_content, is_error)` — merges into the
  last user message if it already holds tool results (Anthropic API requirement)
- `get_messages()` — returns list **after** calling `_maybe_compact()`

**Compaction algorithm**:
1. Estimate token count: `total_chars / 3.5`
2. If > `max_tokens × compact_threshold` (default 0.85): compact
3. Find the midpoint; search forward then backward for an **assistant** message
4. Replace messages before that cutoff with a single `[System: Earlier conversation compacted…]` placeholder
5. Maintains user↔assistant alternation required by the API

#### `cost_tracker.py` — Token & Cost Accounting

Per-turn records (`TurnUsage`) hold:
- `input_tokens`, `output_tokens`, `cache_read_tokens`, `cache_write_tokens`
- `cost_usd` auto-computed via `_price_for(model)` prefix-match against `_PRICE_TABLE`

Cache pricing: reads at 10% of normal input; writes at 125% of normal input.

Supported providers in the price table: Anthropic Claude 3/4, DeepSeek V3/R1,
OpenAI GPT-4o, o1, o3, o4-mini.

Display methods: `summary()` (one-liner), `turn_summary(turn)`, `per_model_summary()`.

#### `context.py` — System Prompt Builder

`build_system_prompt()` assembles the prompt from up to 7 parts, in order:

1. **Core identity** (`_CORE_SYSTEM_PROMPT`) — who Claude is, capabilities, runtime rules
2. **Date/time** — current timestamp
3. **CWD** — current working directory
4. **Runtime context** (`get_runtime_context()`) — detected Python/Node interpreters,
   Python 2 warnings; cached with `@lru_cache(maxsize=1)`
5. **Git context** (`get_git_context()`) — branch, status, recent commits; only when CWD is in a git repo
6. **CLAUDE.md** (`_load_claude_md()`) — searches up from CWD to root, then `~/.claude/`
7. **Extra/append instructions** — from `Settings.system_prompt_extra` and `Settings.append_system_prompt`

Git status is truncated at 2000 chars to prevent large repos from bloating the prompt.

---

### 3.2 API Layer (`api/`)

The API layer abstracts all LLM communication behind a single interface.

#### `base_client.py` — Provider Interface

`BaseClient` is an abstract class with one required method:

```python
stream_turn(
    *, model, system, messages, tools, max_tokens,
    on_text, thinking, thinking_budget
) -> TurnResult
```

`messages` and `tools` are always in **Anthropic format**; subclasses convert
to their native wire format before sending.

Optional property overrides: `supports_thinking` (default False), `supports_tool_use` (default True).

#### `claude_client.py` — Anthropic SDK Wrapper

`ClaudeClient(BaseClient)` wraps the official `anthropic` Python SDK.

**Streaming protocol** (Anthropic SSE events):

```
message_start       → grab input_tokens, cache_read, cache_write
content_block_start → open new text or tool_use block
content_block_delta → append text chunk (calls on_text) or accumulate JSON
content_block_stop  → finalise block; parse JSON for tool_use
message_delta       → capture stop_reason + output_tokens
```

**Retry loop**: wraps `_do_stream()` in a `while True` loop; `classify_error()`
maps SDK exceptions to typed errors; `RetryableError` subclasses get up to
`max_retries` attempts with exponential back-off + ±25% jitter:

```python
delay = base * (2 ** attempt) + random.uniform(-jitter, jitter)
```

Extended thinking: when `thinking=True`, adds beta header `"interleaved-thinking-2025-05-14"`
and a `thinking` config block.

#### `openai_compat_client.py` — OpenAI / DeepSeek / Custom Endpoint

`OpenAICompatClient(BaseClient)` supports any OpenAI Chat Completions API.

Flow:
1. Convert Anthropic-format `messages` + `system` → OpenAI format via `messages_to_openai()`
2. Convert Anthropic tool defs → OpenAI `function` format via `tools_to_openai()`
3. Stream with `openai.OpenAI.chat.completions.create(stream=True)`
4. Accumulate text chunks and tool-call delta fragments
5. Assemble Anthropic-format content blocks + `TurnResult`
6. Map `finish_reason` → Anthropic `stop_reason` vocabulary

Configurable `base_url` allows: DeepSeek (`api.deepseek.com`), OpenAI (`api.openai.com/v1`),
Azure, local Ollama/llama.cpp.

#### `message_converter.py` — Bidirectional Format Translation

Pure functions (no state) for format conversion:

| Function | Direction |
|----------|-----------|
| `messages_to_openai(anthropic_msgs, system_prompt)` | Anthropic → OpenAI |
| `tools_to_openai(anthropic_tools)` | Anthropic → OpenAI function format |
| `openai_choice_to_content_blocks(choice)` | OpenAI response → Anthropic content blocks |

The trickiest conversion is tool results: Anthropic uses `{"role": "user", "content": [{"type": "tool_result", ...}]}`
while OpenAI uses `{"role": "tool", "tool_call_id": ..., "content": ...}`.

#### `client_factory.py` — Provider Routing

`create_client(provider, api_key, base_url, max_retries)` maps provider name to
the correct client class:
- `"anthropic"` → `ClaudeClient`
- `"deepseek"`, `"openai"`, or any custom string → `OpenAICompatClient`

`provider_from_model(model)` infers provider from model name prefix:
- `claude-*` → anthropic
- `deepseek-*` → deepseek
- `gpt-*`, `o1`, `o3`, `o4-*` → openai

#### `errors.py` — Typed Error Hierarchy

```
ClaudeAPIError
├── RetryableError
│   ├── RateLimitError          (HTTP 429)
│   ├── ServiceUnavailableError (HTTP 503/529)
│   └── NetworkError            (connection failures)
├── AuthenticationError         (HTTP 401)
├── PermissionDeniedError       (HTTP 403)
├── NotFoundError               (HTTP 404)
├── ValidationError             (HTTP 422)
├── QuotaExceededError          (monthly quota exhausted)
├── InsufficientBalanceError    (HTTP 402, DeepSeek balance=0)
├── ContextWindowExceededError  (prompt too long)
└── UnknownAPIError             (any other non-2xx)
```

`classify_error(exc)` converts raw Anthropic SDK exceptions to typed errors.
`OpenAICompatClient._classify(exc)` does the same for the openai SDK.

---

### 3.3 Config Layer (`config/`)

#### `settings.py` — Settings Dataclass + Loader

**`Settings`** is a flat `@dataclass` with sensible defaults covering:

| Group | Fields |
|-------|--------|
| Provider/Model | `provider`, `model`, `fallback_model`, `base_url`, `max_tokens`, `api_key` |
| Permissions | `permission_mode`, `allowed_tools`, `denied_tools` |
| System prompt | `system_prompt_extra`, `append_system_prompt`, `custom_system_prompt` |
| Session | `cwd`, `verbose`, `stream` |
| Budget | `max_budget_usd`, `task_budget_usd` |
| Thinking | `thinking_enabled`, `thinking_budget` |
| Persistence | `sessions_dir` |
| Context | `compact_threshold` |

**`SettingsLoader.load(overrides)`** merges 6 sources (lowest → highest priority):

```
built-in defaults
  → env vars (ANTHROPIC_API_KEY, CLAUDE_MODEL, CLAUDE_PROVIDER, …)
    → ~/.claude/config.json
      → ~/.claude/settings.json
        → .claude.json (in CWD)
          → CLI overrides (passed as dict)
```

`_validate()` enforces: valid `permission_mode`, `max_tokens` in [1, 100_000],
positive `max_budget_usd`, auto-infers `provider` from model name.

`save_user_settings(updates)` merges `updates` into `~/.claude/settings.json`
(creates the file if it does not exist).

---

### 3.4 Permissions Layer (`permissions/`)

#### `permission_manager.py` — Access Control

`PermissionManager` is a **stateful** evaluator for one session.

**`check(tool, input_data) → PermissionDecision`** applies guards in this order:

```
1. Explicit deny-list (tool name in self._denied)  → DENY
2. Dangerous-path guard (for Write/Edit tools)     → DENY if sensitive file/dir
3. Path-pattern deny rules (user-configured globs) → DENY
4. Path-pattern allow rules (user-configured globs)→ ALLOW  (short-circuit)
5. Explicit allow-list (tool name in self._allowed) → ALLOW
6. Mode == "bypass"                                 → ALLOW
7. Mode == "auto"   → _auto_check()
8. Mode == "default"→ _default_check()
```

**Dangerous path guard** blocks writes to:
- Dirs: `.ssh`, `.gnupg`, `.aws`, `.kube`, `.docker`
- Files: `.gitconfig`, `.bashrc`, `.zshrc`, `.env`, `id_rsa`, `authorized_keys`,
  `passwd`, `shadow`, `credentials`, and ~20 more

**Three permission modes**:

| Mode | Behaviour |
|------|-----------|
| `default` | Safe tools (`dangerous=False`) → auto-allow; dangerous tools → interactive prompt |
| `bypass` | Everything allowed (CI/automation use case) |
| `auto` | Heuristic classifier: read-only tools always allowed; dangerous bash patterns (rm -rf, mkfs, etc.) blocked; everything else allowed |

**Interactive prompt** (default mode): uses injected `prompt_fn` callback if provided
(enables rich-formatted prompts from main.py), otherwise falls back to stdin.
Accepts `y`/`n`/`a` — `a` switches the entire session to `bypass` mode.

All denials are appended to `denial_log` (list of `DenialEntry`) for session summary.

---

### 3.5 Session Layer (`session/`)

#### `session_manager.py` — Disk Persistence

Sessions are stored as JSON files in `~/.claude/sessions/<session_id>.json`.

**`Session`** dataclass fields:
- `id` — 12-char hex UUID
- `created_at`, `updated_at` — Unix timestamps
- `cwd`, `model`, `messages`, `total_cost`

**`SessionManager`** operations:
- `save(session)` — write to `~/.claude/sessions/<id>.json`; update `updated_at`
- `load(session_id)` → `Optional[Session]` — deserialise from JSON; returns None on missing/corrupt
- `list_sessions(limit=20)` — return most-recently-updated sessions (sorted by mtime)
- `delete(session_id)` → `bool`
- `append_message(session, message)` — append then immediately `save()` (incremental durability)

The incremental save pattern means sessions survive a process kill mid-conversation.

---

### 3.6 Tools Layer (`tools/`)

All tools inherit from `Tool` (defined in `tool.py`).  `build_default_registry()`
instantiates all 14 and registers them in a `ToolRegistry`.

#### Tool Summary Table

| Class | `name` | `dangerous` | Summary |
|-------|--------|-------------|---------|
| `BashTool` | `Bash` | ✓ | Run shell commands in a subprocess |
| `REPLTool` | `REPL` | ✓ | Run commands in a **persistent** bash session |
| `FileReadTool` | `Read` | ✗ | Read files with line numbers; .ipynb support |
| `FileWriteTool` | `Write` | ✓ | Create or overwrite files |
| `FileEditTool` | `Edit` | ✓ | Exact string replacement with diff output |
| `GlobTool` | `Glob` | ✗ | Find files by glob pattern |
| `GrepTool` | `Grep` | ✗ | Regex search via ripgrep or Python fallback |
| `WebFetchTool` | `WebFetch` | ✗ | Fetch URLs; HTML→text; SSRF guard |
| `WebSearchTool` | `WebSearch` | ✗ | Web search via Brave or DuckDuckGo |
| `TodoWriteTool` | `TodoWrite` | ✗ | CRUD on `~/.claude/todos.json` |
| `ConfigTool` | `Config` | ✗ | Read/update session settings |
| `NotebookEditTool` | `NotebookEdit` | ✓ | Read/edit Jupyter .ipynb cells |
| `SleepTool` | `Sleep` | ✗ | Pause execution (0.1–60 s) |
| `AskUserQuestionTool` | `AskUser` | ✗ | Ask user a clarifying question via stdin |

---

## 4. Data-Flow Walkthrough

Here is what happens when the user types `"Add a docstring to foo()"`:

```
1. Caller invokes: engine.run("Add a docstring to foo()")

2. QueryEngine
   ├── budget pre-check (CostTracker.totals.total_cost_usd)
   ├── history.add_user_message("Add a docstring to foo()")
   ├── build_system_prompt() → string (CWD + git + CLAUDE.md + extra)
   ├── registry.to_api_list() → [{name, description, input_schema}, …]
   └── TURN 1:
       ├── ClaudeClient.stream_turn(model, system, messages, tools, …)
       │   ├── POST /v1/messages (streaming)
       │   ├── on_text("I'll read…") → caller prints chunks
       │   └── returns TurnResult(
       │         tool_uses=[ToolUseBlock(id="tu_01", name="Read",
       │                                 input={"file_path":"/src/foo.py"})],
       │         stop_reason="tool_use", input_tokens=1200, …
       │       )
       ├── cost_tracker.record(model, 1200, 80, …)
       ├── history.add_assistant_message([{type:tool_use, …}])
       └── _execute_tool_uses([ToolUseBlock(…)])
           ├── registry.find("Read") → FileReadTool instance
           ├── permission_manager.check(FileReadTool, {file_path:…})
           │   → ALLOW  (dangerous=False)
           ├── FileReadTool.execute({file_path: "/src/foo.py"}) → ToolResult.ok("1\tdef foo()…")
           └── history.add_tool_result("tu_01", "1\tdef foo()…", is_error=False)

3. TURN 2:
   ├── ClaudeClient.stream_turn(…) — now messages includes tool result
   │   on_text("```python\ndef foo():\n    \"\"\"…\"\"\"\n```\n")
   │   returns TurnResult(tool_uses=[ToolUseBlock(name="Edit", …)], …)
   └── _execute_tool_uses → FileEditTool.execute → ToolResult.ok("Replaced 1 …")

4. TURN 3:
   ├── ClaudeClient.stream_turn → stop_reason="end_turn"
   └── loop breaks → return final text
```

---

## 5. File-by-File Reference

### Root-Level Files

---

#### `__init__.py`

**Role**: Public package surface.

Re-exports the most commonly used classes so callers can write:
```python
from claude_code import QueryEngine, Settings, ToolRegistry
```

Exports: `QueryEngine`, `Settings`, `SettingsLoader`, `Tool`, `ToolResult`,
`ToolRegistry`, `PermissionDecision`, `HistoryManager`, `CostTracker`,
`build_system_prompt`, `PermissionManager`, `SessionManager`, `Session`,
`ClaudeClient`, `build_default_registry`.

---

#### `query_engine.py`

**Role**: Agentic loop orchestrator — the central coordinator of the entire system.

**Class**: `QueryEngine`

**Constructor parameters**:
- `settings: Settings` — model, budget, thinking config
- `client: BaseClient` — primary LLM client
- `registry: ToolRegistry` — all available tools
- `permission_manager: PermissionManager` — access control
- `history: HistoryManager` — conversation memory (shared across `run()` calls)
- `cost_tracker: Optional[CostTracker]` — token/cost accounting
- `fallback_client: Optional[BaseClient]` — secondary client for error recovery

**Key method**: `run(user_message, on_text, on_tool_start, on_tool_end, on_turn_complete) → str`

Callbacks allow callers to stream output (`on_text`), show tool progress
(`on_tool_start`, `on_tool_end`), and display per-turn stats (`on_turn_complete`).

**Safety constant**: `MAX_TURNS = 150` — hard cap on agentic iterations per query.

---

#### `tool.py`

**Role**: Base contract for all tools.

**`PermissionDecision`** (Enum):
- `ALLOW` — proceed without asking
- `DENY` — block and return error result
- `ASK` — not used directly by tools; reserved for UI layer

**`ToolResult`** (dataclass):
- `content: str` — always present; the tool's output
- `is_error: bool` — signals failure to the API (Claude sees an error tool-result block)
- `metadata: dict` — optional, not sent to the API; used for UI display

**`Tool`** (abstract class):
- Class attrs define the API contract: `name`, `description`, `input_schema`, `dangerous`
- `validate_input()` checks required fields from `input_schema["required"]`
- `execute()` **must** be overridden; receives `input_data` and optional `permission_manager`
- `check_permission()` calls `permission_manager.check(tool=self, input_data=…)`
- `to_api_dict()` serialises to `{"name":…, "description":…, "input_schema":…}`

**`ToolRegistry`** (container):
- Insertion-ordered dict mapping `name → Tool`
- `register(tool)` raises `ValueError` on duplicate name
- `to_api_list()` returns the list passed as `tools=` to the Anthropic SDK

---

#### `history.py`

**Role**: In-memory conversation history with automatic context-window compaction.

**Class**: `HistoryManager`

**Constructor**: `max_tokens=8192`, `compact_threshold=0.85`

**Message format** (Anthropic SDK dict):
```python
{"role": "user", "content": "text or list of blocks"}
{"role": "assistant", "content": [{"type": "text", "text": "…"}, …]}
{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "…", "content": "…"}]}
```

**Compaction trigger**: when `estimated_tokens > max_tokens × compact_threshold`

**Compaction cut rule**: find first assistant message at or after the midpoint
(search forward, then backward) so the API's user↔assistant alternation is preserved.

**Token estimation**: `total_chars / 3.5` (conservative approximation; avoids tokeniser dependency).

---

#### `cost_tracker.py`

**Role**: Accumulate and display token usage and USD cost across all turns.

**`TurnUsage`** (dataclass): records one API call's usage; `cost_usd` is computed
in `__post_init__` using `_price_for(model)`.

**`SessionCosts`** (dataclass): aggregated totals; returned by `CostTracker.totals` property.

**Price table** (`_PRICE_TABLE`): prefix-matched against model name.
Supports Anthropic (Claude 3/4), DeepSeek (V3/R1), OpenAI (GPT-4o, o1, o3, o4-mini).

**Cache pricing** (industry-standard Anthropic rates):
- Cache read: 10% of normal input price
- Cache write: 125% of normal input price

**Display methods**:
- `summary()` — one-line session total (turns, tokens, cost)
- `turn_summary(turn)` — one-line per-turn breakdown
- `per_model_summary()` — multi-line grouped by model (useful for mixed-model sessions)

---

#### `context.py`

**Role**: Build the system prompt injected at the start of every request.

**`build_system_prompt(cwd, extra, append, custom, include_git, include_date) → str`**

If `custom` is provided it replaces the core prompt (but git/CLAUDE.md/extra still append).

**`get_runtime_context() → str`** (`@lru_cache`):
- Probes Python interpreter names: `python3.14` → `python3.9`, `python2`, `python`
- Emits a `WARNING` when `python` resolves to Python 2 (common on macOS)
- Also probes `node`

**`get_git_context(cwd) → str`**:
- `git rev-parse --show-toplevel` → returns `""` if not a git repo
- Runs: `rev-parse HEAD`, `status --short`, `log --oneline -5`, `config user.name`
- Truncates status at 2000 chars

**`_load_claude_md(cwd) → str`**:
- Walks from `cwd` up to filesystem root, then `~/.claude/`
- First `CLAUDE.md` or `claude.md` found is returned with a header
- Returns `""` if none found

---

### `api/` Files

---

#### `api/__init__.py`

Empty; marks `api` as a Python package.

---

#### `api/base_client.py`

**Role**: Provider interface — the only type `QueryEngine` depends on.

**Abstract method**: `stream_turn(*, model, system, messages, tools, max_tokens, on_text, thinking, thinking_budget) → TurnResult`

Inputs always in Anthropic format; each subclass converts internally.

Optional properties: `supports_thinking` (default `False`), `supports_tool_use` (default `True`).

---

#### `api/claude_client.py`

**Role**: Anthropic SDK wrapper implementing `BaseClient`.

**`ToolUseBlock`** (dataclass): `id: str`, `name: str`, `input: Dict` — one tool call from an assistant response.

**`TurnResult`** (dataclass): `message`, `tool_uses`, `stop_reason`, `input_tokens`,
`output_tokens`, `cache_read_tokens`, `cache_write_tokens`.

**`ClaudeClient`**:
- Lazy-imports `anthropic` in `__init__` to keep the class importable without the SDK installed
- `stream_turn()` → retry loop calling `_do_stream()`
- `_do_stream()` → streaming context manager; accumulates content blocks
- `_backoff(attempt)` → `base × 2^attempt ± 25% jitter`

**Extended thinking**: adds `betas=["interleaved-thinking-2025-05-14"]` and `thinking={type, budget_tokens}` to the stream kwargs.

---

#### `api/openai_compat_client.py`

**Role**: OpenAI-compatible client for DeepSeek, OpenAI, Azure, Ollama, etc.

**`PROVIDER_BASE_URLS`**: `{"deepseek": "https://api.deepseek.com", "openai": "https://api.openai.com/v1"}`

**`OpenAICompatClient`**:
- Wraps `openai.OpenAI(api_key=…, base_url=…)`
- `_do_stream()`: accumulates streamed `delta.content` chunks and `delta.tool_calls` fragments indexed by `idx`
- Maps `finish_reason` → Anthropic `stop_reason`: `"stop"→"end_turn"`, `"tool_calls"→"tool_use"`, `"length"→"max_tokens"`
- `_classify(exc)` → maps openai SDK exceptions; HTTP 402 → `InsufficientBalanceError` (DeepSeek billing)
- Does NOT support `thinking` (ignored silently)

---

#### `api/message_converter.py`

**Role**: Pure-function bidirectional format translation.

**`messages_to_openai(anthropic_messages, system_prompt)`**:
- Prepends `{"role": "system", "content": system_prompt}`
- `tool_result` blocks → `{"role": "tool", "tool_call_id": …, "content": …}`
- `tool_use` blocks → `{"role": "assistant", "tool_calls": [{…}]}`

**`tools_to_openai(anthropic_tools)`**:
- `{"name", "description", "input_schema"}` → `{"type":"function","function":{…,"parameters":…}}`

**`openai_choice_to_content_blocks(choice)`**:
- `choice.message.content` → `{"type":"text","text":…}`
- `choice.message.tool_calls[i]` → `{"type":"tool_use","id","name","input"}`

---

#### `api/client_factory.py`

**Role**: Factory that selects and constructs the right `BaseClient`.

**`create_client(provider, api_key, base_url, max_retries) → BaseClient`**:
1. Resolve API key from argument or env var (`_API_KEY_ENV` mapping)
2. `"anthropic"` → `ClaudeClient`
3. All others → `OpenAICompatClient` with `PROVIDER_BASE_URLS.get(provider, "https://api.openai.com/v1")`

**`provider_from_model(model) → str`**:
- `claude-*` → `"anthropic"`
- `deepseek-*` → `"deepseek"`
- `gpt-*`, `o1`, `o3`, `o4-*` → `"openai"`
- fallback → `"anthropic"`

---

#### `api/errors.py`

**Role**: Typed error hierarchy for provider-agnostic error handling.

`classify_error(exc) → ClaudeAPIError` converts raw Anthropic SDK exceptions:
- `anthropic.AuthenticationError` → `AuthenticationError(status_code=401)`
- `anthropic.RateLimitError` → `RateLimitError(status_code=429)`
- `anthropic.APIConnectionError` → `NetworkError`
- Falls back to `NetworkError` for non-Anthropic exceptions (e.g. network timeouts)

---

### `config/` Files

---

#### `config/__init__.py`

Empty; marks `config` as a Python package.

---

#### `config/settings.py`

**Role**: Flat settings dataclass + multi-source loader.

**`Settings`** (dataclass, ~20 fields): see [3.3 Config Layer](#33-config-layer-config).

**`SettingsLoader`**:
- `load(overrides) → Settings`
- `_from_env() → Dict` — reads `ANTHROPIC_API_KEY`, `CLAUDE_MODEL`, `CLAUDE_PROVIDER`, `CLAUDE_BASE_URL`, `CLAUDE_PERMISSION_MODE`, `CLAUDE_VERBOSE`, `CLAUDE_THINKING`
- `_read_json(path) → Dict` — silently returns `{}` on missing/malformed files
- `_validate(data) → Settings` — filters to known keys; validates `permission_mode`, `max_tokens`, `max_budget_usd`; auto-infers `provider` from model name
- `save_user_settings(updates)` — merges into `~/.claude/settings.json`; creates directory if needed

---

### `permissions/` Files

---

#### `permissions/__init__.py`

Empty; marks `permissions` as a Python package.

---

#### `permissions/permission_manager.py`

**Role**: Stateful per-session access control evaluator.

**`DenialEntry`** (dataclass): `tool_name`, `input_summary` (≤120 chars), `reason` string.

**`PermissionManager`** constructor:
- `mode: str` — `"default"` | `"bypass"` | `"auto"`
- `allowed_tools: List[str]` — explicit allow-list (bypasses mode logic)
- `denied_tools: List[str]` — explicit deny-list (overrides everything)
- `interactive: bool` — when False, dangerous tools → DENY without prompting
- `allowed_paths / denied_paths: List[str]` — glob patterns for file-write operations
- `prompt_fn: Optional[Callable]` — injected prompt renderer; signature `(tool_name, summary, full_input) → str` returning `"yes"/"no"/"all"`

**`_auto_check`** heuristic:
- Read-only tools (`Read`, `Glob`, `Grep`, `WebFetch`, `WebSearch`) → always ALLOW
- Bash with `rm -rf`, `mkfs`, `dd if=`, `> /dev/`, fork bomb → DENY
- Everything else → ALLOW

**`_check_path_safety`**: walks `Path.parts` for dangerous directories; checks `path.name.lower()` against `DANGEROUS_FILES` frozenset.

---

### `session/` Files

---

#### `session/__init__.py`

Empty; marks `session` as a Python package.

---

#### `session/session_manager.py`

**Role**: Disk-based conversation persistence.

**`Session`** (dataclass): auto-generates `id` as 12-char hex; `created_at`/`updated_at` as `time.time()`.

`to_dict()` / `from_dict()` use `dataclasses.asdict` and field-name filtering for forward compatibility.

**`SessionManager`**:
- Default directory: `~/.claude/sessions/`
- `list_sessions(limit)` — glob `*.json`, sort by `st_mtime` (descending), return up to `limit`
- `append_message(session, message)` — append to in-memory list AND write to disk immediately (crash safety)

---

### `tools/` Files

---

#### `tools/__init__.py`

**Role**: Assembles the default registry.

`build_default_registry(settings=None) → ToolRegistry`:
Imports and registers all 14 tools in this order:
1. `FileReadTool`, `FileWriteTool`, `FileEditTool`, `GlobTool`, `GrepTool`
2. `BashTool`, `REPLTool`, `SleepTool`
3. `WebFetchTool`, `WebSearchTool`
4. `TodoWriteTool`
5. `ConfigTool(settings=settings)` — receives live settings object
6. `NotebookEditTool`
7. `AskUserQuestionTool`

---

#### `tools/bash_tool.py`

**`BashTool`** (`name="Bash"`, `dangerous=True`)

**Input schema**: `command` (required), `timeout` (1–600 s, default 120), `cwd` (optional).

**Implementation**:
1. Permission check — DENY → return error result
2. Clamp timeout to [1, 600]
3. `subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout)`
4. Combine stdout + stderr; **tail-truncate** at 50,000 chars (error messages are at the end)
5. Call `interpret_exit_code(command, returncode, stdout, stderr)`

**`interpret_exit_code()`** special cases:
- `grep`/`rg`/`ag`/`ack`: exit 1 = "No matches found" (not an error)
- `diff`: exit 1 = "Files differ" (not an error)
- `find`: exit 1 = "Some directories were inaccessible" (not an error)
- `test`/`[`: exit 0 or 1 = not an error; 2+ = error
- All others: exit ≠ 0 → error

**`is_readonly_command(command) → bool`**: heuristic that checks for mutating
shell patterns (`rm`, `>`, `curl | sh`, `pip install`, etc.) and read-only
command prefixes (`cat`, `ls`, `git log`, `grep`, etc.).

---

#### `tools/file_read_tool.py`

**`FileReadTool`** (`name="Read"`, `dangerous=False`)

**Input schema**: `file_path` (required), `offset` (1-indexed line, default 1), `limit` (default 2000).

**Guards**:
- Device paths (`/dev/zero`, `/dev/tty`, etc.) → error
- File size > 256 KB → error with instruction to use Bash head/tail
- Directory path → error

**Features**:
- `cat -n` style output: `"1\tline content\n2\t…"`
- Shows `"[Showing lines X–Y of Z. Use offset=… to read more.]"` when truncated
- `.ipynb` files → cell-by-cell rendering via `_read_notebook()`
- `_read_cached(path)` — mtime-keyed dict cache avoids re-reading unchanged files
- `_find_similar_file(path)` — `difflib.get_close_matches` suggests alternative filenames

---

#### `tools/file_write_tool.py`

**`FileWriteTool`** (`name="Write"`, `dangerous=True`)

**Input schema**: `file_path` (required), `content` (required).

**Guards**:
- Permission check first
- Blocks paths starting with `/etc/`, `/sys/`, `/proc/`, `/dev/`, `/boot/`
- `PermissionManager._check_path_safety()` also blocks `.ssh/`, `~/.bashrc`, etc.

**Features**:
- `Path.parent.mkdir(parents=True, exist_ok=True)` — creates the full directory tree
- Detects new vs existing file: response says "Created" vs "Updated"
- Returns line count and byte count in result

---

#### `tools/file_edit_tool.py`

**`FileEditTool`** (`name="Edit"`, `dangerous=True`)

**Input schema**: `file_path`, `old_string`, `new_string` (all required), `replace_all` (default False).

**Guards**:
- File must exist (returns error suggesting `Write` for new files)
- File size ≤ 10 MB
- `old_string` must be present in the file (returns error if not found)
- If `old_string` appears multiple times and `replace_all=False` → error

**Features**:
- Reads raw bytes to detect **CRLF vs LF**; writes back in the same style
- **Curly-quote fallback** (`_find_with_quote_fallback`): tries exact match, then
  normalises both curly and straight quotes to find a near-match
- **.json validation**: after edit, attempts `json.loads(modified)`; rejects edit on parse error
- Returns a `difflib.unified_diff` with 3 lines of context so Claude can verify the change

---

#### `tools/glob_tool.py`

**`GlobTool`** (`name="Glob"`, `dangerous=False`)

**Input schema**: `pattern` (required), `path` (default CWD), `follow_symlinks` (default False).

**Implementation**:
- `root.glob(pattern)` using Python's `pathlib`
- Filters to files only; symlink handling via `follow_symlinks` flag
- Sorts results by `st_mtime` descending (newest first)
- Caps at 500 results; shows count warning if truncated

---

#### `tools/grep_tool.py`

**`GrepTool`** (`name="Grep"`, `dangerous=False`)

**Input schema**: `pattern` (required), `path`, `glob`, `type`, `-i`, `-A`, `-B`, `-C`, `multiline`, `output_mode`, `head_limit`, `offset`, `-n`.

**Two implementations**:
1. **ripgrep** (`rg`) — tried first via `subprocess.run(["rg", …])`; falls back on `FileNotFoundError`
2. **Pure Python** — `re.compile(pattern, flags)` + manual file walk

Both support:
- Context lines (`-A`, `-B`, `-C`)
- Three output modes: `"content"` (matching lines), `"files_with_matches"`, `"count"`
- Pagination: `offset` + `head_limit`
- VCS exclusion: `.git`, `node_modules`, `__pycache__`, `.pytest_cache`, etc.

The Python fallback uses `re.DOTALL` for multiline mode.

---

#### `tools/web_fetch_tool.py`

**`WebFetchTool`** (`name="WebFetch"`, `dangerous=False`)

**Input schema**: `url` (required), `prompt` (optional focus query), `timeout` (1–120 s, default 30).

**Guards**:
- Only `http://` and `https://` schemes allowed
- **SSRF guard**: `_is_private_ip()` uses `socket.gethostbyname` + `ipaddress.ip_address.is_private`

**Implementation**:
- Uses `httpx` (HTTP/2 support, connection pooling) with `follow_redirects=True`
- HTML → plain text via `_html_to_text()`: removes `<script>`/`<style>`, converts block tags to newlines, strips all tags
- Truncates at 30,000 chars
- **Prompt filter** (`_apply_prompt_filter`): splits text into paragraphs, scores each by keyword overlap with `prompt`, returns top-scoring paragraphs in document order (up to 30,000 chars)

---

#### `tools/web_search_tool.py`

**`WebSearchTool`** (`name="WebSearch"`, `dangerous=False`)

**Input schema**: `query` (required), `num_results` (max 10, default 5).

**Two backends**:
1. **Brave Search API** — used when `BRAVE_API_KEY` env var is set;
   `GET https://api.search.brave.com/res/v1/web/search?q=…&count=N`
2. **DuckDuckGo Instant Answer** — fallback; `GET https://api.duckduckgo.com/?q=…&format=json`

Backend selection: `CLAUDE_SEARCH_BACKEND` env var, or auto-select based on API key presence.

Output: formatted as `**Title**\nURL\nSnippet` blocks joined by `\n\n`.

---

#### `tools/todo_tool.py`

**`TodoWriteTool`** (`name="TodoWrite"`, `dangerous=False`)

Persistent storage: `~/.claude/todos.json` — a JSON array shared across sessions.

**Operations** (dispatched via `operation` field):
- `list` — load and format all todos
- `create` — append new task; auto-generates `t<n>` ID; validates `status` and `priority`
- `update` — find by `id`; patch any of `content`, `status`, `priority`
- `delete` — filter out by `id`

**Status icons** in `_format_todos()`: `○` pending, `◑` in_progress, `●` completed.

---

#### `tools/config_tool.py`

**`ConfigTool`** (`name="Config"`, `dangerous=False`)

Holds a reference to the live `Settings` object injected at construction time.

**Operations**:
- `get` — returns all settings as JSON (masks `api_key` after first 8 chars + `…`)
- `set` — only allows keys in `_SETTABLE_KEYS` (9 keys: `model`, `max_tokens`, `verbose`, etc.);
  updates in-memory Settings object AND persists to `~/.claude/settings.json` via `SettingsLoader.save_user_settings()`

---

#### `tools/notebook_edit_tool.py`

**`NotebookEditTool`** (`name="NotebookEdit"`, `dangerous=True`)

Only `.ipynb` files accepted (checked by filename suffix).

**Operations**:
- `read` — render all cells as `[index] CELL_TYPE\nsource\n  Output: …`
- `edit_cell` — replace `cells[idx]["source"]`; clears `cells[idx]["outputs"]` (mirrors Jupyter UI)
- `insert_cell` — `cells.insert(idx, new_cell)`; constructs valid cell structure for `"code"`, `"markdown"`, or `"raw"`
- `delete_cell` — `cells.pop(idx)`

`_load(nb_path)` and `_save(nb_path, data)` are static helpers returning `(data, None)` or `(None, ToolResult.error(...))`.

---

#### `tools/sleep_tool.py`

**`SleepTool`** (`name="Sleep"`, `dangerous=False`)

**Input schema**: `seconds` (required, clamped to [0.1, 60]).

Calls `time.sleep(duration)`. Returns result noting if value was clamped.

Use case: agentic loops waiting for a build, deployment, or service restart.

---

#### `tools/ask_user_tool.py`

**`AskUserQuestionTool`** (`name="AskUser"`, `dangerous=False`)

**Input schema**: `question` (required), `options` (optional list of suggested answers).

**Interactive mode** (`interactive=True`):
- Prints question and optional numbered options
- Reads line from `stdin` via `input()`
- Returns typed answer; handles `EOFError`/`KeyboardInterrupt` gracefully

**Non-interactive mode** (`interactive=False`):
- Returns `"(Non-interactive mode: no user response available.)"` immediately

---

#### `tools/repl_tool.py`

**`REPLTool`** (`name="REPL"`, `dangerous=True`)

**Input schema**: `command` (required), `session_id` (default `"default"`), `timeout` (default 30, max 300), `cwd` (for new sessions only).

**`_PersistentShell`** class:
- Spawns `bash --norc --noprofile` as a long-lived subprocess
- Background daemon thread (`_read_loop`) drains stdout into a `queue.Queue`
- `run(command, timeout)`:
  1. Acquires `threading.Lock`
  2. Writes `{command} 2>&1\necho '{sentinel}'\n` to `stdin`
  3. Drains queue until sentinel line is seen
  4. Returns joined output lines
- `terminate()` — closes stdin, calls `process.terminate()`, then `process.kill()` if needed

**Session registry** (`_shells: Dict[str, _PersistentShell]`):
- Thread-safe via `_shells_lock`
- `_get_or_create_shell()` — creates new if absent or process has exited
- `_reset_shell()` — terminates old, creates new
- `atexit.register(_cleanup_all_shells)` — kills all shells at process exit

**Command `"__reset__"`**: special value that triggers `_reset_shell()`.

On timeout or `OSError`: resets the session automatically (avoids leaving a hung shell).

---

*Document generated 2026-04-20 for `claude_code` package.*
