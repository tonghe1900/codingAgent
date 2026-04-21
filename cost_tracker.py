"""
cost_tracker.py
===============
Tracks token usage and estimated USD cost across turns in a session.

Pricing is stored as a simple dict keyed by model prefix so it degrades
gracefully when a new model is released before we update the table.

Usage
-----
    tracker = CostTracker()
    tracker.record(model="claude-opus-4-6", input_tokens=500, output_tokens=200)
    print(tracker.summary())
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List


# ---------------------------------------------------------------------------
# Pricing table  (USD per million tokens, as of mid-2025)
# ---------------------------------------------------------------------------

# Each entry: (input_cost_per_M, output_cost_per_M)
# Pricing as of mid-2025 (USD).
_PRICE_TABLE: Dict[str, tuple[float, float]] = {
    # --- Anthropic ---
    "claude-opus-4":     (15.00,  75.00),
    "claude-sonnet-4":   (3.00,   15.00),
    "claude-haiku-4":    (0.80,    4.00),
    "claude-3-5-sonnet": (3.00,   15.00),
    "claude-3-5-haiku":  (0.80,    4.00),
    "claude-3-opus":     (15.00,  75.00),
    # --- DeepSeek ---
    "deepseek-chat":     (0.27,    1.10),   # DeepSeek-V3 (cache miss)
    "deepseek-reasoner": (0.55,    2.19),   # DeepSeek-R1 (cache miss)
    # --- OpenAI (more specific prefixes must come before shorter ones) ---
    "gpt-4o-mini":       (0.15,    0.60),
    "gpt-4o":            (2.50,   10.00),
    "o1":                (15.00,  60.00),
    "o3":                (10.00,  40.00),
    "o4-mini":           (1.10,    4.40),
}

_DEFAULT_PRICE = (3.00, 15.00)  # fallback when model not in table


def _price_for(model: str) -> tuple[float, float]:
    """Return (input_$/M, output_$/M) for *model*, matching by prefix."""
    for prefix, prices in _PRICE_TABLE.items():
        if model.startswith(prefix):
            return prices
    return _DEFAULT_PRICE


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TurnUsage:
    """Token counts and cost for a single API turn."""
    timestamp: float
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0      # prompt-cache read (cheaper than normal input)
    cache_write_tokens: int = 0     # prompt-cache write (stored for future reads)
    cost_usd: float = field(init=False)

    def __post_init__(self) -> None:
        in_price, out_price = _price_for(self.model)
        # Cache reads are billed at 10 % of normal input price.
        # Cache writes are billed at 125 % of normal input price.
        self.cost_usd = (
            self.input_tokens * in_price / 1_000_000
            + self.output_tokens * out_price / 1_000_000
            + self.cache_read_tokens * in_price * 0.10 / 1_000_000
            + self.cache_write_tokens * in_price * 1.25 / 1_000_000
        )


@dataclass
class SessionCosts:
    """Aggregated totals for the whole session."""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cache_write_tokens: int = 0
    total_cost_usd: float = 0.0
    turn_count: int = 0


# ---------------------------------------------------------------------------
# Main tracker
# ---------------------------------------------------------------------------

class CostTracker:
    """
    Accumulates per-turn usage records and exposes session-level totals.

    Thread-safety: not required (single-threaded interactive CLI).
    """

    def __init__(self) -> None:
        self._turns: List[TurnUsage] = []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> TurnUsage:
        """
        Record one API turn's token usage.

        Returns the :class:`TurnUsage` so callers can log it immediately.
        """
        turn = TurnUsage(
            timestamp=time.time(),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
        )
        self._turns.append(turn)
        return turn

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    @property
    def totals(self) -> SessionCosts:
        """Return aggregated session totals (recomputed on access)."""
        sc = SessionCosts()
        for t in self._turns:
            sc.total_input_tokens += t.input_tokens
            sc.total_output_tokens += t.output_tokens
            sc.total_cache_read_tokens += t.cache_read_tokens
            sc.total_cache_write_tokens += t.cache_write_tokens
            sc.total_cost_usd += t.cost_usd
            sc.turn_count += 1
        return sc

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Human-readable one-liner for display in the REPL status bar."""
        t = self.totals
        tokens = t.total_input_tokens + t.total_output_tokens
        return (
            f"turns={t.turn_count}  "
            f"tokens={tokens:,}  "
            f"cost=${t.total_cost_usd:.4f}"
        )

    def turn_summary(self, turn: TurnUsage) -> str:
        """Human-readable one-liner for a single turn (shown after each response)."""
        return (
            f"[{turn.model}]  "
            f"in={turn.input_tokens:,}  out={turn.output_tokens:,}  "
            f"cost=${turn.cost_usd:.4f}"
        )

    def per_model_summary(self) -> str:
        """
        Return a multi-line breakdown of token usage and cost grouped by model.

        Useful when a session uses multiple models (e.g. primary + fallback,
        or DeepSeek alongside Claude).  Output example::

            Model breakdown:
              claude-sonnet-4-6  turns=3  in=1,500  out=800  cost=$0.0285
              claude-haiku-4-5   turns=1  in=200    out=50   cost=$0.0004
        """
        from collections import defaultdict

        # Aggregate per-model counters.
        model_data: Dict[str, Dict] = defaultdict(lambda: {
            "turns": 0, "in": 0, "out": 0,
            "cache_read": 0, "cache_write": 0, "cost": 0.0,
        })
        for t in self._turns:
            d = model_data[t.model]
            d["turns"]      += 1
            d["in"]         += t.input_tokens
            d["out"]        += t.output_tokens
            d["cache_read"] += t.cache_read_tokens
            d["cache_write"]+= t.cache_write_tokens
            d["cost"]       += t.cost_usd

        if not model_data:
            return "No usage recorded."

        lines = ["Model breakdown:"]
        for model, d in sorted(model_data.items()):
            lines.append(
                f"  {model:<30}  turns={d['turns']}  "
                f"in={d['in']:,}  out={d['out']:,}  "
                f"cost=${d['cost']:.4f}"
            )
        return "\n".join(lines)

    def reset(self) -> None:
        """Clear all recorded turns (used when starting a new session)."""
        self._turns.clear()
