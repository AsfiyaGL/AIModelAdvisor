"""
Token economics for Claude: monthly cost of a workload under three scenarios.

  1. All on Opus 5      — every request on the frontier model (the "no routing" baseline)
  2. Routed             — each task type on its Claude match (claude_equivalents.py)
  3. Routed + optimised — routing plus prompt caching and the Batch API

Pure functions, no API calls. Every number that is an assumption lives in
`Assumptions` so the UI can show it and let people change it.
"""

from dataclasses import dataclass, field

import pandas as pd

from claude_equivalents import CLAUDE_MATRIX, OPUS, ClaudeModel

# Anthropic pricing multipliers (relative to the model's base input/output price).
CACHE_READ_MULTIPLIER = 0.10     # cached input tokens read back
CACHE_WRITE_MULTIPLIER = 1.25    # input tokens written to the cache (5-minute TTL)
BATCH_DISCOUNT = 0.50            # Message Batches API

BASELINE_EFFORT = "high"         # effort assumed for the all-on-Opus baseline

SCENARIOS = ["All on Opus 5", "Routed", "Routed + optimised"]


@dataclass
class Assumptions:
    tokenizer_factor: float = 1.0
    """Multiply token counts by this to convert them to Claude tokens (1.0 = already Claude)."""

    thinking_overhead: dict[str, float] = field(default_factory=lambda: {
        "low": 0.25, "medium": 0.5, "high": 1.0, "xhigh": 2.0, "max": 3.0,
    })
    """Extra thinking tokens, as a multiple of visible output tokens, per effort level.
    Billed as output. Haiku 4.5 (no effort setting) is assumed not to think."""

    cacheable_share: float = 0.5
    """Share of each request's input that is a stable, cacheable prefix (system prompt, docs)."""

    cache_hit_rate: float = 0.8
    """Share of cacheable-prefix reads that hit the cache (the rest are cache writes)."""

    batch_share: float = 0.0
    """Share of requests that can run through the Batch API (not latency-sensitive)."""


def default_mix() -> pd.DataFrame:
    """A plausible starting workload. Share is % of monthly requests."""
    rows = [
        # category,   complexity, share %, avg input tok, avg output tok
        ("coding",     "low",     8,   800,  400),
        ("coding",     "medium", 10,  1500, 1200),
        ("coding",     "high",    4,  3000, 3000),
        ("reasoning",  "low",     4,   500,  300),
        ("reasoning",  "medium",  6,  1200,  800),
        ("reasoning",  "high",    3,  2500, 2000),
        ("writing",    "low",    10,   300,  250),
        ("writing",    "medium",  8,   600,  500),
        ("writing",    "high",    3,  1000, 1500),
        ("extraction", "low",    15,   600,  100),
        ("extraction", "medium", 10,  2000,  300),
        ("extraction", "high",    4,  4000,  800),
        ("general",    "low",     8,   200,  200),
        ("general",    "medium",  5,   800,  500),
        ("general",    "high",    2,  2000, 1500),
    ]
    return pd.DataFrame(rows, columns=["Category", "Complexity", "Share %",
                                       "Avg input tokens", "Avg output tokens"])


def cost_per_request(model: ClaudeModel, effort: str | None, input_tokens: float,
                     output_tokens: float, a: Assumptions, *,
                     use_cache: bool, use_batch: bool) -> float:
    in_tok = input_tokens * a.tokenizer_factor
    out_tok = output_tokens * a.tokenizer_factor
    thinking_tok = out_tok * a.thinking_overhead.get(effort, 0.0) if effort else 0.0

    input_multiplier = 1.0
    if use_cache:
        c, h = a.cacheable_share, a.cache_hit_rate
        input_multiplier = (1 - c) + c * (h * CACHE_READ_MULTIPLIER + (1 - h) * CACHE_WRITE_MULTIPLIER)

    cost = (in_tok * input_multiplier * model.input_cost_per_m
            + (out_tok + thinking_tok) * model.output_cost_per_m) / 1_000_000
    if use_batch:
        cost *= 1 - a.batch_share * BATCH_DISCOUNT
    return cost


def cost_breakdown(model: ClaudeModel, effort: str | None, input_tokens: float,
                   output_tokens: float, a: Assumptions) -> dict:
    """Tokens and list-price cost of one request, split into input / output / thinking."""
    in_tok = input_tokens * a.tokenizer_factor
    out_tok = output_tokens * a.tokenizer_factor
    thinking_tok = out_tok * a.thinking_overhead.get(effort, 0.0) if effort else 0.0
    return {
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "thinking_tokens": thinking_tok,
        "input_cost": in_tok * model.input_cost_per_m / 1_000_000,
        "output_cost": out_tok * model.output_cost_per_m / 1_000_000,
        "thinking_cost": thinking_tok * model.output_cost_per_m / 1_000_000,
    }


def default_output_tokens(category: str, complexity: str) -> int:
    mix = default_mix()
    row = mix[(mix["Category"] == category) & (mix["Complexity"] == complexity)]
    return int(row["Avg output tokens"].iloc[0]) if len(row) else 800


def compute(mix: pd.DataFrame, monthly_requests: int, a: Assumptions) -> pd.DataFrame:
    """One row per (scenario, task type) with the Claude model used and the monthly cost."""
    total_share = mix["Share %"].sum()
    rows = []
    for _, r in mix.iterrows():
        share = r["Share %"] / total_share if total_share else 0.0
        requests = monthly_requests * share
        rec = CLAUDE_MATRIX[(r["Category"], r["Complexity"])]
        setups = {
            "All on Opus 5": (OPUS, BASELINE_EFFORT, False, False),
            "Routed": (rec.model, rec.effort, False, False),
            "Routed + optimised": (rec.model, rec.effort, True, True),
        }
        for scenario, (model, effort, cache, batch) in setups.items():
            per_req = cost_per_request(model, effort, r["Avg input tokens"], r["Avg output tokens"],
                                       a, use_cache=cache, use_batch=batch)
            rows.append({
                "Scenario": scenario,
                "Category": r["Category"],
                "Complexity": r["Complexity"],
                "Claude model": model.model_id,
                "Effort": effort or "—",
                "Requests / month": requests,
                "Cost / request": per_req,
                "Monthly cost": per_req * requests,
            })
    return pd.DataFrame(rows)


def mix_from_classified(classified: pd.DataFrame, fallback: pd.DataFrame) -> pd.DataFrame:
    """Build a mix from classified real prompts.

    `classified` has columns Category, Complexity, Input tokens (estimated). Output
    tokens can't be known from a prompt alone, so they come from `fallback`.
    """
    counts = classified.groupby(["Category", "Complexity"]).agg(
        n=("Input tokens", "size"), avg_in=("Input tokens", "mean")).reset_index()
    mix = fallback.copy()
    mix["Share %"] = 0.0
    for _, c in counts.iterrows():
        sel = (mix["Category"] == c["Category"]) & (mix["Complexity"] == c["Complexity"])
        mix.loc[sel, "Share %"] = round(100 * c["n"] / len(classified), 1)
        mix.loc[sel, "Avg input tokens"] = int(round(c["avg_in"]))
    return mix
