"""
Water footprint of a Claude workload: an estimate built from tokens.

    IT energy (kWh)  = tokens x energy per token for the model
    On-site water    = IT energy x WUE           (data-centre cooling)
    Off-site water   = IT energy x PUE x EWIF    (water used to generate the electricity)

This follows the scope split used in "Making AI Less Thirsty" (Li et al., 2023).
Anthropic does not publish energy or water per token, so every factor here is an
assumption kept in `WaterAssumptions` for the UI to show and change. Treat the
result as an order-of-magnitude estimate, not a measurement.
"""

from dataclasses import dataclass, field

import pandas as pd

import token_economics as te
from claude_equivalents import CLAUDE_MATRIX, OPUS, ClaudeModel

SCENARIOS = ["All on Opus 5", "Routed"]
BOTTLE_ML = 500


@dataclass
class WaterAssumptions:
    wh_per_1k_output: dict[str, float] = field(default_factory=lambda: {
        "claude-opus-5": 2.0, "claude-sonnet-5": 0.8, "claude-haiku-4-5": 0.4,
    })
    """IT energy (Wh) per 1,000 generated tokens (output and thinking). Assumed to scale
    roughly with model size; not published by Anthropic."""

    input_energy_ratio: float = 0.05
    """Energy of one input token relative to one output token. Reading the prompt
    (prefill) is processed in parallel and costs far less per token than generating."""

    pue: float = 1.2
    """Power usage effectiveness: facility energy / IT energy."""

    wue_onsite: float = 0.5
    """On-site water usage effectiveness, litres per kWh of IT energy (cooling)."""

    ewif_offsite: float = 3.1
    """Water consumed generating electricity, litres per kWh (roughly the US grid average)."""


def request_footprint(model: ClaudeModel, effort: str | None, input_tokens: float,
                      output_tokens: float, a: te.Assumptions, w: WaterAssumptions) -> dict:
    """Energy and water for one request. Thinking tokens count as generated tokens."""
    b = te.cost_breakdown(model, effort, input_tokens, output_tokens, a)
    generated = b["output_tokens"] + b["thinking_tokens"]
    wh_out = w.wh_per_1k_output[model.model_id] / 1000
    it_wh = generated * wh_out + b["input_tokens"] * wh_out * w.input_energy_ratio
    onsite_ml = it_wh * w.wue_onsite                 # Wh x L/kWh = mL
    offsite_ml = it_wh * w.pue * w.ewif_offsite
    return {
        "input_tokens": b["input_tokens"],
        "generated_tokens": generated,
        "it_wh": it_wh,
        "facility_wh": it_wh * w.pue,
        "onsite_ml": onsite_ml,
        "offsite_ml": offsite_ml,
        "total_ml": onsite_ml + offsite_ml,
    }


def compute(mix: pd.DataFrame, monthly_requests: int, a: te.Assumptions,
            w: WaterAssumptions) -> pd.DataFrame:
    """One row per (scenario, task type) with the Claude model used and monthly water."""
    total_share = mix["Share %"].sum()
    rows = []
    for _, r in mix.iterrows():
        share = r["Share %"] / total_share if total_share else 0.0
        requests = monthly_requests * share
        rec = CLAUDE_MATRIX[(r["Category"], r["Complexity"])]
        setups = {"All on Opus 5": (OPUS, te.BASELINE_EFFORT), "Routed": (rec.model, rec.effort)}
        for scenario, (model, effort) in setups.items():
            f = request_footprint(model, effort, r["Avg input tokens"], r["Avg output tokens"], a, w)
            rows.append({
                "Scenario": scenario,
                "Category": r["Category"],
                "Complexity": r["Complexity"],
                "Claude model": model.model_id,
                "Effort": effort or "—",
                "Requests / month": requests,
                "mL / request": f["total_ml"],
                "Energy kWh / month": f["facility_wh"] * requests / 1000,
                "On-site L / month": f["onsite_ml"] * requests / 1000,
                "Off-site L / month": f["offsite_ml"] * requests / 1000,
                "Water L / month": f["total_ml"] * requests / 1000,
            })
    return pd.DataFrame(rows)
