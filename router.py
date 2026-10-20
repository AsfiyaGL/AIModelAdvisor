"""
Step 2 — TIER: map (category, complexity) to a model tier.

The routing matrix encodes the golden rule of the framework:
start one tier lower than you think you need, escalate only on failure.

This is deliberately a plain dict, not an LLM call — the policy should be
readable, auditable, and editable by a governance team.
"""

from model_registry import Tier, ModelSpec, load_registry


ROUTING_MATRIX: dict[tuple[str, str], Tier] = {
    # coding
    ("coding", "low"): Tier.FAST,
    ("coding", "medium"): Tier.WORKHORSE,
    ("coding", "high"): Tier.FRONTIER,
    # reasoning
    ("reasoning", "low"): Tier.WORKHORSE,
    ("reasoning", "medium"): Tier.WORKHORSE,
    ("reasoning", "high"): Tier.FRONTIER,
    # writing
    ("writing", "low"): Tier.FAST,
    ("writing", "medium"): Tier.WORKHORSE,
    ("writing", "high"): Tier.WORKHORSE,
    # extraction
    ("extraction", "low"): Tier.FAST,
    ("extraction", "medium"): Tier.FAST,
    ("extraction", "high"): Tier.WORKHORSE,
    # general
    ("general", "low"): Tier.FAST,
    ("general", "medium"): Tier.WORKHORSE,
    ("general", "high"): Tier.FRONTIER,
}


def route(category: str, complexity: str) -> ModelSpec:
    registry = load_registry()
    tier = ROUTING_MATRIX.get((category, complexity), Tier.WORKHORSE)
    return registry[tier]


def execute(user_prompt: str, spec: ModelSpec) -> dict:
    """Run the prompt on the routed provider; return answer + usage."""
    result = spec.provider.create_completion(
        messages=[{"role": "user", "content": user_prompt}],
        temperature=0.4,
        max_tokens=800,
    )

    cost = (
        result["prompt_tokens"] / 1_000_000 * spec.input_cost_per_m
        + result["completion_tokens"] / 1_000_000 * spec.output_cost_per_m
    )
    return {
        "answer": result["answer"],
        "tier": spec.tier.value,
        "deployment": spec.deployment,
        "provider": spec.provider.provider_type.value,
        "prompt_tokens": result["prompt_tokens"],
        "completion_tokens": result["completion_tokens"],
        "estimated_cost_usd": round(cost, 6),
    }
