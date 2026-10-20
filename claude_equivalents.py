"""
Claude recommendation layer.

The POC executes on Azure OpenAI, but for every (category, complexity) it also
reports the exact Claude model — and effort level — you would use for the same
task. No Anthropic API call is made; this is a lookup, like the routing matrix.

Effort (low | medium | high | xhigh | max) is Claude's quality/cost dial on
Opus 5 and Sonnet 5. Haiku 4.5 does not take an effort setting.
"""

from dataclasses import dataclass

from model_registry import Tier


@dataclass(frozen=True)
class ClaudeModel:
    model_id: str
    display_name: str
    input_cost_per_m: float     # Anthropic list price, $/1M tokens
    output_cost_per_m: float


OPUS = ClaudeModel("claude-opus-5", "Claude Opus 5", 5.0, 25.0)
SONNET = ClaudeModel("claude-sonnet-5", "Claude Sonnet 5", 2.0, 10.0)
HAIKU = ClaudeModel("claude-haiku-4-5", "Claude Haiku 4.5", 1.0, 5.0)

# Tier-level equivalents (used by the eval summary).
CLAUDE_BY_TIER: dict[Tier, ClaudeModel] = {
    Tier.FRONTIER: OPUS,
    Tier.WORKHORSE: SONNET,
    Tier.FAST: HAIKU,
}


@dataclass(frozen=True)
class ClaudeRecommendation:
    model: ClaudeModel
    effort: str | None
    why: str

    @property
    def label(self) -> str:
        effort = f" (effort={self.effort})" if self.effort else ""
        return f"{self.model.display_name} [{self.model.model_id}]{effort}"

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return round(
            prompt_tokens / 1_000_000 * self.model.input_cost_per_m
            + completion_tokens / 1_000_000 * self.model.output_cost_per_m,
            6,
        )


# Same tier as ROUTING_MATRIX in router.py; effort adds finer control within a tier.
CLAUDE_MATRIX: dict[tuple[str, str], ClaudeRecommendation] = {
    # coding
    ("coding", "low"): ClaudeRecommendation(HAIKU, None, "Snippets and boilerplate are well within Haiku's range."),
    ("coding", "medium"): ClaudeRecommendation(SONNET, "medium", "Real functions with tests; Sonnet is a strong everyday coder."),
    ("coding", "high"): ClaudeRecommendation(OPUS, "xhigh", "Architecture and agentic coding; xhigh is the recommended effort for hard coding work."),
    # reasoning
    ("reasoning", "low"): ClaudeRecommendation(SONNET, "low", "Simple logic; Sonnet at low effort stays cheap and fast."),
    ("reasoning", "medium"): ClaudeRecommendation(SONNET, "high", "Multi-step analysis benefits from more thinking, not a bigger model."),
    ("reasoning", "high"): ClaudeRecommendation(OPUS, "high", "Hard, high-stakes reasoning justifies Opus."),
    # writing
    ("writing", "low"): ClaudeRecommendation(HAIKU, None, "Short routine copy — reminders, notes."),
    ("writing", "medium"): ClaudeRecommendation(SONNET, "low", "Tone-sensitive prose; little reasoning needed."),
    ("writing", "high"): ClaudeRecommendation(SONNET, "medium", "Long-form writing; escalate to Opus only if the judge disagrees."),
    # extraction
    ("extraction", "low"): ClaudeRecommendation(HAIKU, None, "Single-label classification or field extraction."),
    ("extraction", "medium"): ClaudeRecommendation(HAIKU, None, "Structured extraction at volume."),
    ("extraction", "high"): ClaudeRecommendation(SONNET, "low", "Messy multi-entity extraction needs more judgment."),
    # general
    ("general", "low"): ClaudeRecommendation(HAIKU, None, "Simple Q&A and short summaries."),
    ("general", "medium"): ClaudeRecommendation(SONNET, "low", "Default middle ground."),
    ("general", "high"): ClaudeRecommendation(OPUS, "high", "Open-ended, high-stakes requests."),
}


def recommend_claude(category: str, complexity: str) -> ClaudeRecommendation:
    return CLAUDE_MATRIX.get(
        (category, complexity),
        ClaudeRecommendation(SONNET, "medium", "Unknown classification; defaulting to the workhorse."),
    )
