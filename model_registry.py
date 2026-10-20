"""
Model registry: the 'Tier' layer of the 3T framework.

Maps (tier, category) to a ModelProvider that can be from any vendor:
Azure OpenAI, OpenAI, Anthropic Claude, Google Gemini, etc.

Tier philosophy:
  FRONTIER  -> hardest, highest-stakes tasks (deep reasoning, complex code)
  WORKHORSE -> default tier for most business work
  FAST      -> high-volume, low-complexity tasks (classification, extraction)
"""

import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from provider import ModelProvider, AzureOpenAIProvider, OpenAIProvider, AnthropicProvider, GoogleProvider


class Tier(str, Enum):
    FRONTIER = "frontier"
    WORKHORSE = "workhorse"
    FAST = "fast"


@dataclass(frozen=True)
class ModelSpec:
    tier: Tier
    provider: ModelProvider
    input_cost_per_m: float      # Indicative $/1M tokens
    output_cost_per_m: float
    notes: str

    @property
    def deployment(self) -> str:
        """Backward compatibility: return provider's model_id as deployment name."""
        return self.provider.model_id

    @property
    def display_name(self) -> str:
        return self.provider.display_name


def load_registry() -> dict[Tier, ModelSpec]:
    """Build the registry from environment variables."""

    # Determine which providers to use via .env
    providers_config = {
        "azure": os.getenv("USE_AZURE_OPENAI", "true").lower() == "true",
        "openai": os.getenv("USE_OPENAI", "false").lower() == "true",
        "anthropic": os.getenv("USE_ANTHROPIC", "false").lower() == "true",
        "google": os.getenv("USE_GOOGLE", "false").lower() == "true",
    }

    registry = {}

    # FRONTIER tier
    if providers_config["azure"]:
        provider = AzureOpenAIProvider(
            deployment_name=os.getenv("AZURE_DEPLOYMENT_FRONTIER", "gpt-4.1-mini"),
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
        )
    elif providers_config["openai"]:
        provider = OpenAIProvider(
            model_name=os.getenv("OPENAI_MODEL_FRONTIER", "gpt-4"),
            api_key=os.getenv("OPENAI_API_KEY", ""),
        )
    elif providers_config["anthropic"]:
        provider = AnthropicProvider(
            model_name=os.getenv("ANTHROPIC_MODEL_FRONTIER", "claude-opus-5"),
            api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        )
    else:
        provider = GoogleProvider(
            model_name=os.getenv("GOOGLE_MODEL_FRONTIER", "gemini-pro"),
            api_key=os.getenv("GOOGLE_API_KEY", ""),
        )

    registry[Tier.FRONTIER] = ModelSpec(
        tier=Tier.FRONTIER,
        provider=provider,
        input_cost_per_m=float(os.getenv("COST_IN_FRONTIER", "0.4")),
        output_cost_per_m=float(os.getenv("COST_OUT_FRONTIER", "1.6")),
        notes="Deep reasoning, complex multi-step code, high stakes",
    )

    # WORKHORSE tier
    if providers_config["azure"]:
        provider = AzureOpenAIProvider(
            deployment_name=os.getenv("AZURE_DEPLOYMENT_WORKHORSE", "gpt-4.1-mini"),
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
        )
    elif providers_config["openai"]:
        provider = OpenAIProvider(
            model_name=os.getenv("OPENAI_MODEL_WORKHORSE", "gpt-4-turbo"),
            api_key=os.getenv("OPENAI_API_KEY", ""),
        )
    elif providers_config["anthropic"]:
        provider = AnthropicProvider(
            model_name=os.getenv("ANTHROPIC_MODEL_WORKHORSE", "claude-sonnet-5"),
            api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        )
    else:
        provider = GoogleProvider(
            model_name=os.getenv("GOOGLE_MODEL_WORKHORSE", "gemini-1.5-pro"),
            api_key=os.getenv("GOOGLE_API_KEY", ""),
        )

    registry[Tier.WORKHORSE] = ModelSpec(
        tier=Tier.WORKHORSE,
        provider=provider,
        input_cost_per_m=float(os.getenv("COST_IN_WORKHORSE", "0.4")),
        output_cost_per_m=float(os.getenv("COST_OUT_WORKHORSE", "1.6")),
        notes="Default for drafting, analysis, standard coding",
    )

    # FAST tier
    if providers_config["azure"]:
        provider = AzureOpenAIProvider(
            deployment_name=os.getenv("AZURE_DEPLOYMENT_FAST", "gpt-4.1-mini"),
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
        )
    elif providers_config["openai"]:
        provider = OpenAIProvider(
            model_name=os.getenv("OPENAI_MODEL_FAST", "gpt-4-mini"),
            api_key=os.getenv("OPENAI_API_KEY", ""),
        )
    elif providers_config["anthropic"]:
        provider = AnthropicProvider(
            model_name=os.getenv("ANTHROPIC_MODEL_FAST", "claude-haiku-4-5"),
            api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        )
    else:
        provider = GoogleProvider(
            model_name=os.getenv("GOOGLE_MODEL_FAST", "gemini-1.5-flash"),
            api_key=os.getenv("GOOGLE_API_KEY", ""),
        )

    registry[Tier.FAST] = ModelSpec(
        tier=Tier.FAST,
        provider=provider,
        input_cost_per_m=float(os.getenv("COST_IN_FAST", "0.4")),
        output_cost_per_m=float(os.getenv("COST_OUT_FAST", "1.6")),
        notes="Classification, extraction, summarisation at volume",
    )

    return registry
