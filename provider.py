"""
Provider abstraction layer for multi-vendor model support.

Supports: Azure OpenAI, OpenAI, Anthropic Claude, Google Gemini, etc.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ProviderType(str, Enum):
    AZURE_OPENAI = "azure_openai"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


@dataclass
class ModelProvider(ABC):
    """Base class for LLM providers."""

    provider_type: ProviderType
    model_id: str
    display_name: str

    @abstractmethod
    def create_completion(self, messages: list[dict], temperature: float = 0.4, max_tokens: int = 800) -> dict:
        """
        Call the model and return response.

        Returns: {
            "answer": str,
            "prompt_tokens": int,
            "completion_tokens": int,
            "cost_usd": float
        }
        """
        pass


class AzureOpenAIProvider(ModelProvider):
    """Azure OpenAI deployment."""

    def __init__(self, deployment_name: str, endpoint: str, api_key: str, api_version: str = "2024-12-01-preview"):
        super().__init__(
            provider_type=ProviderType.AZURE_OPENAI,
            model_id=deployment_name,
            display_name=f"Azure OpenAI: {deployment_name}"
        )
        self.deployment_name = deployment_name
        self.endpoint = endpoint
        self.api_key = api_key
        self.api_version = api_version

    def create_completion(self, messages: list[dict], temperature: float = 0.4, max_tokens: int = 800) -> dict:
        from openai import AzureOpenAI

        client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=self.api_version,
        )
        response = client.chat.completions.create(
            model=self.deployment_name,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=messages,
        )
        return {
            "answer": response.choices[0].message.content,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        }


class OpenAIProvider(ModelProvider):
    """OpenAI API (gpt-4, gpt-4-turbo, etc.)."""

    def __init__(self, model_name: str, api_key: str):
        super().__init__(
            provider_type=ProviderType.OPENAI,
            model_id=model_name,
            display_name=f"OpenAI: {model_name}"
        )
        self.model_name = model_name
        self.api_key = api_key

    def create_completion(self, messages: list[dict], temperature: float = 0.4, max_tokens: int = 800) -> dict:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=messages,
        )
        return {
            "answer": response.choices[0].message.content,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        }


class AnthropicProvider(ModelProvider):
    """Anthropic Claude API."""

    # Current-generation models: they reject temperature/top_p/top_k, and Opus 5 /
    # Sonnet 5 think adaptively by default. Effort is the quality/cost dial instead.
    EFFORT_BY_MODEL = {
        "claude-opus-5": "high",
        "claude-sonnet-5": "medium",
    }
    # Thinking tokens count toward max_tokens, so small caps (e.g. the judge's 120)
    # would cut the answer off. This is a ceiling, not a spend.
    MIN_MAX_TOKENS_WITH_THINKING = 16000
    # Opus 5's safety classifiers can decline a request; "default" fallbacks re-run
    # it server-side on Anthropic's recommended model instead of returning a refusal.
    FALLBACK_BETA = "server-side-fallback-2026-07-01"

    def __init__(self, model_name: str, api_key: str):
        super().__init__(
            provider_type=ProviderType.ANTHROPIC,
            model_id=model_name,
            display_name=f"Anthropic: {model_name}"
        )
        self.model_name = model_name
        self.api_key = api_key

    def create_completion(self, messages: list[dict], temperature: float = 0.4, max_tokens: int = 800) -> dict:
        import anthropic

        # Empty key -> let the SDK resolve ANTHROPIC_API_KEY / `ant auth login` profile.
        client = anthropic.Anthropic(api_key=self.api_key or None)

        # The Messages API takes the system prompt as a top-level field, not a message.
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        kwargs = {
            "model": self.model_name,
            "messages": [m for m in messages if m["role"] != "system"],
        }
        if system:
            kwargs["system"] = system

        effort = self.EFFORT_BY_MODEL.get(self.model_name)
        if effort:
            kwargs["max_tokens"] = max(max_tokens, self.MIN_MAX_TOKENS_WITH_THINKING)
            kwargs["output_config"] = {"effort": effort}
        else:
            kwargs["max_tokens"] = max_tokens
            kwargs["temperature"] = temperature

        if self.model_name == "claude-opus-5":
            response = client.beta.messages.create(
                betas=[self.FALLBACK_BETA], fallbacks="default", **kwargs
            )
        else:
            response = client.messages.create(**kwargs)

        if response.stop_reason == "refusal":
            category = response.stop_details.category if response.stop_details else None
            answer = f"[refused - category: {category}]"
        else:
            # Responses can include thinking blocks; keep only the text.
            answer = "".join(b.text for b in response.content if b.type == "text")

        return {
            "answer": answer,
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
        }


class GoogleProvider(ModelProvider):
    """Google Gemini API."""

    def __init__(self, model_name: str, api_key: str):
        super().__init__(
            provider_type=ProviderType.GOOGLE,
            model_id=model_name,
            display_name=f"Google: {model_name}"
        )
        self.model_name = model_name
        self.api_key = api_key

    def create_completion(self, messages: list[dict], temperature: float = 0.4, max_tokens: int = 800) -> dict:
        import google.generativeai as genai

        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model_name)

        # Convert messages to Gemini format
        content = [{"role": msg["role"], "parts": [{"text": msg["content"]}]} for msg in messages]

        response = model.generate_content(
            content,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        return {
            "answer": response.text,
            "prompt_tokens": 0,  # Gemini doesn't expose token counts in free API
            "completion_tokens": 0,
        }
