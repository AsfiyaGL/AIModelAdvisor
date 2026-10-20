"""Shared Azure OpenAI client factory."""

import os
from openai import AzureOpenAI


def get_client() -> AzureOpenAI:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")

    if not endpoint or not api_key:
        raise EnvironmentError(
            "Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY "
            "(see .env.example)."
        )

    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )
