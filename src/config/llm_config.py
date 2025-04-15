import os
from typing import Optional

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

"""
Configuration for LLM providers.
"""


class LLMProviderSettings(BaseSettings):
    """Base settings for LLM providers."""

    # Aiming for controlled creativity
    temperature: float = 1.0
    top_p: float = 0.7
    max_tokens: Optional[int] = None
    max_retries: int = 3


class OpenAISettings(LLMProviderSettings):
    """Settings for OpenAI."""

    api_key: str = os.getenv("OPENAI_API_KEY")
    default_model: str = "gpt-4o-mini-2024-07-18"


class AzureOpenAISettings(LLMProviderSettings):
    """Settings for AzureOpenAI."""

    api_key: str = os.getenv("AZURE_OPENAI_API_KEY")
    default_model: str = "gpt-4o-mini"
    api_version: str = "2024-02-01"
    azure_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT")


class AnthropicSettings(LLMProviderSettings):
    """Settings for Anthropic."""

    api_key: str = os.getenv("ANTHROPIC_API_KEY")
    default_model: str = "claude-3-5-sonnet-20240620"
    max_tokens: int = 4096


class OllamaSettings(LLMProviderSettings):
    """Settings for Llama."""

    api_key: str = "key"  # required, but not used
    default_model: str = "phi4"
    base_url: str = "http://localhost:11434/v1"


class LLMConfig(BaseSettings):
    """Configuration for all LLM providers."""

    openai: OpenAISettings = OpenAISettings()
    azureopenai: AzureOpenAISettings = AzureOpenAISettings()
    anthropic: AnthropicSettings = AnthropicSettings()
    ollama: OllamaSettings = OllamaSettings()
