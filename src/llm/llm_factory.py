from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Type

import instructor
from anthropic import Anthropic
from openai import AzureOpenAI, OpenAI
from pydantic import BaseModel

from config.settings import get_settings

"""
LLM Provider Factory Module

This module implements a factory pattern for creating and managing different LLM providers
(OpenAI, Anthropic, etc.). It provides a unified interface for LLM interactions while
supporting structured output using Pydantic models.
"""


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def _initialize_client(self) -> Any:
        """Initialize the client for the LLM provider."""
        pass

    @abstractmethod
    def create_completion(
        self, response_model: Type[BaseModel], messages: List[Dict[str, str]], **kwargs
    ) -> Any:
        """Create a completion using the LLM provider."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI provider implementation."""

    def __init__(self, settings):
        self.settings = settings
        self.client = self._initialize_client()

    def _initialize_client(self) -> Any:
        return instructor.from_openai(OpenAI(api_key=self.settings.api_key))

    def create_completion(
        self, response_model: Type[BaseModel], messages: List[Dict[str, str]], **kwargs
    ) -> Tuple[BaseModel, Any]:
        completion_params = {
            "model": kwargs.get("model", self.settings.default_model),
            "temperature": kwargs.get("temperature", self.settings.temperature),
            "max_retries": kwargs.get("max_retries", self.settings.max_retries),
            "max_tokens": kwargs.get("max_tokens", self.settings.max_tokens),
            "response_model": response_model,
            "messages": messages,
        }
        return self.client.chat.completions.create_with_completion(**completion_params)


class AzureOpenAIProvider(LLMProvider):
    """AzureOpenAI provider implementation."""

    def __init__(self, settings):
        self.settings = settings
        self.client = self._initialize_client()

    def _initialize_client(self) -> Any:
        return instructor.from_openai(
            AzureOpenAI(
                api_key=self.settings.api_key,
                api_version=self.settings.api_version,
                azure_endpoint=self.settings.azure_endpoint,
            )
        )

    def create_completion(
        self, response_model: Type[BaseModel], messages: List[Dict[str, str]], **kwargs
    ) -> Tuple[BaseModel, Any]:
        completion_params = {
            "model": kwargs.get("model", self.settings.default_model),
            "temperature": kwargs.get("temperature", self.settings.temperature),
            "max_retries": kwargs.get("max_retries", self.settings.max_retries),
            "max_tokens": kwargs.get("max_tokens", self.settings.max_tokens),
            "response_model": response_model,
            "messages": messages,
        }
        return self.client.chat.completions.create_with_completion(**completion_params)


class AnthropicProvider(LLMProvider):
    """Anthropic provider implementation."""

    def __init__(self, settings):
        self.settings = settings
        self.client = self._initialize_client()

    def _initialize_client(self) -> Any:
        return instructor.from_anthropic(Anthropic(api_key=self.settings.api_key))

    def create_completion(
        self, response_model: Type[BaseModel], messages: List[Dict[str, str]], **kwargs
    ) -> Any:
        system_message = next(
            (m["content"] for m in messages if m["role"] == "system"), None
        )
        user_messages = [m for m in messages if m["role"] != "system"]

        completion_params = {
            "model": kwargs.get("model", self.settings.default_model),
            "temperature": kwargs.get("temperature", self.settings.temperature),
            "max_retries": kwargs.get("max_retries", self.settings.max_retries),
            "max_tokens": kwargs.get("max_tokens", self.settings.max_tokens),
            "response_model": response_model,
            "messages": user_messages,
        }
        if system_message:
            completion_params["system"] = system_message

        return self.client.messages.create_with_completion(**completion_params)


class LlamaProvider(LLMProvider):
    """Llama provider implementation."""

    def __init__(self, settings):
        self.settings = settings
        self.client = self._initialize_client()

    def _initialize_client(self) -> Any:
        return instructor.from_openai(
            OpenAI(base_url=self.settings.base_url, api_key=self.settings.api_key),
            mode=instructor.Mode.JSON,
        )

    def create_completion(
        self, response_model: Type[BaseModel], messages: List[Dict[str, str]], **kwargs
    ) -> Any:
        completion_params = {
            "model": kwargs.get("model", self.settings.default_model),
            "temperature": kwargs.get("temperature", self.settings.temperature),
            "max_retries": kwargs.get("max_retries", self.settings.max_retries),
            "max_tokens": kwargs.get("max_tokens", self.settings.max_tokens),
            "response_model": response_model,
            "messages": messages,
        }
        return self.client.chat.completions.create_with_completion(**completion_params)


class LLMFactory:
    """
    Factory class for creating and managing LLM provider instances.

    This class implements the Factory pattern to create appropriate LLM provider
    instances based on the specified provider type. It supports multiple providers
    and handles their initialization and configuration.

    Attributes:
        provider: The name of the LLM provider to use
        settings: Configuration settings for the LLM provider
        llm_provider: The initialized LLM provider instance
    """

    def __init__(self, provider: str):
        self.provider = provider
        settings = get_settings()
        self.settings = getattr(settings.llm, provider)
        self.llm_provider = self._create_provider()

    def _create_provider(self) -> LLMProvider:
        providers = {
            "openai": OpenAIProvider,
            "anthropic": AnthropicProvider,
            "llama": LlamaProvider,
            "azureopenai": AzureOpenAIProvider,
        }
        provider_class = providers.get(self.provider)
        if provider_class:
            return provider_class(self.settings)
        raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def create_completion(
        self, response_model: Type[BaseModel], messages: List[Dict[str, str]], **kwargs
    ) -> Tuple[BaseModel, Any]:
        """
        Create a completion using the configured LLM provider.

        Args:
            response_model: Pydantic model class defining the expected response structure
            messages: List of message dictionaries containing the conversation
            **kwargs: Additional arguments to pass to the provider

        Returns:
            Tuple containing the parsed response model and raw completion

        Raises:
            TypeError: If response_model is not a Pydantic BaseModel
            ValueError: If the provider is not supported
        """
        if not issubclass(response_model, BaseModel):
            raise TypeError("response_model must be a subclass of pydantic.BaseModel")

        return self.llm_provider.create_completion(response_model, messages, **kwargs)


# Example usage of the LLMFactory

if __name__ == "__main__":

    class ExampleResponseModel(BaseModel):
        """Example response model for structured output."""

        message: str

    # Initialize the factory with the desired provider
    provider_name = "llama"  # Change to "anthropic" or "llama" as needed
    factory = LLMFactory(provider=provider_name)

    # Define the input messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]

    # Create a completion
    try:
        response_model, raw_response = factory.create_completion(
            response_model=ExampleResponseModel,
            messages=messages,
            model="phi4:latest",
            temperature=0.7,
            max_tokens=100,
        )
        print("Parsed Response:", response_model)
        print("Raw Response:", raw_response)
    except Exception as e:
        print("Error:", e)
