from functools import lru_cache

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

from config.llm_config import LLMConfig

load_dotenv()

"""
Main settings for the application using Pydantic Settings.
"""


class Settings(BaseSettings):
    """Main settings for the application."""

    llm: LLMConfig = LLMConfig()


@lru_cache
def get_settings() -> Settings:
    """
    Get the application settings.

    Returns:
        Settings: The application settings.
    """
    return Settings()
