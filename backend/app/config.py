"""
Store env variables and other config settings.
"""
from __future__ import annotations

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, SecretStr


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Ignore extra env vars to prevent crashes
    )

    # Infrastructure
    redis_url: str = "redis://localhost:6379"
    database_url: str = "postgresql://localhost:5432/postgres"
    cache_ttl_seconds: int = int(24 * 3600)
    llm_base_url: str = "http://localhost:8000"

    # LLM Configuration
    # NOTE: keep it optional for import-time, enforce at call-time.
    openai_api_key: SecretStr | None = Field(default=None, description="Primary LLM provider")
    gemini_api_key: SecretStr | None = Field(default=None, description="Fallback LLM provider")

    openai_model: str = "gpt-4o"
    gemini_model: str = "gemini-2.5-flash"
    max_retries: int = 3
    timeout_seconds: int = 30

    # MLflow
    mlflow_tracking_uri: str = "file:./mlruns"
    experiment_name: str = "cv_helper_v1"

@lru_cache
def get_settings() -> Settings:
    return Settings()

settings = get_settings()