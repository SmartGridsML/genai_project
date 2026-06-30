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

    # Deployment environment — set to "production" in prod to enforce security checks
    environment: str = Field(default="development")

    # Proxy/header trust for rate limiting
    trust_proxy_headers: bool = Field(
        default=False,
        description="Whether to trust X-Forwarded-For for client IP extraction.",
    )
    trusted_proxy_ips: str = Field(
        default="",
        description="Comma-separated proxy IPs allowed to forward client IP headers.",
    )

    # LLM Configuration
    # NOTE: keep it optional for import-time, enforce at call-time.
    openai_api_key: SecretStr | None = Field(default=None, description="Primary LLM provider")
    gemini_api_key: SecretStr | None = Field(default=None, description="Fallback LLM provider")

    openai_model: str = "gpt-4o"
    gemini_model: str = "gemini-2.5-flash"
    max_retries: int = 3
    timeout_seconds: int = 30

    # MLflow — disabled by default; enable explicitly via MLFLOW_ENABLED=true
    mlflow_tracking_uri: str = "file:./mlruns"
    experiment_name: str = "cv_helper_v1"
    mlflow_enabled: bool = False

    # Access token secret — MUST be set in production via RESULT_TOKEN_SECRET env var
    result_token_secret: SecretStr = Field(
        default="dev-secret-change-in-production",
        description="HMAC secret for signing result access tokens",
    )

@lru_cache
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
