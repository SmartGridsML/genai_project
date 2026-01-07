"""
Store env variables and other config settings.
"""
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, SecretStr

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore" # Ignore extra env vars to prevent crashes
    )

    # Infrastructure
    redis_url: str = "redis://localhost:6379"
    database_url: str = "postgresql://localhost:5432/postgres"
    
    # LLM Configuration
    openai_api_key: SecretStr = Field(..., description="Required for LLM calls")
    openai_model: str = "gpt-4o"
    max_retries: int = 3
    timeout_seconds: int = 30
    
    # MLflow
    mlflow_tracking_uri: str = "file:./mlruns"
    experiment_name: str = "cv_helper_v1"

settings = Settings()
