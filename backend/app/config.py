import os
from pydantic import BaseModel


class Settings(BaseModel):
    redis_url: str = os.getenv("REDIS_URL", "redis://redis:6379/0")
    llm_base_url: str | None = os.getenv("LLM_BASE_URL")  # Person A service URL
    cache_ttl_seconds: int = int(os.getenv("CACHE_TTL_SECONDS", str(24 * 3600)))


settings = Settings()
