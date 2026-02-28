import base64
import hashlib
import json
from functools import lru_cache
from typing import Any, Optional

import redis.asyncio as redis
from cryptography.fernet import Fernet

from backend.app.config import get_settings


@lru_cache(maxsize=1)
def _get_pool() -> redis.ConnectionPool:
    # decode_responses=False so we can handle both plain strings and encrypted bytes
    return redis.ConnectionPool.from_url(
        get_settings().redis_url,
        max_connections=20,
        decode_responses=False,
    )


def _fernet(secret: str) -> Fernet:
    """Derive a Fernet key from the token secret using a purpose-specific prefix."""
    raw = hashlib.sha256(f"cache-encryption:{secret}".encode()).digest()
    return Fernet(base64.urlsafe_b64encode(raw))


class CacheService:
    def __init__(self):
        self.client = redis.Redis(connection_pool=_get_pool())

    async def get_json(self, key: str) -> Optional[Any]:
        val = await self.client.get(key)
        if val is None:
            return None
        return json.loads(val.decode() if isinstance(val, bytes) else val)

    async def set_json(self, key: str, value: Any, ttl_seconds: int) -> None:
        await self.client.set(key, json.dumps(value).encode(), ex=ttl_seconds)

    async def get_encrypted_json(self, key: str) -> Optional[Any]:
        val = await self.client.get(key)
        if val is None:
            return None
        secret = get_settings().result_token_secret.get_secret_value()
        plaintext = _fernet(secret).decrypt(val if isinstance(val, bytes) else val.encode())
        return json.loads(plaintext)

    async def set_encrypted_json(self, key: str, value: Any, ttl_seconds: int) -> None:
        secret = get_settings().result_token_secret.get_secret_value()
        ciphertext = _fernet(secret).encrypt(json.dumps(value).encode())
        await self.client.set(key, ciphertext, ex=ttl_seconds)
