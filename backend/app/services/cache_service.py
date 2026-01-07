import json
from typing import Any, Optional
import redis

class CacheService:
    def __init__(self, redis_url: str):
        self.client = redis.Redis.from_url(redis_url, decode_responses=True)

    def get_json(self, key: str) -> Optional[Any]:
        val = self.client.get(key)
        if val is None:
            return None
        return json.loads(val)

    def set_json(self, key: str, value: Any, ttl_seconds: int) -> None:
        self.client.set(key, json.dumps(value), ex=ttl_seconds)
