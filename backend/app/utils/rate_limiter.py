from __future__ import annotations

import os
from typing import Tuple

from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from backend.app.config import settings
from backend.app.services.cache_service import CacheService


def _check_limit(client, key: str, limit: int, window_seconds: int) -> Tuple[bool, int]:
    pipe = client.pipeline()
    pipe.incr(key)
    pipe.ttl(key)
    count, ttl = pipe.execute()

    if ttl < 0:
        client.expire(key, window_seconds)
        ttl = window_seconds

    allowed = count <= limit
    retry_after = ttl if ttl > 0 else window_seconds
    return allowed, retry_after


async def rate_limit_middleware(request: Request, call_next) -> Response:
    if not settings.rate_limit_enabled or os.getenv("PYTEST_CURRENT_TEST"):
        return await call_next(request)

    path = request.url.path
    if path in {"/health", "/metrics"}:
        return await call_next(request)

    cache = CacheService(settings.redis_url)
    client = cache.client

    ip_address = request.client.host if request.client else "unknown"
    ip_key = f"ratelimit:ip:{ip_address}"
    allowed, retry_after = _check_limit(
        client,
        ip_key,
        settings.rate_limit_ip_per_hour,
        settings.rate_limit_window_seconds,
    )

    if not allowed:
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded"},
            headers={"Retry-After": str(retry_after)},
        )

    user_id = request.headers.get("X-User-Id")
    if user_id:
        user_key = f"ratelimit:user:{user_id}"
        allowed, retry_after = _check_limit(
            client,
            user_key,
            settings.rate_limit_user_per_hour,
            settings.rate_limit_window_seconds,
        )
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"detail": "User rate limit exceeded"},
                headers={"Retry-After": str(retry_after)},
            )

    return await call_next(request)
