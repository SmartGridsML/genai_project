import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from backend.app.api.routes.health import router as health_router
from backend.app.api.routes.applications import (
    router as applications_router,
    start_generation_worker,
    stop_generation_worker,
)
from backend.app.utils.request_id import RequestIDMiddleware
from backend.app.utils.rate_limiter import limiter
from backend.app.api.routes.downloads import router as downloads_router
from backend.app.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _check_redis() -> str:
    try:
        import redis as _redis
        r = _redis.Redis.from_url(get_settings().redis_url, socket_connect_timeout=2)
        r.ping()
        return "ok"
    except Exception as exc:
        return f"UNAVAILABLE ({exc})"


def _check_secret(settings) -> None:
    if settings.result_token_secret.get_secret_value() == "dev-secret-change-in-production":
        if settings.environment == "production":
            raise RuntimeError(
                "RESULT_TOKEN_SECRET must be set to a secure random value in production. "
                "Generate one with: openssl rand -hex 32"
            )
        logger.warning(
            "SECURITY: RESULT_TOKEN_SECRET is using the insecure default. "
            "Set RESULT_TOKEN_SECRET env var before deploying to production."
        )


def _check_llm() -> str:
    try:
        settings = get_settings()
        if not settings.gemini_api_key:
            return "UNAVAILABLE (GEMINI_API_KEY not set)"
        settings.gemini_api_key.get_secret_value()  # just ensure it's readable
        return "ok"
    except Exception as exc:
        return f"UNAVAILABLE ({exc})"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== CVForge backend starting ===")
    logger.info("Service: API              | status: initialising")

    redis_status = _check_redis()
    logger.info(f"Service: Redis            | status: {redis_status}")

    llm_status = _check_llm()
    logger.info(f"Service: LLM (Gemini)     | status: {llm_status}")

    logger.info("Service: Auditor          | status: ok (lazy-loaded)")
    logger.info("Service: CV Enhancer      | status: ok (lazy-loaded)")
    logger.info("Service: Document Parser  | status: ok")

    _check_secret(get_settings())
    await start_generation_worker()

    logger.info("=== CVForge backend ready ===")

    yield

    await stop_generation_worker()
    logger.info("=== CVForge backend shutting down ===")


def create_app() -> FastAPI:
    app = FastAPI(title="CV Application Helper API", version="0.1.0", lifespan=lifespan)

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(RequestIDMiddleware)

    allowed_origins = os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://localhost:5173"
    ).split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-Request-ID", "Authorization"],
    )

    app.include_router(health_router)
    app.include_router(applications_router, prefix="/v1")
    app.include_router(downloads_router, prefix="/v1")
    return app


app = create_app()
