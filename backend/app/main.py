from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.health import router as health_router
from app.api.routes.applications import router as applications_router
from app.utils.request_id import RequestIDMiddleware


def create_app() -> FastAPI:
    app = FastAPI(title="CV Application Helper API", version="0.1.0")

    app.add_middleware(RequestIDMiddleware)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(applications_router)
    return app


app = create_app()
