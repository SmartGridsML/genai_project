from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes.health import router as health_router
from backend.app.api.routes.applications import router as applications_router
from backend.app.utils.request_id import RequestIDMiddleware
from backend.app.api.routes.llm import router as llm_router
from backend.app.api.routes.downloads import router as downloads_router


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
    app.include_router(llm_router)
    app.include_router(downloads_router)
    return app


app = create_app()
