"""Application factory."""

from fastapi import FastAPI

from .config import settings
from .exception_handlers import register_exception_handlers
from .lifespan import lifespan
from .logging_setup import configure_logging
from .middleware import install_cors_middleware, install_correlation_middleware
from .routers import health, jobs, transcription


def create_app() -> FastAPI:
    configure_logging()

    application = FastAPI(
        title=settings.api_title,
        description=settings.api_description,
        version=settings.api_version,
        lifespan=lifespan,
    )

    install_cors_middleware(application)
    install_correlation_middleware(application)
    register_exception_handlers(application)

    application.include_router(transcription.router)
    application.include_router(health.router)
    application.include_router(jobs.router)

    return application
