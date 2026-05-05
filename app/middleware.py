"""HTTP middleware (CORS, correlation ID)."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .utils import get_correlation_id

_loopback_origin_regex = r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$"


def install_cors_middleware(app: FastAPI) -> None:
    cors_kwargs: dict[str, object] = {
        "allow_origins": settings.cors_origins,
        "allow_credentials": True,
        "allow_methods": ["*"],
        "allow_headers": ["*"],
        "expose_headers": ["X-Correlation-ID"],
    }
    if settings.cors_allow_loopback_regex:
        cors_kwargs["allow_origin_regex"] = _loopback_origin_regex
    app.add_middleware(CORSMiddleware, **cors_kwargs)


def install_correlation_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def correlation_middleware(request, call_next):
        correlation_id = get_correlation_id()
        request.state.correlation_id = correlation_id
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = correlation_id
        return response
