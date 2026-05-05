"""Registered FastAPI exception handlers."""

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .exceptions import WhisperrrException
from .models import ErrorResponse

logger = logging.getLogger(__name__)

_WHISPERRR_STATUS_CODES: dict[str, int] = {
    "INVALID_AUDIO_FORMAT": 400,
    "FILE_TOO_LARGE": 413,
    "MODEL_NOT_LOADED": 503,
    "TRANSCRIPTION_FAILED": 500,
    "MODEL_LOAD_FAILED": 500,
    "AUDIO_PROCESSING_ERROR": 400,
    "FILE_SYSTEM_ERROR": 500,
}


def register_exception_handlers(app: FastAPI) -> None:

    @app.exception_handler(WhisperrrException)
    async def whisperrr_exception_handler(request: Request, exc: WhisperrrException):
        correlation_id = getattr(request.state, "correlation_id", None)
        error_response = ErrorResponse(
            error_type=exc.error_code or "WHISPERRR_ERROR",
            message=exc.message,
            details=exc.details,
            correlation_id=correlation_id,
        )
        status = _WHISPERRR_STATUS_CODES.get(exc.error_code, 500)
        return JSONResponse(status_code=status, content=error_response.model_dump())

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        correlation_id = getattr(request.state, "correlation_id", None)
        logger.error(
            "Unhandled exception: %s: %s",
            type(exc).__name__,
            str(exc),
            exc_info=True,
            extra={"correlation_id": correlation_id},
        )
        error_response = ErrorResponse(
            error_type="INTERNAL_SERVER_ERROR",
            message="An internal server error occurred",
            details=None,
            correlation_id=correlation_id,
        )
        return JSONResponse(status_code=500, content=error_response.model_dump())
