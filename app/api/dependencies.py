"""Shared FastAPI dependencies and upload helpers for transcription endpoints."""

import os
import tempfile
from typing import Annotated

from fastapi import Depends, UploadFile, HTTPException

from ..config import settings
from ..utils import safe_filename
from ..whisper_service import whisper_service


async def require_whisper_model_loaded() -> None:
    """Dependency: ensure Whisper is ready before transcription routes run."""
    if not whisper_service.is_model_loaded():
        raise HTTPException(
            status_code=503,
            detail="Transcription model is not loaded. Please wait for the service to initialize.",
        )


RequireWhisperModel = Annotated[None, Depends(require_whisper_model_loaded)]


async def validate_and_materialize_upload(file: UploadFile) -> str:
    """
    Validate multipart upload constraints and persist body to a temp file path.

    The returned path uses a readable suffix derived from the sanitized filename.
    The caller owns the path and must remove it after use (success or failure).
    """
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    safe_name = safe_filename(file.filename)

    if file.size and file.size > settings.max_file_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {settings.max_file_size_mb}MB",
        )

    content = await file.read()

    ext = safe_name.split(".")[-1] if "." in safe_name else "bin"
    fd, temp_file_path = tempfile.mkstemp(suffix=f".{ext}")
    try:
        os.write(fd, content)
    finally:
        os.close(fd)

    return temp_file_path
