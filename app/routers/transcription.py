import logging
import os
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile

from ..api.dependencies import RequireWhisperModel, validate_and_materialize_upload
from ..models import TranscriptionResponse
from ..whisper_service import whisper_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["transcription"])


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    request: Request,
    _: RequireWhisperModel,
    file: UploadFile = File(...),
    model_size: Optional[str] = Query(None, description="Whisper model size"),
    language: Optional[str] = Query(
        None, max_length=16, description="ISO 639-1 language hint (optional)"
    ),
    task: Optional[str] = Query(
        None, description="Whisper task: transcribe or translate (optional)"
    ),
    temperature: float = Query(0.0, ge=0.0, le=1.0, description="Temperature for sampling"),
):
    """Transcribe audio file using Whisper model."""
    correlation_id = getattr(request.state, "correlation_id", None)
    temp_file_path = None

    try:
        temp_file_path = await validate_and_materialize_upload(file)

        result = await whisper_service.transcribe_audio(
            file_path=temp_file_path,
            model_size=model_size,
            language=language,
            temperature=temperature,
            task=task,
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        file_name = file.filename if file else None
        logger.error(
            f"Error processing transcription request: {type(e).__name__}: {str(e)}",
            exc_info=True,
            extra={"correlation_id": correlation_id, "file_name": file_name},
        )
        raise
    finally:
        if temp_file_path:
            try:
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
            except Exception:
                pass
