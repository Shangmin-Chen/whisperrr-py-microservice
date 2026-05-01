import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api.dependencies import RequireWhisperModel, validate_and_materialize_upload
from .application.transcription_jobs import process_transcription_job
from .config import settings
from .exceptions import WhisperrrException
from .job_manager import JobStatus, job_manager
from .models import (
    TranscriptionResponse,
    HealthResponse,
    ErrorResponse,
    JobSubmissionResponse,
    JobProgressResponse,
)
from .utils import get_correlation_id
from .whisper_service import whisper_service

# Configure logging before creating module logger
root_logger = logging.getLogger()
root_logger.setLevel(getattr(logging, settings.log_level, logging.INFO))
if not root_logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, settings.log_level, logging.INFO))
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
else:
    for h in root_logger.handlers:
        h.setLevel(getattr(logging, settings.log_level, logging.INFO))

logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, settings.log_level, logging.INFO))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan: startup and shutdown."""
    # Startup
    cleanup_task = None
    try:
        # Load Whisper model
        await whisper_service.load_model(settings.model_size)
        
        # Start background job cleanup task
        async def periodic_cleanup():
            """Periodically cleanup old jobs."""
            while True:
                try:
                    await asyncio.sleep(settings.job_cleanup_interval_seconds)
                    job_manager.cleanup_old_jobs()
                    logger.info("Completed periodic job cleanup")
                except asyncio.CancelledError:
                    logger.info("Job cleanup task cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error in job cleanup: {e}", exc_info=True)
        
        cleanup_task = asyncio.create_task(periodic_cleanup())
        logger.info(f"Started periodic job cleanup (interval: {settings.job_cleanup_interval_seconds}s)")
        
        yield
    finally:
        # Shutdown
        if cleanup_task:
            cleanup_task.cancel()
            try:
                await cleanup_task
            except asyncio.CancelledError:
                pass
        await whisper_service.cleanup()


# Create FastAPI application
app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    lifespan=lifespan
)

# Add CORS middleware
_loopback_origin_regex = r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$"
_cors_mw_kwargs: dict[str, object] = {
    "allow_origins": settings.cors_origins,
    "allow_credentials": True,
    "allow_methods": ["*"],
    "allow_headers": ["*"],
    "expose_headers": ["X-Correlation-ID"],
}
if settings.cors_allow_loopback_regex:
    _cors_mw_kwargs["allow_origin_regex"] = _loopback_origin_regex

app.add_middleware(CORSMiddleware, **_cors_mw_kwargs)


@app.middleware("http")
async def correlation_middleware(request, call_next):
    correlation_id = get_correlation_id()
    request.state.correlation_id = correlation_id
    response = await call_next(request)
    response.headers["X-Correlation-ID"] = correlation_id
    return response


# Global exception handler
@app.exception_handler(WhisperrrException)
async def whisperrr_exception_handler(request: Request, exc: WhisperrrException):
    """Handle custom Whisperrr exceptions."""
    correlation_id = getattr(request.state, 'correlation_id', None)
    
    status_codes = {
        "INVALID_AUDIO_FORMAT": 400, "FILE_TOO_LARGE": 413, "MODEL_NOT_LOADED": 503,
        "TRANSCRIPTION_FAILED": 500, "MODEL_LOAD_FAILED": 500,
        "AUDIO_PROCESSING_ERROR": 400, "FILE_SYSTEM_ERROR": 500
    }
    
    error_response = ErrorResponse(
        error_type=exc.error_code or "WHISPERRR_ERROR",
        message=exc.message,
        details=exc.details,
        correlation_id=correlation_id
    )
    
    return JSONResponse(
        status_code=status_codes.get(exc.error_code, 500),
        content=error_response.model_dump(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions (details logged server-side only)."""
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


# API Endpoints

@app.post("/transcribe", response_model=TranscriptionResponse)
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


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if whisper_service.is_model_loaded() else "degraded",
        model_loaded=whisper_service.is_model_loaded(),
        model_size=whisper_service.get_current_model_size(),
        uptime=round(whisper_service.get_uptime(), 2)
    )


@app.post("/jobs/submit", response_model=JobSubmissionResponse)
async def submit_transcription_job(
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
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """Submit a transcription job for asynchronous processing."""
    correlation_id = getattr(request.state, "correlation_id", None)

    job = job_manager.create_job()

    try:
        temp_file_path = await validate_and_materialize_upload(file)

        background_tasks.add_task(
            process_transcription_job,
            job.job_id,
            temp_file_path,
            model_size,
            temperature,
            language,
            task,
        )

        return JobSubmissionResponse(
            job_id=job.job_id,
            status=JobStatus.PENDING.value,
            message="Job submitted successfully",
        )
    except HTTPException:
        job_manager.delete_job(job.job_id)
        raise
    except Exception as e:
        job_manager.delete_job(job.job_id)
        logger.error(
            f"Error submitting job: {type(e).__name__}: {str(e)}",
            exc_info=True,
            extra={"correlation_id": correlation_id},
        )
        raise HTTPException(status_code=500, detail=f"Failed to submit job: {str(e)}")


@app.get("/jobs/{job_id}/progress", response_model=JobProgressResponse)
async def get_job_progress(job_id: str):
    """Get progress of a transcription job."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_dict = job.to_dict()
    return JobProgressResponse(**job_dict)








if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "app.main:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=settings.server_reload,
        log_level=settings.log_level.lower()
    )
