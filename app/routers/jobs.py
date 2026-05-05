import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Query, Request, UploadFile

from ..api.dependencies import RequireWhisperModel, validate_and_materialize_upload
from ..application.transcription_jobs import process_transcription_job
from ..job_manager import JobStatus, job_manager
from ..models import JobProgressResponse, JobSubmissionResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.post("/submit", response_model=JobSubmissionResponse)
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


@router.get("/{job_id}/progress", response_model=JobProgressResponse)
async def get_job_progress(job_id: str):
    """Get progress of a transcription job."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    job_dict = job.to_dict()
    return JobProgressResponse(**job_dict)
