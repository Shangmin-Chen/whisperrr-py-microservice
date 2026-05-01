"""Background transcription jobs (async JobManager-backed processing)."""

import logging
import os
from typing import Optional

from ..job_manager import job_manager, JobStatus
from ..whisper_service import whisper_service

logger = logging.getLogger(__name__)


async def process_transcription_job(
    job_id: str,
    file_path: str,
    model_size: Optional[str],
    temperature: float,
    language: Optional[str] = None,
    task: Optional[str] = None,
) -> None:
    """Run one async job: Whisper with progress updates into JobManager."""
    job = job_manager.get_job(job_id)
    if not job:
        return

    try:
        job.set_status(JobStatus.PROCESSING, "Starting transcription...")

        def progress_callback(progress: float, message: str) -> None:
            job.update_progress(progress, message)

        result = await whisper_service.transcribe_audio(
            file_path=file_path,
            model_size=model_size,
            language=language,
            temperature=temperature,
            task=task,
            progress_callback=progress_callback,
        )

        job.set_result(result)
    except Exception as e:
        logger.error("Job %s failed: %s", job_id, str(e), exc_info=True)
        job.set_error(f"Transcription failed: {str(e)}")
    finally:
        if os.path.exists(file_path):
            try:
                os.unlink(file_path)
            except Exception:
                pass
