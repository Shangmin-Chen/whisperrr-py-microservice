"""Job Manager for tracking asynchronous transcription jobs."""

import uuid
import threading
from datetime import datetime, timedelta
from typing import Optional, Dict
from enum import Enum
from .models import TranscriptionResponse


class JobStatus(str, Enum):
    """Status of a transcription job."""
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class Job:
    """Represents a single transcription job."""
    
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.status = JobStatus.PENDING
        self.progress = 0.0
        self.message = "Job created"
        self.result: Optional[TranscriptionResponse] = None
        self.error: Optional[str] = None
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self._lock = threading.Lock()
    
    def update_progress(self, progress: float, message: Optional[str] = None):
        """Update job progress (0-100)."""
        with self._lock:
            self.progress = max(0.0, min(100.0, progress))
            if message:
                self.message = message
            self.updated_at = datetime.utcnow()
    
    def set_status(self, status: JobStatus, message: Optional[str] = None):
        """Set job status."""
        with self._lock:
            self.status = status
            if message:
                self.message = message
            self.updated_at = datetime.utcnow()
    
    def set_result(self, result: TranscriptionResponse):
        """Set job result and mark as completed."""
        with self._lock:
            self.result = result
            self.status = JobStatus.COMPLETED
            self.progress = 100.0
            self.message = "Transcription completed"
            self.updated_at = datetime.utcnow()
    
    def set_error(self, error: str):
        """Set job error and mark as failed."""
        with self._lock:
            self.error = error
            self.status = JobStatus.FAILED
            self.message = error
            self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> Dict:
        """Convert job to dictionary."""
        with self._lock:
            return {
                "job_id": self.job_id,
                "status": self.status.value,
                "progress": self.progress,
                "message": self.message,
                "result": self.result.dict() if self.result else None,
                "error": self.error,
                "created_at": self.created_at.isoformat(),
                "updated_at": self.updated_at.isoformat()
            }


class JobManager:
    """Manages transcription jobs."""
    
    def __init__(self):
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
    
    def create_job(self) -> Job:
        """Create a new job and return it."""
        job_id = str(uuid.uuid4())
        job = Job(job_id)
        with self._lock:
            self._jobs[job_id] = job
        return job
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by ID."""
        with self._lock:
            return self._jobs.get(job_id)
    
    def delete_job(self, job_id: str):
        """Delete a job."""
        with self._lock:
            self._jobs.pop(job_id, None)
    
    def cleanup_old_jobs(self, max_age_seconds: Optional[int] = None):
        """Remove jobs older than max_age_seconds."""
        from .config import settings
        
        if max_age_seconds is None:
            max_age_seconds = settings.job_cleanup_max_age_seconds
        
        cutoff = datetime.utcnow() - timedelta(seconds=max_age_seconds)
        with self._lock:
            to_delete = [
                job_id for job_id, job in self._jobs.items()
                if job.created_at < cutoff
            ]
            for job_id in to_delete:
                del self._jobs[job_id]


# Global job manager instance
job_manager = JobManager()

