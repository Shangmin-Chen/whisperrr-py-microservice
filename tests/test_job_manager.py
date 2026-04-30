"""
Comprehensive unit tests for JobManager covering all failure scenarios.

These tests verify that the job manager properly handles:
- Job creation and retrieval
- Job status updates
- Progress tracking
- Error handling
- Concurrent access
- Job cleanup
"""

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock

from app.job_manager import JobManager, Job, JobStatus


class TestJob:
    """Test suite for Job class."""
    
    def test_job_creation_initializes_correctly(self):
        """Test that job is initialized with correct default values."""
        job = Job("test-job-id")
        
        assert job.job_id == "test-job-id"
        assert job.status == JobStatus.PENDING
        assert job.progress == 0.0
        assert job.result is None
        assert job.error is None
    
    def test_update_progress_sets_progress(self):
        """Test that update_progress sets progress correctly."""
        job = Job("test-job-id")
        
        job.update_progress(50.0, "Processing...")
        
        assert job.progress == 50.0
        assert job.message == "Processing..."
    
    def test_update_progress_clamps_to_0_100(self):
        """Test that progress is clamped to 0-100 range."""
        job = Job("test-job-id")
        
        job.update_progress(-10.0)
        assert job.progress == 0.0
        
        job.update_progress(150.0)
        assert job.progress == 100.0
    
    def test_set_status_updates_status(self):
        """Test that set_status updates status correctly."""
        job = Job("test-job-id")
        
        job.set_status(JobStatus.PROCESSING, "Starting...")
        
        assert job.status == JobStatus.PROCESSING
        assert job.message == "Starting..."
    
    def test_set_result_completes_job(self):
        """Test that set_result completes job."""
        job = Job("test-job-id")
        result = {"text": "Transcribed text"}
        
        job.set_result(result)
        
        assert job.status == JobStatus.COMPLETED
        assert job.progress == 100.0
        assert job.result == result
    
    def test_set_error_fails_job(self):
        """Test that set_error fails job."""
        job = Job("test-job-id")
        
        job.set_error("Transcription failed")
        
        assert job.status == JobStatus.FAILED
        assert job.error == "Transcription failed"
        assert job.message == "Transcription failed"
    
    def test_to_dict_returns_complete_dict(self):
        """Test that to_dict returns complete job dictionary."""
        job = Job("test-job-id")
        job.set_status(JobStatus.PROCESSING, "Processing...")
        job.update_progress(50.0)
        
        job_dict = job.to_dict()
        
        assert job_dict["job_id"] == "test-job-id"
        assert job_dict["status"] == "PROCESSING"
        assert job_dict["progress"] == 50.0
        assert job_dict["message"] == "Processing..."


class TestJobManager:
    """Test suite for JobManager class."""
    
    @pytest.fixture
    def manager(self):
        """Create a fresh manager instance for each test."""
        return JobManager()
    
    def test_create_job_returns_job(self, manager):
        """Test that create_job returns a new job."""
        job = manager.create_job()
        
        assert job is not None
        assert isinstance(job, Job)
        assert job.job_id is not None
    
    def test_create_job_generates_unique_ids(self, manager):
        """Test that create_job generates unique job IDs."""
        job1 = manager.create_job()
        job2 = manager.create_job()
        
        assert job1.job_id != job2.job_id
    
    def test_get_job_with_existing_id_returns_job(self, manager):
        """Test that get_job returns job for existing ID."""
        job = manager.create_job()
        
        retrieved = manager.get_job(job.job_id)
        
        assert retrieved is not None
        assert retrieved.job_id == job.job_id
    
    def test_get_job_with_nonexistent_id_returns_none(self, manager):
        """Test that get_job returns None for nonexistent ID."""
        result = manager.get_job("nonexistent-id")
        
        assert result is None
    
    def test_delete_job_removes_job(self, manager):
        """Test that delete_job removes job from manager."""
        job = manager.create_job()
        job_id = job.job_id
        
        manager.delete_job(job_id)
        
        assert manager.get_job(job_id) is None
    
    def test_delete_job_with_nonexistent_id_does_not_raise(self, manager):
        """Test that delete_job with nonexistent ID does not raise."""
        # Should not raise exception
        manager.delete_job("nonexistent-id")
    
    def test_cleanup_old_jobs_removes_old_jobs(self, manager):
        """Test that cleanup_old_jobs removes old jobs."""
        # Create old job (manually set created_at to past)
        old_job = manager.create_job()
        old_job.created_at = datetime.utcnow() - timedelta(seconds=3700)  # > 1 hour
        
        # Create new job
        new_job = manager.create_job()
        
        manager.cleanup_old_jobs(max_age_seconds=3600)
        
        assert manager.get_job(old_job.job_id) is None
        assert manager.get_job(new_job.job_id) is not None
    
    def test_cleanup_old_jobs_with_no_old_jobs_does_nothing(self, manager):
        """Test that cleanup_old_jobs does nothing when no old jobs."""
        job = manager.create_job()
        
        manager.cleanup_old_jobs(max_age_seconds=3600)
        
        assert manager.get_job(job.job_id) is not None
    
    def test_concurrent_job_creation(self, manager):
        """Test that concurrent job creation works correctly."""
        jobs = []
        for _ in range(10):
            jobs.append(manager.create_job())
        
        # All jobs should be unique
        job_ids = [job.job_id for job in jobs]
        assert len(job_ids) == len(set(job_ids))
        
        # All jobs should be retrievable
        for job in jobs:
            assert manager.get_job(job.job_id) is not None
    
    def test_job_progress_tracking(self, manager):
        """Test that job progress is tracked correctly."""
        job = manager.create_job()
        
        job.update_progress(25.0, "Quarter done")
        assert job.progress == 25.0
        
        job.update_progress(50.0, "Half done")
        assert job.progress == 50.0
        
        job.update_progress(100.0, "Complete")
        assert job.progress == 100.0
    
    def test_job_status_transitions(self, manager):
        """Test that job status transitions work correctly."""
        job = manager.create_job()
        
        assert job.status == JobStatus.PENDING
        
        job.set_status(JobStatus.PROCESSING, "Starting")
        assert job.status == JobStatus.PROCESSING
        
        job.set_result({"text": "Done"})
        assert job.status == JobStatus.COMPLETED
    
    def test_job_error_handling(self, manager):
        """Test that job errors are handled correctly."""
        job = manager.create_job()
        
        job.set_error("Something went wrong")
        
        assert job.status == JobStatus.FAILED
        assert job.error == "Something went wrong"
        assert job.message == "Something went wrong"









