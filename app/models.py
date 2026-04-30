"""Pydantic data models for the Whisperrr FastAPI service."""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator


class TranscriptionRequest(BaseModel):
    """Request model for transcription parameters."""
    
    model_size: Optional[str] = Field(
        default=None,
        description="Whisper model size (tiny, base, small, medium, large)"
    )
    language: Optional[str] = Field(
        default=None,
        description="Language hint for transcription (ISO 639-1 code)"
    )
    temperature: Optional[float] = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Temperature for sampling (0.0 = deterministic)"
    )
    
    @validator("model_size")
    def validate_model_size(cls, v):
        """Validate model size if provided."""
        if v is not None:
            valid_sizes = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
            if v not in valid_sizes:
                raise ValueError(f"Model size must be one of: {valid_sizes}")
        return v
    
    @validator("language")
    def validate_language(cls, v):
        """Validate language code if provided."""
        if v is not None and len(v) != 2:
            raise ValueError("Language must be a 2-character ISO 639-1 code")
        return v


class TranscriptionSegment(BaseModel):
    """Individual segment of transcription with timing information."""
    
    start_time: float = Field(description="Start time in seconds")
    end_time: float = Field(description="End time in seconds")
    text: str = Field(description="Transcribed text for this segment")
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence score for this segment"
    )
    
    @validator("end_time")
    def validate_end_time(cls, v, values):
        """Ensure end time is after start time."""
        if "start_time" in values and v <= values["start_time"]:
            raise ValueError("End time must be after start time")
        return v


class TranscriptionResponse(BaseModel):
    """Response model for transcription results."""
    
    text: str = Field(description="Full transcribed text")
    language: Optional[str] = Field(description="Detected language")
    duration: float = Field(description="Audio duration in seconds")
    segments: List[TranscriptionSegment] = Field(
        description="Individual segments with timing information"
    )
    confidence_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Overall confidence score"
    )
    model_used: str = Field(description="Whisper model size used")
    processing_time: float = Field(description="Processing time in seconds")


class ModelInfoResponse(BaseModel):
    """Response model for model information."""
    
    model_size: str = Field(description="Current model size")
    memory_usage_mb: float = Field(description="Model memory usage in MB")
    load_time_seconds: float = Field(description="Model load time in seconds")
    supported_languages: List[str] = Field(description="Supported languages")
    is_loaded: bool = Field(description="Whether model is currently loaded")
    last_loaded: Optional[datetime] = Field(description="When model was last loaded")


class HealthResponse(BaseModel):
    """Response model for health check."""
    
    status: str = Field(description="Service status")
    model_loaded: bool = Field(description="Whether model is loaded")
    model_size: Optional[str] = Field(description="Current model size")
    uptime: float = Field(description="Service uptime in seconds")


class ErrorResponse(BaseModel):
    """Response model for error responses."""
    
    error_type: str = Field(description="Type of error")
    message: str = Field(description="Error message")
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional error details"
    )
    correlation_id: Optional[str] = Field(
        default=None,
        description="Request correlation ID"
    )
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class JobSubmissionResponse(BaseModel):
    """Response model for job submission."""
    
    job_id: str = Field(description="Unique job identifier")
    status: str = Field(description="Initial job status")
    message: str = Field(description="Status message")


class JobProgressResponse(BaseModel):
    """Response model for job progress."""
    
    job_id: str = Field(description="Job identifier")
    status: str = Field(description="Current job status")
    progress: float = Field(description="Progress percentage (0-100)", ge=0.0, le=100.0)
    message: str = Field(description="Status message")
    result: Optional[TranscriptionResponse] = Field(
        default=None,
        description="Transcription result (available when status is COMPLETED)"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message (available when status is FAILED)"
    )
    created_at: str = Field(description="Job creation timestamp")
    updated_at: str = Field(description="Last update timestamp")




