"""Configuration management for the Whisperrr FastAPI service."""

import os
from typing import List, Union, Dict
from pydantic import validator, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Model configuration
    model_size: str = "base"
    max_file_size_mb: int = 50  # 50MB - demo site limit
    upload_dir: str = "/tmp/whisperrr_uploads"
    log_level: str = "INFO"
    
    # API configuration
    api_title: str = "Whisperrr Transcription Service"
    api_description: str = "Production-ready audio transcription using Faster Whisper"
    api_version: str = "1.0.0"
    cors_origins: Union[List[str], str] = ["http://localhost:7331", "http://localhost:3737", "http://127.0.0.1:7331", "http://127.0.0.1:3737"]
    
    # Processing configuration
    max_concurrent_transcriptions: int = 3
    request_timeout_seconds: int = 300
    cleanup_temp_files: bool = True
    
    # Performance and monitoring
    enable_metrics: bool = True
    enable_health_checks: bool = True
    
    # Performance tuning configuration
    # Uvicorn worker count (set via UVICORN_WORKERS env var, default calculated as (2 * CPU_COUNT) + 1)
    uvicorn_workers: int = 4
    # Compute type for Whisper model (int8, float16, float32)
    # int8: Fastest on CPU, lower accuracy
    # float32: Best accuracy, slower on CPU
    # float16: GPU only, good balance
    compute_type: str = "int8"  # Can be overridden via COMPUTE_TYPE env var
    # Thread count for NumPy/OpenMP operations (set via OMP_NUM_THREADS env var)
    num_threads: int = 4
    
    # Supported audio formats (including video formats that will be converted)
    supported_formats: List[str] = [
        # Audio formats
        "mp3", "wav", "m4a", "flac", "ogg", "wma", "aac",
        # Video formats (will be converted to audio)
        "mp4", "avi", "mov", "mkv", "flv", "webm", "wmv", "m4v", "3gp"
    ]
    
    # Formats that require conversion (video/music formats)
    formats_requiring_conversion: List[str] = [
        "mp4", "avi", "mov", "mkv", "flv", "webm", "wmv", "m4v", "3gp", "aac"
    ]
    
    # Model size options
    available_model_sizes: List[str] = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
    
    # Transcription configuration
    beam_size: int = 5  # Beam size for transcription
    default_task: str = "translate"  # Transcription task type
    target_sample_rate: int = 16000  # Target sample rate in Hz
    audio_channels: int = 1  # Mono audio
    
    # FFmpeg/FFprobe configuration
    ffprobe_timeout_seconds: int = 10  # FFprobe timeout
    ffmpeg_conversion_timeout_seconds: int = 300  # FFmpeg conversion timeout (5 minutes)
    
    # Progress calculation
    preprocessing_progress_max: float = 40.0  # Maximum progress for preprocessing phase
    transcription_progress_min: float = 40.0  # Minimum progress for transcription phase
    transcription_progress_max: float = 100.0  # Maximum progress for transcription phase
    segment_progress_base: float = 10.0  # Base progress for segments
    segment_progress_multiplier: float = 1.5  # Progress multiplier per segment
    segment_progress_max: float = 90.0  # Maximum segment progress
    
    # Job management
    job_cleanup_max_age_seconds: int = 7200  # Job cleanup max age: 2 hours (must be > MAX_JOB_DURATION)
    job_cleanup_interval_seconds: int = 300  # Run cleanup every 5 minutes
    
    # Server configuration (for development)
    server_host: str = "0.0.0.0"
    server_port: int = 5001
    server_reload: bool = True  # Auto-reload for development
    
    # Model descriptions
    model_descriptions: Dict[str, str] = {
        "tiny": "Fastest, least accurate (39 MB)",
        "base": "Good balance of speed and accuracy (74 MB)",
        "small": "Better accuracy, slower (244 MB)",
        "medium": "Good accuracy, slower (769 MB)",
        "large": "Best accuracy, slowest (1550 MB)",
        "large-v2": "Best accuracy, slowest (1550 MB)",
        "large-v3": "Best accuracy, slowest (1550 MB)"
    }
    
    # Supported languages (Whisper supports 99 languages)
    supported_languages: List[str] = [
        "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr",
        "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi",
        "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no",
        "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk",
        "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk",
        "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
        "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc",
        "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
        "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl",
        "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su"
    ]
    
    class Config:
        # Read from .env file in the python-service directory, then environment variables
        # This makes config.py the single source of truth for defaults
        # Environment variables override .env file values
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    @validator("model_size")
    def validate_model_size(cls, v):
        """Validate model size is supported."""
        valid_sizes = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
        if v not in valid_sizes:
            raise ValueError(f"Model size must be one of: {valid_sizes}")
        return v
    
    @validator("log_level")
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()
    
    @validator("max_file_size_mb")
    def validate_max_file_size(cls, v):
        """Validate max file size is reasonable."""
        if v <= 0:
            raise ValueError("Max file size must be greater than 0 MB")
        return v
    
    @validator("compute_type")
    def validate_compute_type(cls, v):
        """Validate compute type is supported."""
        valid_types = ["int8", "float16", "float32"]
        if v.lower() not in valid_types:
            raise ValueError(f"Compute type must be one of: {valid_types}")
        return v.lower()
    
    @validator("uvicorn_workers")
    def validate_uvicorn_workers(cls, v):
        """Validate uvicorn worker count is reasonable."""
        if v < 1:
            raise ValueError("Uvicorn workers must be at least 1")
        if v > 32:
            raise ValueError("Uvicorn workers should not exceed 32")
        return v
    
    @validator("num_threads")
    def validate_num_threads(cls, v):
        """Validate thread count is reasonable."""
        if v < 1:
            raise ValueError("Thread count must be at least 1")
        if v > 64:
            raise ValueError("Thread count should not exceed 64")
        return v
    
    @validator("upload_dir")
    def validate_upload_dir(cls, v):
        """Ensure upload directory exists."""
        os.makedirs(v, exist_ok=True)
        return v
    
    @validator("cors_origins", pre=True)
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            # Handle comma-separated string from environment variables
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v
    
    @property
    def max_file_size_bytes(self) -> int:
        """Get max file size in bytes."""
        return self.max_file_size_mb * 1024 * 1024
    
    @property
    def supported_formats_set(self) -> set:
        """Get supported formats as a set for faster lookup."""
        return set(self.supported_formats)
    
    @property
    def formats_requiring_conversion_set(self) -> set:
        """Get formats requiring conversion as a set for faster lookup."""
        return set(self.formats_requiring_conversion)


# Global settings instance
settings = Settings()
