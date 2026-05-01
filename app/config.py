"""Configuration management for the Whisperrr FastAPI service."""

import os
from typing import List, Union, Dict

from pydantic import field_validator, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Model configuration
    model_size: str = "base"
    max_file_size_mb: int = 50  # 50MB - demo site limit
    upload_dir: str = "/tmp/whisperrr_uploads"
    log_level: str = "INFO"

    # API configuration
    api_title: str = "Whisperrr Transcription Service"
    api_description: str = "Production-ready audio transcription using Faster Whisper"
    api_version: str = "1.0.0"
    cors_origins: Union[List[str], str] = [
        "http://localhost:7331",
        "http://localhost:3737",
        "http://127.0.0.1:7331",
        "http://127.0.0.1:3737",
    ]

    # Processing configuration
    max_concurrent_transcriptions: int = 3
    request_timeout_seconds: int = 300
    cleanup_temp_files: bool = True

    # Performance and monitoring
    enable_metrics: bool = True
    enable_health_checks: bool = True

    uvicorn_workers: int = 4
    compute_type: str = "int8"
    num_threads: int = 4

    supported_formats: List[str] = [
        "mp3",
        "wav",
        "m4a",
        "flac",
        "ogg",
        "wma",
        "aac",
        "mp4",
        "avi",
        "mov",
        "mkv",
        "flv",
        "webm",
        "wmv",
        "m4v",
        "3gp",
    ]

    formats_requiring_conversion: List[str] = [
        "mp4",
        "avi",
        "mov",
        "mkv",
        "flv",
        "webm",
        "wmv",
        "m4v",
        "3gp",
        "aac",
    ]

    available_model_sizes: List[str] = [
        "tiny",
        "base",
        "small",
        "medium",
        "large",
        "large-v2",
        "large-v3",
    ]

    beam_size: int = 5
    default_task: str = "translate"
    target_sample_rate: int = 16000
    audio_channels: int = 1

    ffprobe_timeout_seconds: int = 10
    ffmpeg_conversion_timeout_seconds: int = 300

    preprocessing_progress_max: float = 40.0
    transcription_progress_min: float = 40.0
    transcription_progress_max: float = 100.0
    segment_progress_base: float = 10.0
    segment_progress_multiplier: float = 1.5
    segment_progress_max: float = 90.0

    job_cleanup_max_age_seconds: int = 7200
    job_cleanup_interval_seconds: int = 300

    server_host: str = "0.0.0.0"
    server_port: int = 5001
    server_reload: bool = True

    cors_allow_loopback_regex: bool = Field(
        default=True,
        description="Allow any loopback Origin (localhost / 127.0.0.1 / [::1], any port) via regex.",
    )

    model_descriptions: Dict[str, str] = {
        "tiny": "Fastest, least accurate (39 MB)",
        "base": "Good balance of speed and accuracy (74 MB)",
        "small": "Better accuracy, slower (244 MB)",
        "medium": "Good accuracy, slower (769 MB)",
        "large": "Best accuracy, slowest (1550 MB)",
        "large-v2": "Best accuracy, slowest (1550 MB)",
        "large-v3": "Best accuracy, slowest (1550 MB)",
    }

    supported_languages: List[str] = [
        "en",
        "zh",
        "de",
        "es",
        "ru",
        "ko",
        "fr",
        "ja",
        "pt",
        "tr",
        "pl",
        "ca",
        "nl",
        "ar",
        "sv",
        "it",
        "id",
        "hi",
        "fi",
        "vi",
        "he",
        "uk",
        "el",
        "ms",
        "cs",
        "ro",
        "da",
        "hu",
        "ta",
        "no",
        "th",
        "ur",
        "hr",
        "bg",
        "lt",
        "la",
        "mi",
        "ml",
        "cy",
        "sk",
        "te",
        "fa",
        "lv",
        "bn",
        "sr",
        "az",
        "sl",
        "kn",
        "et",
        "mk",
        "br",
        "eu",
        "is",
        "hy",
        "ne",
        "mn",
        "bs",
        "kk",
        "sq",
        "sw",
        "gl",
        "mr",
        "pa",
        "si",
        "km",
        "sn",
        "yo",
        "so",
        "af",
        "oc",
        "ka",
        "be",
        "tg",
        "sd",
        "gu",
        "am",
        "yi",
        "lo",
        "uz",
        "fo",
        "ht",
        "ps",
        "tk",
        "nn",
        "mt",
        "sa",
        "lb",
        "my",
        "bo",
        "tl",
        "mg",
        "as",
        "tt",
        "haw",
        "ln",
        "ha",
        "ba",
        "jw",
        "su",
    ]

    @field_validator("model_size")
    @classmethod
    def validate_model_size(cls, v: str) -> str:
        valid_sizes = [
            "tiny",
            "base",
            "small",
            "medium",
            "large",
            "large-v2",
            "large-v3",
        ]
        if v not in valid_sizes:
            raise ValueError(f"Model size must be one of: {valid_sizes}")
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()

    @field_validator("max_file_size_mb")
    @classmethod
    def validate_max_file_size(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Max file size must be greater than 0 MB")
        return v

    @field_validator("compute_type")
    @classmethod
    def validate_compute_type(cls, v: str) -> str:
        valid_types = ["int8", "float16", "float32"]
        if v.lower() not in valid_types:
            raise ValueError(f"Compute type must be one of: {valid_types}")
        return v.lower()

    @field_validator("uvicorn_workers")
    @classmethod
    def validate_uvicorn_workers(cls, v: int) -> int:
        if v < 1:
            raise ValueError("Uvicorn workers must be at least 1")
        if v > 32:
            raise ValueError("Uvicorn workers should not exceed 32")
        return v

    @field_validator("num_threads")
    @classmethod
    def validate_num_threads(cls, v: int) -> int:
        if v < 1:
            raise ValueError("Thread count must be at least 1")
        if v > 64:
            raise ValueError("Thread count should not exceed 64")
        return v

    @field_validator("upload_dir")
    @classmethod
    def validate_upload_dir(cls, v: str) -> str:
        os.makedirs(v, exist_ok=True)
        return v

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    @property
    def supported_formats_set(self) -> set:
        return set(self.supported_formats)

    @property
    def formats_requiring_conversion_set(self) -> set:
        return set(self.formats_requiring_conversion)


settings = Settings()
