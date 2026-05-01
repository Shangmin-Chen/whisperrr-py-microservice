"""High-level validation of paths before whisper_service runs."""

import os
from typing import Any, Dict

from ..config import settings
from ..exceptions import (
    AudioProcessingError,
    FileSystemError,
    FileTooLarge,
    InvalidAudioFormat,
)

from .audio_info import get_audio_info
from .files import get_file_extension, validate_file_size
from .format_detection import detect_audio_format


def validate_audio_file(file_path: str) -> Dict[str, Any]:
    """Comprehensive audio file validation."""
    try:
        if not os.path.exists(file_path):
            raise FileSystemError(
                message="File does not exist",
                file_path=file_path,
            )

        file_size = os.path.getsize(file_path)

        if not validate_file_size(file_size):
            raise FileTooLarge(
                file_size=file_size,
                max_size=settings.max_file_size_bytes,
            )

        detected_format = detect_audio_format(file_path)
        extension = get_file_extension(file_path)
        format_to_check = detected_format or extension

        if not format_to_check or format_to_check not in settings.supported_formats_set:
            raise InvalidAudioFormat(
                file_format=format_to_check or "unknown",
                supported_formats=settings.supported_formats,
            )

        is_video_format = detected_format in settings.formats_requiring_conversion_set

        if is_video_format:
            return {
                "valid": True,
                "format": detected_format,
                "file_size": file_size,
                "file_size_mb": round(file_size / (1024 * 1024), 2),
                "duration": None,
                "sample_rate": None,
            }

        try:
            audio_info = get_audio_info(file_path)
            return {
                "valid": True,
                "format": detected_format,
                "file_size": file_size,
                "file_size_mb": round(file_size / (1024 * 1024), 2),
                "duration": audio_info.get("duration"),
                "sample_rate": audio_info.get("sample_rate"),
            }
        except Exception:
            return {
                "valid": True,
                "format": detected_format,
                "file_size": file_size,
                "file_size_mb": round(file_size / (1024 * 1024), 2),
                "duration": None,
                "sample_rate": None,
            }

    except (InvalidAudioFormat, FileTooLarge, FileSystemError):
        raise
    except Exception as e:
        raise AudioProcessingError(
            message="Audio file validation failed",
            original_error=str(e),
            processing_step="validate_audio_file",
        )
